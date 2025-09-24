# train.py
import os
import argparse
import yaml
from pathlib import Path
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


from src.dataset import WavReverbDataset, split_dataset
from src.losses import CompositeLoss
from src.utils.optimizers import create_optimizer
import soundfile as sf

from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM


MODEL_REGISTRY = {
"CONV": CONV,
"CONV_SSM_L": CONV_SSM_L,
"GCN": GCN,
"GCN_SSM": GCN_SSM,

}

def train(config, run_name,  resume_id=None):
    
    # 1. SETUP
    
    train_cfg = config['training']
    data_cfg = config['data']

    project_name = train_cfg.get('project_name', 'spring-reverb-emulation')
    if resume_id:
        run = wandb.init(project=project_name, id=resume_id, resume="must")
    else:
        run = wandb.init(project=project_name, name=run_name,  config=config)
    
    output_dir = Path(config.get('output_dir', 'training_runs')) / run.name
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_path = data_cfg.get('dry_path') or data_cfg.get('dry_dir')
    wet_path = data_cfg.get('wet_path') or data_cfg.get('wet_dir')
    
    if not dry_path or not wet_path:
        raise KeyError("Config file must contain 'data.dry_path' and 'data.wet_path'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # 2. DATA LOADING
    
    print("Loading data...")
    data_cfg = config['data']
    full_dataset = WavReverbDataset(
        dry_path=data_cfg['dry_path'],
        wet_path=data_cfg['wet_path'],
        sample_rate=data_cfg['sample_rate'],
        duration=data_cfg['duration'],
        load_in_memory=data_cfg.get('load_in_memory', False) 
    )
    train_set, val_set, test_set = split_dataset(full_dataset)
    
    train_loader = DataLoader(
        train_set, batch_size=train_cfg['batch_size'], shuffle=True,
        num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=train_cfg['batch_size'])
    test_loader_preview = DataLoader(test_set, batch_size=train_cfg['batch_size'])
    print(f"Data loaded: {len(train_set)} training samples, {len(val_set)} validation samples.")

    preview_cfg = train_cfg.get('previews', {})
    if preview_cfg.get('enable', False):
        print("Fetching a fixed test batch for audio previews...")
        try:
            # Use the new test loader
            fixed_dry_wav_test, fixed_wet_wav_test = next(iter(test_loader_preview))
            fixed_dry_wav_test, fixed_wet_wav_test = fixed_dry_wav_test.to(device), fixed_wet_wav_test.to(device)
            print(f"Preview batch fetched with shape: {fixed_dry_wav_test.shape}")
        except StopIteration:
            print("WARNING: Test loader is empty. Cannot generate audio previews.")
            preview_cfg['enable'] = False

    
    # 3. MODEL, OPTIMIZER, LOSS
    
    print("Initializing model, optimizer, and loss function...")
    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**config['model']['params']).to(device)
    wandb.watch(model, log="all", log_freq=250)
    print(f"Initialized model: {model_name}")

    optimizer = create_optimizer(model, train_cfg['optimizer'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=train_cfg['scheduler']['patience'],
        factor=train_cfg['scheduler']['factor']
    )
    warmup_cfg = train_cfg.get('warmup', {})
    warmup_epochs = warmup_cfg.get('epochs', 0)
    base_lr = train_cfg['optimizer']['learning_rate']
    loss_fn = CompositeLoss(
        device=device, sampling_rate=data_cfg['sample_rate'], **config.get('loss', {})
    )
    
    
    # 4. CHECKPOINT 
    
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_id:
        print(f"Attempting to resume from run ID: {resume_id}")
        try:
            artifact_path = f"{run.entity}/{run.project}/{run.name}_model:latest"
            artifact = run.use_artifact(artifact_path)
            artifact_dir = artifact.download()
            checkpoint_path = Path(artifact_dir) / "checkpoint_best.pth"
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_loss"]
            print(f"Successfully resumed from epoch {start_epoch} with best validation loss {best_val_loss:.6f}")
        except Exception as e:
            print(f"Could not resume from checkpoint. Starting from scratch. Error: {e}")

    
    # 5. TRAINING LOOP
    
    print("Starting training...")
    for epoch in range(start_epoch, train_cfg['epochs']):
        if epoch < warmup_epochs:
            
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Epoch {epoch+1}/{train_cfg['epochs']} [WARMUP] -> LR set to {warmup_lr:.6f}")
        elif epoch == warmup_epochs:

            print(f"Epoch {epoch+1}/{train_cfg['epochs']} [END WARMUP] -> LR set to base {base_lr:.6f}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [TRAIN]", leave=False)
        for dry_wav, wet_wav in pbar:
            dry_wav, wet_wav = dry_wav.to(device), wet_wav.to(device)
            optimizer.zero_grad()
            pred_wav = model(dry_wav)
            loss, _, _, _, _ = loss_fn(pred_wav, wet_wav)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for dry_wav, wet_wav in tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]", leave=False):
                dry_wav, wet_wav = dry_wav.to(device), wet_wav.to(device)
                pred_wav = model(dry_wav)
                loss, _, _, _, _= loss_fn(pred_wav, wet_wav)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # --- Logging and Checkpointing ---
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)



        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / "checkpoint_best.pth"
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_val_loss, 'config': config
            }, checkpoint_path)
            artifact = wandb.Artifact(f"{run.name}_model", type="model")
            artifact.add_file(checkpoint_path)
            run.log_artifact(artifact)
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
            
        if preview_cfg.get('enable', False) and (epoch + 1) % preview_cfg.get('every_n_epochs', 10) == 0:
            print("Generating audio preview from test set...")
            model.eval()
            with torch.no_grad():
                pred_wav_preview = model(fixed_dry_wav_test)

            num_to_log = min(preview_cfg.get('num_samples', 1), fixed_dry_wav_test.shape[0])
            
            audio_previews = []
            for i in range(num_to_log):
                pred_np = pred_wav_preview[i].squeeze().cpu().numpy()
                dry_np = fixed_dry_wav_test[i].squeeze().cpu().numpy()
                wet_np = fixed_wet_wav_test[i].squeeze().cpu().numpy()

                audio_previews.extend([
                    wandb.Audio(dry_np, caption=f"Epoch {epoch+1} Test Sample {i}: Dry Input", sample_rate=data_cfg['sample_rate']),
                    wandb.Audio(wet_np, caption=f"Epoch {epoch+1} Test Sample {i}: Wet Ground Truth", sample_rate=data_cfg['sample_rate']),
                    wandb.Audio(pred_np, caption=f"Epoch {epoch+1} Test Sample {i}: Model Prediction", sample_rate=data_cfg['sample_rate']),
                ])
            
            wandb.log({"audio_previews": audio_previews}, step=epoch + 1)
            print(f"Logged {num_to_log} audio preview(s) to wandb.")

    print("Training complete.")
    wandb.finish()


def merge_configs(base, update):
    merged = copy.deepcopy(base)
    for k, v in update.items():
        if k in merged and isinstance(merged.get(k), dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


if __name__ == "__main__":
    
    IS_SWEEP = 'WANDB_SWEEP_ID' in os.environ

    if not IS_SWEEP:

        print("Running in MANUAL mode.")
        parser = argparse.ArgumentParser(description="Train a spring reverb emulation model.")
        parser.add_argument('--config', type=str, required=True, help='Path to the model-specific config YAML file.')
        parser.add_argument('--run_name', type=str, required=True, help='A descriptive name for the Weights & Biases run.')
        parser.add_argument('--notes', type=str, default="", help='Optional longer description for the run.')
        parser.add_argument('--resume_id', type=str, help='Optional wandb run ID to resume a previous training run.')
        args = parser.parse_args()

        with open('configs/base.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        with open(args.config, 'r') as f:
            model_config_update = yaml.safe_load(f)
        
        final_config = merge_configs(base_config, model_config_update)
        run_name = args.run_name

        resume_id = args.resume_id

    else:
        # --- SWEEP MODE ---
        print("Running in SWEEP mode (wandb agent).")
        wandb.init()
        base_model_config_path = wandb.config.base_config
        with open(base_model_config_path, 'r') as f:
            base_model_config = yaml.safe_load(f)
        
        # 2. base config
        with open('configs/base.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        
        # 3. Merge configs
        final_config = merge_configs(base_config, base_model_config)

        sweep_params = dict(wandb.config)
        

        if 'combo' in sweep_params:
            for key, value in sweep_params['combo'].items():
                final_config['model']['params'][key] = value
        

        for key, value in sweep_params.items():
            if key != 'base_config' and key != 'combo':
                # model param or a training param
                if key in final_config['model']['params']:
                    final_config['model']['params'][key] = value
                elif key in final_config['training']['optimizer']:
                    final_config['training']['optimizer'][key] = value
                elif key in final_config['training']:
                    final_config['training'][key] = value
        
        # Create a dynamic run name
        run_name = f"{final_config['model']['name']}"
        # Add identifiers from the sweep params to the name
        run_name += f"_ks{final_config['model']['params'].get('kernel_size', 'N/A')}"
        run_name += f"_nl{final_config['model']['params'].get('num_layers') or final_config['model']['params'].get('n_blocks')}"
        run_name += f"_nc{final_config['model']['params'].get('hidden_channels') or final_config['model']['params'].get('n_channels')}"
        if 'd_state' in final_config['model']['params']:
             run_name += f"_ds{final_config['model']['params']['d_state']}"
        
        notes = f"Sweep run for {run_name}"
        resume_id = None


    print("\n--- Final Merged Configuration ---")
    print(yaml.dump(final_config, default_flow_style=False, sort_keys=False))
    print("----------------------------------\n")
    train(final_config, run_name,  resume_id)