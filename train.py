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

# Local module imports from the 'src' directory
from src.dataset import WavReverbDataset, split_dataset
from src.losses import CompositeLoss
from src.utils.optimizers import create_optimizer

# Import all models to create a registry
from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM

MODEL_REGISTRY = {
    "CONV": CONV, "CONV_SSM_L": CONV_SSM_L, "GCN": GCN, "GCN_SSM": GCN_SSM,
}

def train(config, run_name, notes, resume_id=None):
    
    # 1. SETUP
    
    train_cfg = config['training'] # Use a shorthand for clarity
    data_cfg = config['data']

    project_name = train_cfg.get('project_name', 'spring-reverb-emulation')
    if resume_id:
        run = wandb.init(project=project_name, id=resume_id, resume="must")
    else:
        run = wandb.init(project=project_name, name=run_name, notes=notes, config=config)
    
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
        load_in_memory=data_cfg.get('load_in_memory', False) # Pass the new flag
    )
    train_set, val_set, _ = split_dataset(full_dataset)
    
    train_loader = DataLoader(
        train_set, batch_size=train_cfg['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=train_cfg['batch_size'])
    print(f"Data loaded: {len(train_set)} training samples, {len(val_set)} validation samples.")

    
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
    loss_fn = CompositeLoss(
        device=device, sampling_rate=data_cfg['sample_rate'], **config.get('loss', {})
    )
    
    
    # 4. RESUME FROM CHECKPOINT (if applicable)
    
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
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [TRAIN]", leave=False)
        for dry_wav, wet_wav in pbar:
            dry_wav, wet_wav = dry_wav.to(device), wet_wav.to(device)
            optimizer.zero_grad()
            pred_wav = model(dry_wav)
            loss, _, _, _, _, _ = loss_fn(pred_wav, wet_wav)
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
                loss, _, _, _, _, _ = loss_fn(pred_wav, wet_wav)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # --- Logging and Checkpointing ---
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

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
    parser = argparse.ArgumentParser(description="Train a spring reverb emulation model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the model config YAML file.')
    parser.add_argument('--run_name', type=str, required=True, help='A descriptive name for the Weights & Biases run.')
    parser.add_argument('--notes', type=str, default="", help='Optional longer description for the run.')
    parser.add_argument('--resume_id', type=str, help='Optional wandb run ID to resume a previous training run.')
    args = parser.parse_args()

    # --- Load Base Config ---
    try:
        with open('configs/base.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: configs/base.yaml not found. Please create it.")
        exit()
        
    # --- Load Model-Specific Config and Merge ---
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    final_config = merge_configs(base_config, model_config)
    
    print("--- Final Merged Configuration ---")
    print(yaml.dump(final_config, default_flow_style=False, sort_keys=False))
    print("----------------------------------")

    train(final_config, args.run_name, args.notes, args.resume_id)