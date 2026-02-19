"""
S5 Codec — Training Script

Standalone training loop for the SSM compression experiment.
Follows the same patterns as the main spring-ssm train.py
but adapted for the autoencoder / codec task.

Usage:
    python codec/train_codec.py --config codec/configs/phase1.yaml
    python codec/train_codec.py --config codec/configs/phase4.yaml --decoder learned
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codec.model import S5Codec
from codec.dataset import create_dataloaders, SyntheticSignalDataset
from codec.audio_dataset import create_wav_dataloaders
from codec.losses import CodecLoss, compute_snr, compute_spectral_snr

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train(config: dict):
    # ─── Setup ───
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(config.get('output_dir', 'codec/runs')) / config.get('run_name', 'default')
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = config.get('use_wandb', False) and HAS_WANDB
    if use_wandb:
        wandb.init(
            project=config.get('project_name', 'ssm-codec'),
            name=config.get('run_name', 'codec-experiment'),
            config=config,
        )

    # ─── Data ───
    data_cfg = config.get('data', {})
    segment_length = data_cfg.get('segment_length', 1600)
    
    # Check if using real audio or synthetic
    if 'wav_path' in data_cfg and data_cfg['wav_path']:
        print(f"Using real audio: {data_cfg['wav_path']}")
        train_loader, val_loader, test_loader = create_wav_dataloaders(
            wav_path=data_cfg['wav_path'],
            segment_length=segment_length,
            sample_rate=data_cfg.get('sample_rate', 16000),
            train_val_split=data_cfg.get('train_val_split', 0.9),
            batch_size=config['training']['batch_size'],
        )
    else:
        print(f"Using synthetic signals (phase {data_cfg.get('phase', 1)})")
        train_loader, val_loader, test_loader = create_dataloaders(
            n_train=data_cfg.get('n_train', 8000),
            n_val=data_cfg.get('n_val', 1000),
            n_test=data_cfg.get('n_test', 1000),
            segment_length=segment_length,
            sample_rate=data_cfg.get('sample_rate', 16000),
            phase=data_cfg.get('phase', 1),
            batch_size=config['training']['batch_size'],
        )
    
    print(f"Data: segment_length={segment_length}, "
          f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    # ─── Model ───
    model_cfg = config.get('model', {})
    model = S5Codec(
        d_state=model_cfg.get('d_state', 64),
        segment_length=segment_length,
        n_encoder_layers=model_cfg.get('n_encoder_layers', 2),
        n_decoder_layers=model_cfg.get('n_decoder_layers', 2),
        d_hidden=model_cfg.get('d_hidden', 16),
        n_vq_stages=model_cfg.get('n_vq_stages', 4),
        n_vq_codes=model_cfg.get('n_vq_codes', 512),
        commitment_cost=model_cfg.get('commitment_cost', 0.25),
        decoder_type=model_cfg.get('decoder_type', 'learned'),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bits_per_sample = model.quantizer.bits_per_segment() / segment_length
    print(f"Model: {n_params:,} parameters, "
          f"decoder={model_cfg.get('decoder_type', 'learned')}, "
          f"d_state={model_cfg.get('d_state', 64)}")
    print(f"Codec: {model.quantizer.bits_per_segment():.0f} bits/segment, "
          f"{bits_per_sample:.4f} bits/sample")

    # ─── Optimizer ───
    train_cfg = config['training']
    lr = train_cfg.get('learning_rate', 1e-3)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=train_cfg.get('scheduler_patience', 15),
        factor=train_cfg.get('scheduler_factor', 0.5),
    )

    # ─── Loss ───
    loss_cfg = config.get('loss', {})
    loss_fn = CodecLoss(
        alpha_spectral=loss_cfg.get('alpha_spectral', 0.0),  # default OFF for stability
        alpha_vq=loss_cfg.get('alpha_vq', 0.0),
        time_loss_type=loss_cfg.get('time_loss_type', 'mse'),
    )

    # ─── Training Loop ───
    epochs = train_cfg.get('epochs', 100)
    best_val_loss = float('inf')
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    quant_warmup_epochs = train_cfg.get('quant_warmup_epochs', 5)

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs")
    print(f"  Quantization warmup: first {quant_warmup_epochs} epochs bypass quantization")
    print(f"  LR warmup: first {warmup_epochs} epochs")
    print(f"  LR: {lr:.1e}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Bypass quantization during warmup
        bypass_quant = epoch < quant_warmup_epochs

        # ── Train ──
        model.train()
        train_metrics = {
            'loss_total': 0, 'loss_time': 0, 'loss_spectral': 0,
            'loss_vq': 0, 'snr': 0,
        }
        n_batches = 0

        phase_str = "Q-OFF" if bypass_quant else "Q-ON "
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{phase_str}]", leave=False)
        for x, target in pbar:
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x_hat, quant_loss, info = model(x, bypass_quantization=bypass_quant)
            loss, metrics = loss_fn(x_hat, target, quant_loss, model)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            snr = compute_snr(x_hat.detach(), target)
            for k, v in metrics.items():
                if k in train_metrics:
                    train_metrics[k] += v
                else:
                    train_metrics[k] = v
            train_metrics['snr'] += snr
            n_batches += 1

            pbar.set_postfix(
                loss=f"{metrics['loss_total']:.4f}",
                snr=f"{snr:.1f}dB",
                qdist=f"{info['quant_distortion']:.4f}",
            )

        # Average
        for k in train_metrics:
            train_metrics[k] /= n_batches

        # ── Validate ──
        model.eval()
        val_metrics = {
            'loss_total': 0, 'loss_time': 0, 'loss_spectral': 0,
            'loss_vq': 0, 'snr': 0, 'spectral_snr': 0,
        }
        n_val_batches = 0

        with torch.no_grad():
            for x, target in tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]", leave=False):
                x, target = x.to(device), target.to(device)
                x_hat, quant_loss, info = model(x, bypass_quantization=bypass_quant)
                _, metrics = loss_fn(x_hat, target, quant_loss, model)

                for k, v in metrics.items():
                    if k in val_metrics:
                        val_metrics[k] += v
                    else:
                        val_metrics[k] = v
                val_metrics['snr'] += compute_snr(x_hat, target)
                val_metrics['spectral_snr'] += compute_spectral_snr(x_hat, target)
                n_val_batches += 1

        for k in val_metrics:
            val_metrics[k] /= n_val_batches

        # ── LR Schedule ──
        if epoch >= warmup_epochs:
            scheduler.step(val_metrics['loss_total'])

        # ── Log ──
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_metrics['loss_total']:.4f} SNR: {train_metrics['snr']:.1f}dB | "
              f"Val Loss: {val_metrics['loss_total']:.4f} SNR: {val_metrics['snr']:.1f}dB "
              f"SpecSNR: {val_metrics['spectral_snr']:.1f}dB | "
              f"LR: {current_lr:.2e}")

        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'lr': current_lr,
                'bits_per_sample': bits_per_sample,
            }
            for k, v in train_metrics.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log_dict[f'val/{k}'] = v
            log_dict['val/quant_distortion'] = info['quant_distortion']
            wandb.log(log_dict)

        # ── Checkpoint ──
        if val_metrics['loss_total'] < best_val_loss:
            best_val_loss = val_metrics['loss_total']
            ckpt_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'val_snr': val_metrics['snr'],
            }, ckpt_path)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f}, "
                  f"SNR={val_metrics['snr']:.1f}dB)")

        # ── Eigenvalue diagnostics (every 10 epochs) ──
        if (epoch + 1) % 10 == 0:
            eig_info = model.get_eigenvalue_info()
            if eig_info.get('decoder_type') == 's5':
                # S5 decoder: report per-layer SSM dynamics
                for key in sorted(eig_info.keys()):
                    if key.endswith('_magnitudes'):
                        layer_name = key.replace('_magnitudes', '')
                        mags = eig_info[f'{layer_name}_magnitudes']
                        halflives = eig_info[f'{layer_name}_halflives']
                        print(f"  {layer_name}: |λ| range [{mags.min():.4f}, "
                              f"{mags.max():.4f}], "
                              f"halflife [{halflives.min():.1f}, "
                              f"{halflives.max():.1f}] steps")
            else:
                # Analytical / Learned decoder
                scale_str = ""
                if 'output_scale' in eig_info:
                    scale_str = f", scale={eig_info['output_scale']:.3f}, bias={eig_info['output_bias']:.4f}"
                print(f"  Eigenvalues: |λ| range [{eig_info['magnitudes'].min():.4f}, "
                      f"{eig_info['magnitudes'].max():.4f}], "
                      f"freq range [{eig_info['frequencies_normalized'].min():.3f}π, "
                      f"{eig_info['frequencies_normalized'].max():.3f}π], "
                      f"halflife range [{eig_info['halflives'].min():.1f}, "
                      f"{eig_info['halflives'].max():.1f}] steps{scale_str}")

    # ─── Final Test ───
    print(f"\n{'='*60}")
    print("Final evaluation on test set")
    print(f"{'='*60}")

    model.eval()
    test_snrs = []
    test_spec_snrs = []

    with torch.no_grad():
        for x, target in test_loader:
            x, target = x.to(device), target.to(device)
            x_hat, _, _ = model(x)
            test_snrs.append(compute_snr(x_hat, target))
            test_spec_snrs.append(compute_spectral_snr(x_hat, target))

    avg_snr = np.mean(test_snrs)
    avg_spec_snr = np.mean(test_spec_snrs)
    print(f"Test SNR: {avg_snr:.2f} dB")
    print(f"Test Spectral SNR: {avg_spec_snr:.2f} dB")
    print(f"Bitrate: {bits_per_sample:.4f} bits/sample")
    print(f"Best val loss: {best_val_loss:.6f}")

    if use_wandb:
        wandb.log({
            'test/snr': avg_snr,
            'test/spectral_snr': avg_spec_snr,
        })
        wandb.finish()

    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train S5 Codec')
    parser.add_argument('--config', type=str, default='codec/configs/phase1.yaml',
                        help='Path to config YAML')
    parser.add_argument('--decoder', type=str, default=None,
                        choices=['analytical', 'learned'],
                        help='Override decoder type')
    parser.add_argument('--phase', type=int, default=None,
                        help='Override curriculum phase (1-4)')
    parser.add_argument('--d_state', type=int, default=None,
                        help='Override state dimension')
    parser.add_argument('--wav', type=str, default=None,
                        help='Path to WAV file (overrides synthetic data)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for logging')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config {config_path} not found, using defaults.")
        config = {}

    # Apply CLI overrides
    if args.decoder:
        config.setdefault('model', {})['decoder_type'] = args.decoder
    if args.phase:
        config.setdefault('data', {})['phase'] = args.phase
    if args.d_state:
        config.setdefault('model', {})['d_state'] = args.d_state
    if args.wav:
        config.setdefault('data', {})['wav_path'] = args.wav
    if args.wandb:
        config['use_wandb'] = True
    if args.run_name:
        config['run_name'] = args.run_name

    # Ensure training config exists
    config.setdefault('training', {})
    config['training'].setdefault('batch_size', 32)
    config['training'].setdefault('epochs', 100)
    config['training'].setdefault('learning_rate', 1e-3)

    print("\n--- Configuration ---")
    print(yaml.dump(config, default_flow_style=False))
    print("---------------------\n")

    train(config)
