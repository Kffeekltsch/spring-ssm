"""
Codec losses: reconstruction quality metrics.

Combines time-domain and frequency-domain losses.
Keeps it lightweight — no auraloss dependency for the codec experiment,
so it can run standalone. But compatible with the same patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-resolution STFT loss.
    Computes L1 on magnitude spectrograms at multiple FFT sizes.
    Captures both fine and coarse spectral structure.
    """
    def __init__(self, fft_sizes=(64, 128, 256, 512),
                 hop_ratio=0.25, window='hann'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.window = window

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: (B, T, 1) or (B, T)
            target: (B, T, 1) or (B, T)
        Returns:
            loss: scalar
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)

        total_loss = 0.0
        for n_fft in self.fft_sizes:
            hop_length = max(1, int(n_fft * self.hop_ratio))
            window = torch.hann_window(n_fft, device=pred.device)

            pred_stft = torch.stft(
                pred, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True,
            )
            target_stft = torch.stft(
                target, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True,
            )

            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Linear magnitude loss
            mag_loss = F.l1_loss(pred_mag, target_mag)

            # Log magnitude loss (perceptual — emphasizes quiet components)
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(target_mag + 1e-7),
            )

            total_loss = total_loss + mag_loss + log_mag_loss

        return total_loss / len(self.fft_sizes)


class CodecLoss(nn.Module):
    """
    Combined loss for the S5 codec experiment.
    
    L_total = L_time + α_spec * L_spectral + α_vq * L_vq
    
    Clean and simple — let the SSM learn freely.
    """
    def __init__(self,
                 alpha_spectral: float = 1.0,
                 alpha_vq: float = 0.0,
                 spectral_fft_sizes=(64, 128, 256, 512),
                 time_loss_type: str = 'mse',
                 # Legacy params — accepted but ignored
                 alpha_eigenvalue_spread: float = 0.0,
                 alpha_c_sparsity: float = 0.0,
                 ):
        super().__init__()
        self.alpha_spectral = alpha_spectral
        self.alpha_vq = alpha_vq

        if time_loss_type == 'mse':
            self.time_loss = nn.MSELoss()
        elif time_loss_type == 'l1':
            self.time_loss = nn.L1Loss()
        else:
            self.time_loss = nn.MSELoss()
        
        self.spectral_loss = MultiScaleSpectralLoss(fft_sizes=spectral_fft_sizes)

    def forward(self, pred, target, vq_loss, model=None):
        """
        Args:
            pred: (B, T, 1) reconstructed signal
            target: (B, T, 1) original signal
            vq_loss: scalar from quantizer (monitoring only for scalar quant)
            model: unused, kept for API compat
        Returns:
            total_loss: scalar
            metrics: dict of individual losses
        """
        l_time = self.time_loss(pred, target)
        l_spec = self.spectral_loss(pred, target)

        total = l_time + self.alpha_spectral * l_spec + self.alpha_vq * vq_loss

        metrics = {
            'loss_time': l_time.item(),
            'loss_spectral': l_spec.item(),
            'loss_vq': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            'loss_total': total.item(),
        }

        return total, metrics


def compute_snr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    with torch.no_grad():
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)
        noise = target - pred
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr.item()


def compute_spectral_snr(pred: torch.Tensor, target: torch.Tensor,
                         n_fft: int = 512) -> float:
    """Compute SNR in frequency domain."""
    with torch.no_grad():
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)
        X = torch.fft.rfft(target)
        X_hat = torch.fft.rfft(pred)
        signal_power = torch.mean(torch.abs(X) ** 2)
        noise_power = torch.mean(torch.abs(X - X_hat) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr.item()
