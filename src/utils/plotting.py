import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path

def save_pcm_plot(dry_np, wet_np, pred_np, tag, output_dir, crop_samples=5000):
    """Saves a plot comparing the PCM waveforms of wet vs. predicted."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # Crop to a representative section for better visibility
    start = int(1.25*44100) #this is where the input is cut off
    end = start + crop_samples
    
    time_axis = np.arange(crop_samples)
    ax.plot(time_axis, dry_np[start:end], label="Input", color="blue", linestyle='-.', alpha=0.8)    
    ax.plot(time_axis, wet_np[start:end], label="Ground Truth", color="green", linestyle='-.', alpha=0.8)
    ax.plot(time_axis, pred_np[start:end], label="Model Prediction", color="orange", alpha=0.8)
    
    ax.legend(loc="upper right")
    ax.set_title(f"Waveform Comparison for {tag}")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = Path(output_dir) / f"{tag}_waveforms.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def save_melspectrogram_plot(dry_np, wet_np, pred_np, sr, tag, output_dir):
    """Saves a plot comparing the Mel spectrograms of input, truth, and prediction."""
    def to_mel_db(y, sr):
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        return librosa.power_to_db(mel_spec, ref=np.max)

    dry_mel = to_mel_db(dry_np, sr)
    wet_mel = to_mel_db(wet_np, sr)
    pred_mel = to_mel_db(pred_np, sr)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    
    librosa.display.specshow(dry_mel, sr=sr, x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title(f"Dry Input ({tag})")
    axes[0].set_ylabel("Frequency (Mel)")
    
    librosa.display.specshow(wet_mel, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title("Wet Ground Truth")
    axes[1].set_ylabel("Frequency (Mel)")

    img = librosa.display.specshow(pred_mel, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title("Model Prediction")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Mel)")



    fig.colorbar(img, ax=axes.ravel().tolist(), format='%+2.0f dB', label="Magnitude (dB)")
    save_path = Path(output_dir) / f"{tag}_melspectrograms.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def save_masked_phase_difference_plot(wet_np, pred_np, sr, tag, output_dir):
    """Saves a plot of the phase difference, masked by the ground truth magnitude."""
    n_fft = 1024
    hop_length = 256
    
    gt_stft = librosa.stft(wet_np, n_fft=n_fft, hop_length=hop_length)
    pred_stft = librosa.stft(pred_np, n_fft=n_fft, hop_length=hop_length)
    
    target_phase = np.angle(gt_stft)
    pred_phase = np.angle(pred_stft)
    phase_diff = np.mod((target_phase - pred_phase) + np.pi, 2 * np.pi) - np.pi
    
    gt_mag = np.abs(gt_stft)
    threshold = np.max(gt_mag) * 10**(-60 / 20.0) # Mask where GT is 60dB below peak
    mask = gt_mag < threshold
    phase_diff[mask] = np.nan # Use NaN for masked areas
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    cmap = plt.cm.twilight_shifted.copy()
    cmap.set_bad(color='white') # Color for NaN values

    img = librosa.display.specshow(phase_diff, sr=sr, hop_length=hop_length,
                                   x_axis='time', y_axis='linear', ax=ax,
                                   cmap=cmap, vmin=-np.pi, vmax=np.pi)
    
    ax.set_title(f"Masked Phase Difference ({tag})")
    ax.set_ylim(0, 16000) # Limit frequency for visibility
    fig.colorbar(img, label='Phase Difference (radians)')
    plt.tight_layout()
    save_path = Path(output_dir) / f"{tag}_phase_diff.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

import librosa.display

def save_full_summary_plot(dry_np, wet_np, pred_np, sr, model_name, tag, output_dir, metrics_dict):
    """
    Creates and saves a comprehensive 3x3 summary plot for a single evaluation sample.
    """
    fig, axs = plt.subplots(3, 3, figsize=(24, 15), constrained_layout=True)
    fig.suptitle(f"Evaluation Summary for {tag} (Model: {model_name})", fontsize=18, weight='bold')

    # --- Row 1: Waveforms ---
    time_axis = np.arange(len(dry_np)) / sr
    axs[0, 0].plot(time_axis, dry_np, color='gray')
    axs[0, 0].set_title("(a) Dry Input Waveform")
    axs[0, 0].set_xlabel("Time (s)"); axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, linestyle=':')

    axs[0, 1].plot(time_axis, wet_np, color='green')
    axs[0, 1].set_title("(b) Wet Ground Truth")
    axs[0, 1].set_xlabel("Time (s)"); axs[0, 1].sharey(axs[0,0])

    axs[0, 2].plot(time_axis, pred_np, color='orange')
    axs[0, 2].set_title("(c) Predicted Output")
    axs[0, 2].set_xlabel("Time (s)"); axs[0, 2].sharey(axs[0,0])

    # --- Row 2: Mel Spectrograms ---
    def to_mel_db(y):
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        return librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(to_mel_db(dry_np), sr=sr, x_axis='time', y_axis='mel', ax=axs[1, 0])
    axs[1, 0].set_title("(d) Dry Input Mel Spectrogram")
    
    librosa.display.specshow(to_mel_db(wet_np), sr=sr, x_axis='time', y_axis='mel', ax=axs[1, 1])
    axs[1, 1].set_title("(e) Wet Ground Truth Mel Spectrogram")
    
    librosa.display.specshow(to_mel_db(pred_np), sr=sr, x_axis='time', y_axis='mel', ax=axs[1, 2])
    axs[1, 2].set_title("(f) Predicted Mel Spectrogram")

    # Add a single shared colorbar for the spectrograms
    fig.colorbar(img, ax=axs[1, :], format='%+2.0f dB', label='Magnitude (dB)', shrink=0.8)

    # --- Row 3: Analysis ---
    # Masked Phase Difference
    gt_stft = librosa.stft(wet_np, n_fft=1024, hop_length=256)
    pred_stft = librosa.stft(pred_np, n_fft=1024, hop_length=256)
    phase_diff = np.angle(gt_stft) - np.angle(pred_stft)
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    
    gt_mag = np.abs(gt_stft)
    mask = gt_mag < (np.max(gt_mag) * 10**(-50 / 20.0))
    phase_diff[mask] = np.nan
    
    cmap = plt.cm.twilight_shifted.copy(); cmap.set_bad(color='lightgray')
    img_phase = librosa.display.specshow(phase_diff, sr=sr, hop_length=256, x_axis='time', y_axis='linear', ax=axs[2, 0], cmap=cmap, vmin=-np.pi, vmax=np.pi)
    axs[2, 0].set_title("(g) Masked Phase Difference"); axs[2, 0].set_ylim(0, 16000)
    fig.colorbar(img_phase, ax=axs[2, 0], label='Phase Diff (rad)', shrink=0.8)

    # Metrics Text
    metric_text = "Objective Metrics:\n\n"
    for key, val in metrics_dict.items():
        metric_text += f"{key}: {val:.4f}\n"
        
    axs[2, 1].axis('off')
    axs[2, 1].text(0.0, 0.95, metric_text, ha='left', va='top', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    axs[2, 2].axis('off')

    save_path = Path(output_dir) / f"full_summary_{tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)