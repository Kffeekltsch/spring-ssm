import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import spectrogram
import librosa
import torch
import wandb


def plot_loss(train_losses, val_losses, modelname, id):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    loss_curve_path = f"train_logs/{modelname}_{id}_loss_curve.png"
    plt.savefig(loss_curve_path)
    #plt.show()
    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

def calculate_magnitude_weighted_phase_error(target_stft, pred_stft, eps=1e-8):
    """
    Calculates the magnitude-weighted average absolute phase error.

    Args:
        target_stft (np.ndarray): Complex STFT of the target signal (Freq, Time).
        pred_stft (np.ndarray): Complex STFT of the predicted signal (Freq, Time).
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: Magnitude-weighted phase error (scalar). Lower is better.
    """
    # Calculate phase difference (handle wrapping)
    target_phase = np.angle(target_stft)
    pred_phase = np.angle(pred_stft)
    phase_diff = target_phase - pred_phase
    # Wrap phase difference to [-pi, pi]
    phase_diff_wrapped = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    abs_phase_error = np.abs(phase_diff_wrapped)

    # Calculate magnitude of the target as weights
    target_mag = np.abs(target_stft)

    # Normalize magnitudes to act as weights summing to 1 (optional but good)
    mag_weights = target_mag / (np.sum(target_mag) + eps)

    # Calculate weighted average absolute phase error
    weighted_error = np.sum(abs_phase_error * mag_weights)

    return weighted_error

def plot_masked_phase_diff(ground_truth, predicted, modelname, sr=44100, id=None,
                            n_fft=512, hop_length=256, output_dir="image_logs_inference",
                            mask_db_threshold=-80): # Threshold in dB relative to max
    """
    Plots phase difference, masking regions where magnitude is low.
    Also returns the magnitude-weighted phase error metric.
    """
    # --- Convert to numpy if needed ---
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.squeeze().cpu().numpy()

    # --- STFT ---
    gt_stft = librosa.stft(ground_truth, n_fft=n_fft, hop_length=hop_length)
    pred_stft = librosa.stft(predicted, n_fft=n_fft, hop_length=hop_length)

    # --- Calculate Phase Difference ---
    target_phase = np.angle(gt_stft)
    pred_phase = np.angle(pred_stft)
    phase_diff = target_phase - pred_phase
    # Wrap phase difference to [-pi, pi] for consistent plotting/calculation
    phase_diff_wrapped = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

    # --- Create Mask based on Magnitude Threshold ---
    gt_mag = np.abs(gt_stft)
    pred_mag = np.abs(pred_stft)

    # Threshold relative to the max magnitude in the ground truth
    max_mag_gt = np.max(gt_mag)
    threshold_linear = max_mag_gt * (10**(mask_db_threshold / 20.0)) if max_mag_gt > 1e-8 else 1e-8

    # Mask where BOTH are below threshold (or just GT, debatable)
    mask = (gt_mag >= threshold_linear) # & (pred_mag >= threshold_linear)
    # Apply mask: Set phase difference to NaN where magnitude is low
    phase_diff_masked = np.copy(phase_diff_wrapped)
    phase_diff_masked[~mask] = np.nan # Use NaN for masked areas

    # --- Plotting ---
    plt.figure(figsize=(10, 4))
    # Use a colormap that handles NaN (like 'viridis', 'magma', set bad color)
    cmap = plt.cm.twilight.copy()
    cmap.set_bad(color='white') # Or 'gray', 'black'

    librosa.display.specshow(phase_diff_masked, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='linear',
                             cmap=cmap, vmin=-np.pi, vmax=np.pi)

    plt.ylim(0, 12000) # Keep your frequency limit
    plt.colorbar(label='Phase Difference (radians)')
    #plt.title(f'Masked Phase Difference (GT Mag > {mask_db_threshold} dB)')
    plt.tight_layout()
    phase_plot_path = f"{output_dir}/phase_plot_masked_diff_{modelname}_{id}.png"
    plt.savefig(phase_plot_path)
    plt.close()

    # --- Calculate Metric ---
    phase_error_metric = calculate_magnitude_weighted_phase_error(gt_stft, pred_stft)

    return phase_plot_path, phase_error_metric


def plot_phase_diff(ground_truth, predicted, modelname,sr=44100, id = None, n_fft=512, hop_length=256, output_dir = None):
    # STFT
    gt_stft = librosa.stft(ground_truth, n_fft=n_fft, hop_length=hop_length)
    pred_stft = librosa.stft(predicted, n_fft=n_fft, hop_length=hop_length)
    
    # Phase difference
    phase_diff = np.angle(pred_stft) - np.angle(gt_stft)
    
    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(phase_diff, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='linear',
                             cmap='twilight',  # a good colormap for phase
                             vmin=-np.pi, vmax=np.pi)
    plt.ylim(0, 12_000)
    plt.colorbar(label='Phase Difference (radians)')
    #plt.title('Phase Difference (Predicted vs. Ground Truth)')
    plt.tight_layout()
    phase_plot_path = f"{output_dir}/phase_plot_diff_{modelname}_{id}.png"
    plt.savefig(phase_plot_path)
    plt.close()
    #plt.show()
    
def plot_phase(audio, ground_truth, predicted, fs, modelname, id=None, output_dir = "image_logs_inference"):
    """
    Plots the phase of the input audio, ground truth, and predicted audio.
    
    Parameters:
        audio (Tensor or np.array): Input audio signal.
        ground_truth (Tensor or np.array): Ground truth audio signal.
        predicted (Tensor or np.array): Predicted audio signal.
        fs (int): Sampling rate.
        modelname (str): Model identifier (used in filename).
        id (str): Additional identifier for the saved image.
    """
    def process_data(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.squeeze(data)
    
    # Process signals to numpy arrays
    
    audio = process_data(audio)
    ground_truth = process_data(ground_truth)
    predicted = process_data(predicted)
    
    plt.figure(figsize=(15, 10))
    
    # Compute STFT for each signal (using common parameters)
    n_fft = 512
    hop_length = 256
    
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase_audio = np.angle(stft_audio)
    
    stft_ground = librosa.stft(ground_truth, n_fft=n_fft, hop_length=hop_length)
    phase_ground = np.angle(stft_ground)
    
    stft_pred = librosa.stft(predicted, n_fft=n_fft, hop_length=hop_length)
    phase_pred = np.angle(stft_pred)
    freqs = np.linspace(0, fs/2, stft_audio.shape[0])
    # Find indices for frequencies between 5 kHz and 6 kHz
    freq_idx = np.where((freqs >= 5000) & (freqs <= 6000))[0]
    
    # Slice the phase arrays to keep only the selected frequency bins
    phase_audio = phase_audio[freq_idx, :]
    phase_ground = phase_ground[freq_idx, :]
    phase_pred = phase_pred[freq_idx, :]
    # Plot Input Audio Phase
    plt.subplot(3, 1, 1)
    librosa.display.specshow(phase_audio, sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.title('Input Audio Phase')
    plt.colorbar(format='%+2.2f rad')
    
    # Plot Ground Truth Phase
    plt.subplot(3, 1, 2)
    librosa.display.specshow(phase_ground, sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.title('Ground Truth Phase')
    plt.colorbar(format='%+2.2f rad')
    
    # Plot Predicted Output Phase
    plt.subplot(3, 1, 3)
    librosa.display.specshow(phase_pred, sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.title('Predicted Output Phase')
    plt.colorbar(format='%+2.2f rad')
    
    plt.tight_layout()
    phase_plot_path = f"{output_dir}/phase_plot_{modelname}_{id}.png"
    plt.savefig(phase_plot_path)
    plt.close()
    
def PCM_plot(audio,ground_truth,predicted, modelname, id = None, output_dir = "image_logs_inference"):

    plt.figure(figsize=(12, 8))
    #plt.title(f"{modelname} PCM-plot")
    # Plot input audio
    """
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
    plt.plot(audio[4000:14000], label="Input Audio")
    plt.title("Input Audio")
    plt.legend(loc="upper right")
    """
    # Plot ground truth (filtered audio)
    plt.subplot(1, 1, 1)  # 3 rows, 1 column, 2nd plot
    plt.plot(ground_truth[5000:10000], label="Ground Truth", color="green")
    #plt.title("Ground Truth")
    plt.legend(loc="upper right")
    """
    plt.subplot(1, 1, 1)  # 3 rows, 1 column, 2nd plot
    plt.plot(audio[0:4000], label="Dry signal", color="blue")
    plt.title("Dry signal")
    plt.legend(loc="upper right")
    """
    # Plot model output
    plt.subplot(1, 1, 1)  # 3 rows, 1 column, 3rd plot
    plt.plot(predicted[5000:10000],linestyle = "-.", label="Model Output", color="orange")
    
    plt.legend(loc="upper right")
    
    # Adjust layout and save
    #plt.tight_layout()
    plt.savefig(f"{output_dir}/PCM_subplots_{modelname}_{id}.png")  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory
    
def plot_melspectrograms_in_out(audio, ground_truth, predicted, fs, modelname, id = None, output_dir = "image_logs_inference"):
    def process_data(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.squeeze(data)

    # Process the input signals
    audio = process_data(audio)
    ground_truth = process_data(ground_truth)
    predicted = process_data(predicted)
    
    plt.figure(figsize=(15, 10))
    
    # Input Audio Mel Spectrogram

    S_audio = librosa.feature.melspectrogram(y=audio,n_mels=64, sr=fs, n_fft=512, hop_length=256)
    S_audio_db = librosa.power_to_db(S_audio, ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_audio_db, sr=fs, hop_length=256, x_axis='time', y_axis='mel')
    plt.title('Input Audio Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    
    # Ground Truth Mel Spectrogram
    S_truth = librosa.feature.melspectrogram(y=ground_truth,n_mels=64,  sr=fs, n_fft=512, hop_length=256)
    S_truth_db = librosa.power_to_db(S_truth, ref=np.max)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_truth_db, sr=fs, hop_length=256, x_axis='time', y_axis='mel')
    plt.title('Ground Truth Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    

    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mel_spectrogram_in_out_comparison{modelname}_{id}.png")
    plt.close()    
    
def plot_melspectrograms(audio, ground_truth, predicted, fs, modelname, id = None, output_dir = "image_logs_inference"):
    def process_data(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.squeeze(data)

    # Process the input signals
    audio = process_data(audio)
    ground_truth = process_data(ground_truth)
    predicted = process_data(predicted)
    
    plt.figure(figsize=(15, 10))
    
    # Input Audio Mel Spectrogram
    """
    S_audio = librosa.feature.melspectrogram(y=audio,n_mels=64, sr=fs, n_fft=512, hop_length=256)
    S_audio_db = librosa.power_to_db(S_audio, ref=np.max)
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_audio_db, sr=fs, hop_length=256, x_axis='time', y_axis='mel')
    plt.title('Input Audio Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    """
    
    # Ground Truth Mel Spectrogram
    """
    S_truth = librosa.feature.melspectrogram(y=ground_truth,n_mels=64,  sr=fs, n_fft=512, hop_length=256)
    S_truth_db = librosa.power_to_db(S_truth, ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_truth_db, sr=fs, hop_length=256, x_axis='time', y_axis='mel')
    plt.title('Ground Truth Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    """
    
    # Predicted Output Mel Spectrogram
    S_pred = librosa.feature.melspectrogram(y=predicted,n_mels=64,  sr=fs, n_fft=512, hop_length=256)
    S_pred_db = librosa.power_to_db(S_pred, ref=np.max)
    plt.subplot(1, 1, 1)
    librosa.display.specshow(S_pred_db, sr=fs, hop_length=256, x_axis='time', y_axis='mel')
    plt.title('Predicted Output Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mel_spectrogram_comparison{modelname}_{id}.png")
    plt.close()

def plot_in_out_spectrograms(audio, ground_truth, predicted, fs, modelname, id=None, output_dir = "image_logs_inference"):
    def process_data(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.squeeze(data)

    audio = process_data(audio)
    ground_truth = process_data(ground_truth)
    predicted = process_data(predicted)
    plt.figure(figsize=(15, 10))

    # Input Audio Spectrogram
    
    f_audio, t_audio, Sxx_audio = spectrogram(audio, fs=fs, nperseg=512, noverlap=256)
    plt.subplot(1, 1, 1)
    plt.pcolormesh(t_audio, f_audio, 10 * np.log10(Sxx_audio+1e-15), shading='gouraud')
    plt.title('Input Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    
    
    # Ground Truth Spectrogram
    f_truth, t_truth, Sxx_truth = spectrogram(ground_truth, fs=fs, nperseg=512, noverlap=256)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_truth, f_truth, 10 * np.log10(Sxx_truth+1e-15), shading='gouraud')
    plt.title('Ground Truth Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    """
    # Predicted Output Spectrogram
    f_pred, t_pred, Sxx_pred = spectrogram(predicted, fs=fs, nperseg=512, noverlap=256)
    plt.subplot(3, 1, 3)
    plt.pcolormesh(t_pred, f_pred, 10 * np.log10(Sxx_pred + 1e-15), shading='gouraud')
    plt.title('Predicted Output Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    """
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectrogram_comparison_in_out_{modelname}_{id}.png")
    plt.close()
    
def plot_spectrograms(audio, ground_truth, predicted, fs, modelname, id=None, output_dir = "image_logs_inference"):
    def process_data(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.squeeze(data)

    audio = process_data(audio)
    ground_truth = process_data(ground_truth)
    predicted = process_data(predicted)
    plt.figure(figsize=(15, 10))

    # Input Audio Spectrogram
    """
    f_audio, t_audio, Sxx_audio = spectrogram(audio, fs=fs, nperseg=512, noverlap=256)
    plt.subplot(3, 1, 1)
    plt.pcolormesh(t_audio, f_audio, 10 * np.log10(Sxx_audio+1e-15), shading='gouraud')
    plt.title('Input Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    """
    
    # Ground Truth Spectrogram
    """
    f_truth, t_truth, Sxx_truth = spectrogram(ground_truth, fs=fs, nperseg=512, noverlap=256)
    plt.subplot(1, 1, 2)
    plt.pcolormesh(t_truth, f_truth, 10 * np.log10(Sxx_truth+1e-15), shading='gouraud')
    plt.title('Ground Truth Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    """
    
    # Predicted Output Spectrogram
    f_pred, t_pred, Sxx_pred = spectrogram(predicted, fs=fs, nperseg=512, noverlap=256)
    #plt.subplot(1, 1, 3)
    plt.pcolormesh(t_pred, f_pred, 10 * np.log10(Sxx_pred + 1e-15), shading='gouraud')
    plt.title('Predicted Output Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectrogram_comparison_{modelname}_{id}.png")
    plt.close()
    

def plot_frequency_spectrum(audio, ground_truth, predicted, fs, modelname, id= None, output_dir = "image_logs_inference"):
    # Compute FFT for each signal
    fft_audio = np.fft.fft(audio)
    fft_ground = np.fft.fft(ground_truth)
    fft_predicted = np.fft.fft(predicted)
    
    # Compute frequencies
    freqs_audio = np.fft.fftfreq(len(audio), d=1/fs)
    freqs_ground = np.fft.fftfreq(len(ground_truth), d=1/fs)
    freqs_predicted = np.fft.fftfreq(len(predicted), d=1/fs)
    
    # Compute magnitudes
    magnitude_audio = np.abs(fft_audio)
    magnitude_ground = np.abs(fft_ground)
    magnitude_predicted = np.abs(fft_predicted)

    # Apply frequency limit (0 to 20kHz)
    mask_audio = (freqs_audio >= 0) & (freqs_audio <= 20000)
    mask_ground = (freqs_ground >= 0) & (freqs_ground <= 20000)
    mask_predicted = (freqs_predicted >= 0) & (freqs_predicted <= 20000)

    # Create subplots
    plt.figure(figsize=(15, 10))
    
    # Input audio spectrum
    plt.subplot(3, 1, 1)
    plt.plot(freqs_audio[mask_audio], magnitude_audio[mask_audio], label="Input Audio")
    plt.title("Input Audio Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend(loc="upper right")
    
    # Ground truth spectrum
    plt.subplot(3, 1, 2)
    plt.plot(freqs_ground[mask_ground], magnitude_ground[mask_ground], label="Ground Truth", color="green")
    plt.title("Ground Truth Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend(loc="upper right")
    
    # Predicted output spectrum
    plt.subplot(3, 1, 3)
    plt.plot(freqs_predicted[mask_predicted], magnitude_predicted[mask_predicted], label="Predicted Output", color="orange")
    plt.title("Predicted Output Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend(loc="upper right")
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/freq_spectrum_{modelname}_{id}.png")
    plt.close()
 

