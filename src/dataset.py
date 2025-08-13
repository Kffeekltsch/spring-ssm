# src/dataset.py
import torch
from torch.utils.data import Dataset, random_split
import torchaudio
from pathlib import Path

def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    """Removes DC offset from a waveform tensor of shape [..., Time]."""
    return waveform - torch.mean(waveform, dim=-1, keepdim=True)

class WavReverbDataset(Dataset):
    """
    Dataset for paired dry/wet audio, supporting two loading modes.
    This version works entirely with PyTorch Tensors, avoiding NumPy conversions.
    """
    def __init__(self, dry_path: str, wet_path: str, sample_rate: int, duration: float, load_in_memory: bool = False):
        """
        Args:
            dry_path (str): Path to the single, long dry audio .wav file.
            wet_path (str): Path to the single, long wet audio .wav file.
            sample_rate (int): The target sample rate for all audio.
            duration (float): The duration of each segment in seconds.
            load_in_memory (bool): If True, pre-loads all audio into RAM.
        """
        super().__init__()
        self.dry_path = Path(dry_path)
        self.wet_path = Path(wet_path)
        self.sample_rate = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.load_in_memory = load_in_memory

        if not self.dry_path.is_file() or not self.wet_path.is_file():
            raise FileNotFoundError(f"Audio file not found at {self.dry_path} or {self.wet_path}")

        if self.load_in_memory:
            print("Pre-loading entire dataset into RAM (performance mode)...")
            self.dry_waveform_full, self.wet_waveform_full = self._load_and_process_full_audio()
            self.num_segments = self.dry_waveform_full.shape[1] // self.target_samples
        else:
            print("Initializing dataset in memory-efficient mode...")
            info = torchaudio.info(str(self.dry_path))
            self.source_sr = info.sample_rate
            self.num_segments = info.num_frames // self.target_samples
            self.resampler = torchaudio.transforms.Resample(self.source_sr, self.sample_rate) \
                if self.source_sr != self.sample_rate else None

    def _load_and_process_full_audio(self):
        dry_waveform, sr_dry = torchaudio.load(str(self.dry_path))
        wet_waveform, sr_wet = torchaudio.load(str(self.wet_path))

        if sr_dry != self.sample_rate:
            dry_waveform = torchaudio.transforms.Resample(sr_dry, self.sample_rate)(dry_waveform)
        if sr_wet != self.sample_rate:
            wet_waveform = torchaudio.transforms.Resample(sr_wet, self.sample_rate)(wet_waveform)
        
        if dry_waveform.shape[0] > 1: dry_waveform = torch.mean(dry_waveform, dim=0, keepdim=True)
        if wet_waveform.shape[0] > 1: wet_waveform = torch.mean(wet_waveform, dim=0, keepdim=True)

        dry_waveform = remove_dc_offset(dry_waveform)
        wet_waveform = remove_dc_offset(wet_waveform)
        
        if dry_waveform.shape[1] != wet_waveform.shape[1]:
            raise ValueError("Dry and wet waveforms must have the same length!")
            
        return dry_waveform, wet_waveform

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        if self.load_in_memory:
            start = idx * self.target_samples
            end = start + self.target_samples
            dry_segment = self.dry_waveform_full[:, start:end]
            wet_segment = self.wet_waveform_full[:, start:end]
        else:
            start_sample = idx * self.target_samples
            dry_segment, _ = torchaudio.load(str(self.dry_path), frame_offset=start_sample, num_frames=self.target_samples)
            wet_segment, _ = torchaudio.load(str(self.wet_path), frame_offset=start_sample, num_frames=self.target_samples)

            if self.resampler:
                dry_segment = self.resampler(dry_segment)
                wet_segment = self.resampler(wet_segment)

            if dry_segment.shape[0] > 1: dry_segment = torch.mean(dry_segment, dim=0, keepdim=True)
            if wet_segment.shape[0] > 1: wet_segment = torch.mean(wet_segment, dim=0, keepdim=True)
                
            dry_segment = remove_dc_offset(dry_segment)
            wet_segment = remove_dc_offset(wet_segment)

        # Final shape should be [Time, Channels=1] for the models
        return dry_segment.permute(1, 0), wet_segment.permute(1, 0)


def split_dataset(dataset, train_split=0.7, val_split=0.2, seed=42):
    """Splits a dataset into training, validation, and test sets."""
    dataset_size = len(dataset)
    if train_split + val_split >= 1.0:
        raise ValueError("Sum of train_split and val_split must be less than 1.")
        
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    if train_size + val_size + test_size != dataset_size: # Adjust for rounding
        train_size += (dataset_size - (train_size + val_size + test_size))
        
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)