"""
Real audio dataset for S5 Codec.

Loads a WAV file and creates overlapping segments for training.
Simple, debuggable, no dependencies on huge datasets.

Usage:
    python codec/train_codec.py --audio /path/to/file.wav
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import tempfile


def convert_to_standard_wav(input_path: str, output_path: Optional[str] = None, 
                           sample_rate: int = 16000) -> str:
    """
    Convert any audio file to standard WAV using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output (if None, creates temp file)
        sample_rate: Target sample rate
    
    Returns:
        Path to converted WAV file
    """
    if output_path is None:
        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        import os
        os.close(fd)
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', str(sample_rate),
        '-ac', '1',  # mono
        '-sample_fmt', 's16',  # 16-bit PCM
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Converted {input_path} → {output_path} using ffmpeg")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{e.stderr}\n"
            f"Make sure ffmpeg is installed: apt-get install ffmpeg"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  Ubuntu/Debian: apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Or convert manually: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav"
        )


class WavFileDataset(Dataset):
    """
    Loads a single WAV file and splits it into segments for codec training.
    
    Segments are overlapping to maximize data usage from a single file.
    Returns (segment, segment) pairs — input = target for autoencoder.
    """
    def __init__(self,
                 wav_path: str,
                 segment_length: int = 1600,
                 sample_rate: int = 16000,
                 hop_length: Optional[int] = None,
                 normalize: bool = True,
                 max_segments: Optional[int] = None,
                 try_ffmpeg_convert: bool = True):
        """
        Args:
            wav_path: path to .wav file
            segment_length: samples per segment (default 0.1s @ 16kHz)
            sample_rate: target sample rate (will resample if needed)
            hop_length: stride between segments (default: segment_length // 2 for 50% overlap)
            normalize: normalize audio to [-1, 1]
            max_segments: limit total segments (for quick testing)
            try_ffmpeg_convert: if loading fails, try converting with ffmpeg
        """
        super().__init__()
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length or segment_length // 2
        
        # Load audio
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
        print(f"Loading audio from: {wav_path}")
        
        # Try torchaudio first, fall back to soundfile with different backends
        waveform = None
        orig_sr = None
        
        try:
            waveform, orig_sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"  torchaudio failed ({e})")
            
            # Try soundfile directly
            try:
                import soundfile as sf
                print(f"  Trying soundfile...")
                waveform, orig_sr = sf.read(wav_path, dtype='float32')
                # soundfile returns (samples, channels), we need (channels, samples)
                if waveform.ndim == 1:
                    waveform = waveform[np.newaxis, :]
                else:
                    waveform = waveform.T
                waveform = torch.from_numpy(waveform)
            except Exception as e2:
                print(f"  soundfile also failed ({e2})")
                
                # Last resort: ffmpeg conversion
                if try_ffmpeg_convert:
                    print(f"  Attempting ffmpeg conversion...")
                    try:
                        converted_path = convert_to_standard_wav(str(wav_path), sample_rate=sample_rate)
                        waveform, orig_sr = torchaudio.load(converted_path)
                        # Clean up temp file if created
                        if converted_path != str(wav_path):
                            import os
                            os.remove(converted_path)
                    except Exception as e3:
                        raise RuntimeError(
                            f"Could not load audio file with any method.\n"
                            f"  torchaudio: {e}\n"
                            f"  soundfile: {e2}\n"
                            f"  ffmpeg: {e3}\n"
                            f"File: {wav_path}"
                        )
                else:
                    raise RuntimeError(
                        f"Could not load audio file.\n"
                        f"  torchaudio: {e}\n"
                        f"  soundfile: {e2}\n"
                        f"Try: python codec/audio_dataset.py {wav_path} --convert"
                    )
        
        if waveform is None:
            raise RuntimeError(f"Failed to load {wav_path}")
        
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print(f"  Converted to mono from {waveform.shape[0]} channels")
        
        # Resample if needed
        if orig_sr != sample_rate:
            print(f"  Resampling from {orig_sr} Hz to {sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        
        # Flatten to 1D
        waveform = waveform.squeeze(0).numpy()
        
        # Normalize
        if normalize:
            peak = np.abs(waveform).max()
            if peak > 0:
                waveform = waveform / peak
                print(f"  Normalized to peak = 1.0 (was {peak:.3f})")
        
        self.waveform = waveform
        
        # Create segment indices
        n_samples = len(waveform)
        self.segment_starts = []
        
        pos = 0
        while pos + segment_length <= n_samples:
            self.segment_starts.append(pos)
            pos += self.hop_length
        
        if max_segments and len(self.segment_starts) > max_segments:
            self.segment_starts = self.segment_starts[:max_segments]
        
        print(f"  Duration: {n_samples / sample_rate:.2f}s ({n_samples:,} samples)")
        print(f"  Created {len(self.segment_starts)} segments "
              f"(length={segment_length}, hop={self.hop_length}, "
              f"overlap={(1 - self.hop_length/segment_length)*100:.0f}%)")
        print(f"  Total training samples: {len(self.segment_starts)}")
    
    def __len__(self):
        return len(self.segment_starts)
    
    def __getitem__(self, idx):
        """
        Returns (segment, segment) — autoencoder format.
        Shape: (T, 1) to match codec convention.
        """
        start = self.segment_starts[idx]
        segment = self.waveform[start : start + self.segment_length]
        
        # Convert to torch tensor and add channel dimension
        segment = torch.from_numpy(segment).float()
        segment = segment.unsqueeze(-1)  # (T, 1)
        
        return segment, segment


def create_wav_dataloaders(
    wav_path: str,
    segment_length: int = 1600,
    sample_rate: int = 16000,
    train_val_split: float = 0.9,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create train/val dataloaders from a single WAV file.
    
    Splits the file chronologically: first 90% for training, last 10% for validation.
    This simulates real-world scenario where you train on past audio and validate on future.
    """
    # Load full dataset
    full_dataset = WavFileDataset(
        wav_path=wav_path,
        segment_length=segment_length,
        sample_rate=sample_rate,
        hop_length=segment_length // 2,  # 50% overlap
    )
    
    # Chronological split (not random!)
    n_total = len(full_dataset)
    n_train = int(n_total * train_val_split)
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))
    
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_set)} segments")
    print(f"  Val:   {len(val_set)} segments")
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    
    # Create a small test set from validation
    test_indices = val_indices[:min(100, len(val_indices))]
    test_set = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test the dataloader"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_path', type=str, help='Path to WAV file')
    parser.add_argument('--segment_length', type=int, default=1600)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--convert', action='store_true',
                       help='Convert file with ffmpeg if loading fails')
    parser.add_argument('--save_converted', type=str, default=None,
                       help='Save converted file to this path')
    args = parser.parse_args()
    
    # If user wants to just convert the file
    if args.save_converted:
        print(f"Converting {args.wav_path} to standard WAV format...")
        output = convert_to_standard_wav(args.wav_path, args.save_converted, args.sample_rate)
        print(f"✓ Saved to: {output}")
        exit(0)
    
    print(f"Testing WavFileDataset on: {args.wav_path}\n")
    
    train_loader, val_loader, test_loader = create_wav_dataloaders(
        wav_path=args.wav_path,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=8,
    )
    
    # Test loading a batch
    print("\nLoading first batch...")
    x, target = next(iter(train_loader))
    print(f"  Batch shape: {x.shape}")
    print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Mean: {x.mean():.3f}, Std: {x.std():.3f}")
    
    print("\n✓ Dataloader test passed!")
