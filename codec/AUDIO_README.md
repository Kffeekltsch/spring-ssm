# S5 Codec — Real Audio Training

## Quick Start

### 1. Test the dataloader

```bash
# Test on any WAV file
python codec/audio_dataset.py /path/to/your/file.wav

# With custom parameters
python codec/audio_dataset.py /path/to/file.wav --segment_length 3200 --sample_rate 16000
```

### 2. Train on real audio

```bash
# Using the audio config + CLI override
python codec/train_codec.py --config codec/configs/audio.yaml --wav /path/to/file.wav

# Or edit audio.yaml to set wav_path directly
# Then just:
python codec/train_codec.py --config codec/configs/audio.yaml
```

### 3. Compare with synthetic baseline

```bash
# Synthetic Phase 1 (single sines)
python codec/train_codec.py --config codec/configs/phase1.yaml

# Real audio
python codec/train_codec.py --config codec/configs/audio.yaml --wav speech.wav
```

## Configuration

The `audio.yaml` config is tuned for real audio:
- **d_state=16**: More eigenvalue capacity (vs 4 for synthetic)
- **d_hidden=32**: Larger hidden dimension
- **n_quantization_levels=32**: 5 bits/dim (vs 4 bits for synthetic)
- **quant_warmup_epochs=20**: Longer warmup for complex signals

## Finding WAV Files

### Quick test files:
```bash
# Download a short speech sample (if you have wget/curl)
wget https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav

# Or use any .wav/.mp3 you have (will be resampled to 16kHz mono)
```

### Public datasets:
- **LibriSpeech**: Free speech dataset (hundreds of hours)
- **MUSDB18**: Music source separation dataset
- **FSDnoisy18k**: Environmental sounds

## Output

The codec will show:
```
Loading audio from: speech.wav
  Resampling from 44100 Hz to 16000 Hz
  Normalized to peak = 1.0
  Duration: 5.23s (83680 samples)
  Created 104 segments (length=1600, hop=800, overlap=50%)
  Total training samples: 104

Dataset split:
  Train: 93 segments
  Val:   11 segments
```

## Bitrate Calculation

With `d_state=16`, `n_levels=32` (5 bits), `segment_length=1600`:
```
bits_per_segment = 2 * d_state * bits_per_dim
                 = 2 * 16 * 5
                 = 160 bits

bits_per_sample  = 160 / 1600 = 0.1 bits/sample

At 16kHz:
bitrate = 0.1 * 16000 = 1.6 kbps
```

Compare to:
- Uncompressed 16-bit PCM: 256 kbps
- Opus @ 64 kbps: standard voice codec
- Your SSM codec @ 1.6 kbps: **160× compression!**

(Of course, quality matters — that's what we're training for)
