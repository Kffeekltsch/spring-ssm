"""
S5 Codec — Audio Inference & Listening Test

Loads a trained checkpoint, encodes+decodes a WAV file,
saves the reconstruction so you can LISTEN to it.

Also prints per-segment SNR and overall stats.

Usage:
    python codec/test_audio.py --checkpoint codec/runs/codec-audio-analytical/best_model.pth --input audio_inference.wav
    python codec/test_audio.py --checkpoint codec/runs/codec-audio-analytical/best_model.pth --input audio_inference.wav --no-quant
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codec.model import S5Codec
from codec.losses import compute_snr, compute_spectral_snr


def load_audio(path: str, target_sr: int = None):
    """Load audio file, return (waveform_1d_numpy, sample_rate)."""
    path = Path(path)
    
    waveform, sr = None, None
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype='float32')
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
        except Exception as e:
            raise RuntimeError(f"Cannot load {path}: {e}")

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if target_sr and sr != target_sr:
        print(f"  Resampling {sr} → {target_sr} Hz")
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr

    audio = waveform.squeeze(0).numpy()

    # Normalize to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    return audio, sr, peak


def save_wav(path: str, audio: np.ndarray, sr: int):
    """Save numpy array as WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    torchaudio.save(path, tensor, sr)
    print(f"  Saved: {path} ({len(audio)/sr:.2f}s, {sr}Hz)")


def run_codec(model, audio, segment_length, device, bypass_quant=False, batch_size=64):
    """
    Run the full encode→decode pipeline on a waveform.
    
    Returns:
        reconstructed: np.ndarray — same length as input
        segment_snrs: list of per-segment SNR values
        total_bits: total bits used
    """
    model.eval()
    n_samples = len(audio)
    
    # Pad to multiple of segment_length
    pad_len = (segment_length - n_samples % segment_length) % segment_length
    audio_padded = np.concatenate([audio, np.zeros(pad_len)])
    n_segments = len(audio_padded) // segment_length
    
    # Segment the audio (non-overlapping for clean reconstruction)
    segments = audio_padded.reshape(n_segments, segment_length)
    
    reconstructed_segments = []
    segment_snrs = []
    total_bits = 0
    
    with torch.no_grad():
        for start in range(0, n_segments, batch_size):
            end = min(start + batch_size, n_segments)
            batch = torch.from_numpy(segments[start:end]).float()
            batch = batch.unsqueeze(-1).to(device)  # (B, T, 1)
            
            x_hat, _, info = model(batch, bypass_quantization=bypass_quant)
            
            total_bits += info['bits_per_segment'] * (end - start)
            
            # Per-segment SNR
            for i in range(end - start):
                seg_in = batch[i:i+1]
                seg_out = x_hat[i:i+1]
                snr = compute_snr(seg_out, seg_in)
                segment_snrs.append(snr)
            
            reconstructed_segments.append(x_hat.squeeze(-1).cpu().numpy())
    
    reconstructed = np.concatenate(reconstructed_segments, axis=0).reshape(-1)
    
    # Trim padding
    reconstructed = reconstructed[:n_samples]
    
    return reconstructed, segment_snrs, total_bits


def main():
    parser = argparse.ArgumentParser(description='S5 Codec — Audio Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input WAV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output WAV path (default: input_reconstructed.wav)')
    parser.add_argument('--output-diff', type=str, default=None,
                        help='Save the error signal (input - output) as WAV')
    parser.add_argument('--no-quant', action='store_true',
                        help='Bypass quantization (upper bound on quality)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Override sample rate (default: from config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto)')
    args = parser.parse_args()

    # ─── Device ───
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"Device: {device}")

    # ─── Load checkpoint ───
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    data_cfg = cfg.get('data', {})
    
    segment_length = data_cfg.get('segment_length', 1600)
    sample_rate = args.sample_rate or data_cfg.get('sample_rate', 16000)
    
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val loss: {ckpt.get('best_val_loss', '?'):.6f}")
    print(f"  Val SNR: {ckpt.get('val_snr', '?'):.1f} dB")
    print(f"  Config: d_state={model_cfg.get('d_state')}, "
          f"d_hidden={model_cfg.get('d_hidden')}, "
          f"decoder={model_cfg.get('decoder_type')}")

    # ─── Build model ───
    model = S5Codec(
        d_state=model_cfg.get('d_state', 64),
        segment_length=segment_length,
        n_encoder_layers=model_cfg.get('n_encoder_layers', 2),
        n_decoder_layers=model_cfg.get('n_decoder_layers', 2),
        d_hidden=model_cfg.get('d_hidden', 16),
        n_quantization_levels=model_cfg.get('n_quantization_levels', 16),
        decoder_type=model_cfg.get('decoder_type', 'analytical'),
    )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    bits_per_sample = model.quantizer.bits_per_segment() / segment_length
    print(f"  Parameters: {n_params:,}")
    print(f"  Bitrate: {bits_per_sample:.4f} bits/sample = "
          f"{bits_per_sample * sample_rate / 1000:.2f} kbps @ {sample_rate}Hz")

    # ─── Load audio ───
    print(f"\nLoading audio: {args.input}")
    audio, sr, original_peak = load_audio(args.input, target_sr=sample_rate)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s ({len(audio):,} samples @ {sr}Hz)")
    print(f"  Original peak: {original_peak:.4f}")

    # ─── Run codec ───
    mode = "NO QUANTIZATION" if args.no_quant else "WITH QUANTIZATION"
    print(f"\nRunning codec ({mode})...")
    
    reconstructed, segment_snrs, total_bits = run_codec(
        model, audio, segment_length, device,
        bypass_quant=args.no_quant,
    )

    # ─── Stats ───
    overall_snr = 10 * np.log10(
        np.mean(audio**2) / (np.mean((audio - reconstructed)**2) + 1e-10)
    )
    segment_snrs_arr = np.array(segment_snrs)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS {'(no quantization)' if args.no_quant else ''}")
    print(f"{'='*60}")
    print(f"  Overall SNR:     {overall_snr:.2f} dB")
    print(f"  Segment SNR:     {segment_snrs_arr.mean():.2f} dB "
          f"(min={segment_snrs_arr.min():.1f}, max={segment_snrs_arr.max():.1f}, "
          f"std={segment_snrs_arr.std():.1f})")
    print(f"  Total bits:      {total_bits:.0f} "
          f"({total_bits/8:.0f} bytes)")
    print(f"  Compression:     {len(audio)*16 / total_bits:.1f}× "
          f"(vs 16-bit PCM)")
    print(f"  Bitrate:         {total_bits / duration / 1000:.2f} kbps")
    print(f"  Input RMS:       {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  Output RMS:      {np.sqrt(np.mean(reconstructed**2)):.4f}")
    print(f"  Error RMS:       {np.sqrt(np.mean((audio-reconstructed)**2)):.4f}")
    print(f"{'='*60}")

    # ─── Worst / best segments ───
    worst_5 = np.argsort(segment_snrs_arr)[:5]
    best_5 = np.argsort(segment_snrs_arr)[-5:][::-1]
    
    print(f"\n  Worst 5 segments:")
    for idx in worst_5:
        t_start = idx * segment_length / sr
        print(f"    Segment {idx:4d} (t={t_start:.2f}s): SNR={segment_snrs_arr[idx]:.1f} dB")
    
    print(f"\n  Best 5 segments:")
    for idx in best_5:
        t_start = idx * segment_length / sr
        print(f"    Segment {idx:4d} (t={t_start:.2f}s): SNR={segment_snrs_arr[idx]:.1f} dB")

    # ─── Save outputs ───
    # Reconstruct with original peak scaling
    reconstructed_scaled = reconstructed * original_peak

    if args.output is None:
        stem = Path(args.input).stem
        suffix = "_noq" if args.no_quant else ""
        args.output = f"{stem}_reconstructed{suffix}.wav"
    
    print(f"\nSaving output...")
    save_wav(args.output, reconstructed_scaled, sr)
    
    # Save error signal (amplified 10× so you can hear it)
    if args.output_diff is None:
        stem = Path(args.output).stem
        args.output_diff = f"{Path(args.output).parent / stem}_error.wav"
    
    error = (audio - reconstructed) * original_peak
    error_amplified = error * 10.0  # amplify so it's audible
    save_wav(args.output_diff, error_amplified, sr)
    print(f"  Error signal saved (10× amplified)")

    # ─── Side-by-side comparison hint ───
    print(f"\n{'='*60}")
    print(f"  Listen & compare:")
    print(f"    Original:      {args.input}")
    print(f"    Reconstructed: {args.output}")
    print(f"    Error (10×):   {args.output_diff}")
    print(f"{'='*60}")

    # ─── Also run without quant if we ran with quant ───
    if not args.no_quant:
        print(f"\n  Running without quantization for comparison...")
        recon_nq, snrs_nq, _ = run_codec(
            model, audio, segment_length, device,
            bypass_quant=True,
        )
        snr_nq = 10 * np.log10(
            np.mean(audio**2) / (np.mean((audio - recon_nq)**2) + 1e-10)
        )
        print(f"  SNR without quantization: {snr_nq:.2f} dB")
        print(f"  Quantization cost:        {snr_nq - overall_snr:.2f} dB")
        
        stem = Path(args.input).stem
        nq_path = f"{stem}_reconstructed_noq.wav"
        save_wav(nq_path, recon_nq * original_peak, sr)


if __name__ == '__main__':
    main()
