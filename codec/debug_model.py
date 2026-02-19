"""
S5 Codec — Comprehensive SSM Diagnostic & Visualization

Plots:
1. Eigenvalue map (complex plane — unit circle)
2. Bode diagrams (magnitude + phase response for each eigenvalue channel)
3. Impulse response of each eigenvalue channel
4. Encoder S5 internal eigenvalues (from the SequenceLayer SSMs)
5. Input vs reconstruction comparison
6. VQ codebook utilization
7. State vector distribution (pre/post quantization)
8. Energy spectrum: which eigenvalue channels carry signal energy

Run:
    python codec/debug_model.py
    
Optionally load a trained checkpoint:
    python codec/debug_model.py --checkpoint codec/runs/codec-phase1-analytical/best_model.pth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codec.model import S5Codec, AnalyticalDecoder
from codec.dataset import SyntheticSignalDataset
from codec.losses import compute_snr, compute_spectral_snr
from ssm.src.model.ssm import as_complex


def extract_encoder_eigenvalues(model):
    """
    Extract eigenvalues from each S5 SequenceLayer inside the encoder.
    These are the continuous-time eigenvalues Λ and the discretized Λ_bar.
    """
    encoder_eigs = []
    for i, layer in enumerate(model.encoder.layers):
        ssm = layer.s5.seq  # the SSM module

        # Continuous-time eigenvalues (stored as (N, 2) real tensor)
        Lambda_raw = ssm.Lambda.data.clone()
        Lambda_c = torch.complex(Lambda_raw[:, 0], Lambda_raw[:, 1])

        # Apply stability constraint (same as in forward)
        if ssm.ensure_stability == 'abs':
            Lambda_c = torch.complex(-torch.abs(Lambda_c.real), Lambda_c.imag)
        elif ssm.ensure_stability == 'relu':
            Lambda_c = torch.complex(-torch.relu(-Lambda_c.real), Lambda_c.imag)

        # Discretize with ZOH
        step = ssm.step_scale * torch.exp(ssm.log_step.data)
        Lambda_bar = torch.exp(Lambda_c * step)

        # B, C matrices
        B_c = torch.complex(ssm.B.data[:, :, 0], ssm.B.data[:, :, 1])
        C_c = torch.complex(ssm.C.data[:, :, 0], ssm.C.data[:, :, 1])

        encoder_eigs.append({
            'layer': i,
            'lambda_continuous': Lambda_c.detach().cpu().numpy(),
            'lambda_discrete': Lambda_bar.detach().cpu().numpy(),
            'step_sizes': step.detach().cpu().numpy(),
            'B': B_c.detach().cpu().numpy(),
            'C': C_c.detach().cpu().numpy(),
        })

    return encoder_eigs


def extract_decoder_eigenvalues(model):
    """Extract eigenvalues from the analytical decoder."""
    decoder = model.decoder
    if hasattr(decoder, 'analytical'):
        decoder = decoder.analytical  # LearnedDecoder wraps AnalyticalDecoder

    lambdas = decoder.get_eigenvalues().detach().cpu().numpy()
    C = torch.complex(decoder.C.data[:, 0], decoder.C.data[:, 1]).detach().cpu().numpy()
    neg_log = decoder.neg_log_one_minus_r.data.detach().cpu().numpy()
    angle = decoder.angle.data.detach().cpu().numpy()

    # Compute r from the parameterization
    one_minus_r = np.log1p(np.exp(neg_log))  # softplus
    one_minus_r = np.clip(one_minus_r, 1e-4, 0.999)
    r_values = 1.0 - one_minus_r

    return {
        'eigenvalues': lambdas,
        'magnitudes': np.abs(lambdas),
        'angles': np.angle(lambdas),
        'halflives': -np.log(2) / np.log(np.abs(lambdas) + 1e-10),
        'C': C,
        'C_magnitudes': np.abs(C),
        'neg_log_raw': neg_log,
        'angle_raw': angle,
        'r_values': r_values,
    }


def compute_bode(eigenvalue, C_weight, sample_rate=16000, n_points=2000):
    """
    Compute Bode diagram (magnitude + phase) for a single eigenvalue channel.
    
    Transfer function of one channel of the autonomous SSM:
        H(z) = C / (z - λ)
    
    Evaluated on the unit circle z = e^(jω) for ω ∈ [0, π].
    """
    omega = np.linspace(0, np.pi, n_points)
    z = np.exp(1j * omega)

    # H(z) = C / (z - λ)
    H = C_weight / (z - eigenvalue)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
    phase_rad = np.angle(H)
    freq_hz = omega * sample_rate / (2 * np.pi)

    return freq_hz, magnitude_db, phase_rad


def compute_impulse_response(eigenvalue, C_weight, n_steps=500):
    """
    Impulse response of one eigenvalue channel:
        h(t) = Re[C · λ^t]
    """
    t = np.arange(n_steps)
    h = np.real(C_weight * eigenvalue ** t)
    return t, h


def plot_diagnostics(model, x, x_hat, z, z_q, info, sample_rate=16000,
                     save_path=None):
    """
    Generate all diagnostic plots.
    """
    # ── Extract data ──
    dec_eigs = extract_decoder_eigenvalues(model)
    enc_eigs = extract_encoder_eigenvalues(model)

    lambdas = dec_eigs['eigenvalues']
    N = len(lambdas)
    C_weights = dec_eigs['C']

    x_np = x[0, :, 0].detach().cpu().numpy()
    xhat_np = x_hat[0, :, 0].detach().cpu().numpy()
    z_np = z[0].detach().cpu().numpy()
    zq_np = z_q[0].detach().cpu().numpy()

    # Color map for eigenvalue channels
    colors = plt.cm.viridis(np.linspace(0, 1, N))

    # ── Figure layout ──
    fig = plt.figure(figsize=(28, 24))
    fig.suptitle('S5 Codec — SSM Diagnostic Dashboard', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

    # ════════════════════════════════════════════════════════════
    # 1. EIGENVALUE MAP (Complex Plane)
    # ════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Decoder Eigenvalues (z-plane)', fontsize=11)

    # Unit circle
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=0.5, alpha=0.3)

    # Eigenvalues
    for i, lam in enumerate(lambdas):
        size = 30 + 200 * np.abs(C_weights[i])  # size proportional to output weight
        ax1.scatter(lam.real, lam.imag, c=[colors[i]], s=size,
                    edgecolors='black', linewidths=0.5, zorder=5)
        ax1.annotate(f'{i}', (lam.real, lam.imag), fontsize=6,
                     ha='center', va='bottom', color=colors[i])

    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.axhline(0, color='gray', linewidth=0.3)
    ax1.axvline(0, color='gray', linewidth=0.3)
    ax1.set_xlabel('Re(lambda)')
    ax1.set_ylabel('Im(lambda)')
    ax1.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 2. ENCODER EIGENVALUES (all layers)
    # ════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Encoder S5 Eigenvalues (discrete, z-plane)', fontsize=11)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=0.5, alpha=0.3)

    enc_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for info_layer in enc_eigs:
        layer_idx = info_layer['layer']
        lam_d = info_layer['lambda_discrete']
        col = enc_colors[layer_idx % len(enc_colors)]
        ax2.scatter(lam_d.real, lam_d.imag, c=col, s=30,
                    edgecolors='black', linewidths=0.3, alpha=0.7,
                    label=f'Layer {layer_idx}')

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='gray', linewidth=0.3)
    ax2.axvline(0, color='gray', linewidth=0.3)
    ax2.set_xlabel('Re(lambda_bar)')
    ax2.set_ylabel('Im(lambda_bar)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 3. BODE: MAGNITUDE RESPONSE
    # ════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Bode: Magnitude Response (per channel)', fontsize=11)

    for i in range(N):
        freq_hz, mag_db, _ = compute_bode(lambdas[i], C_weights[i], sample_rate)
        ax3.plot(freq_hz, mag_db, color=colors[i], linewidth=1.0,
                 alpha=0.8, label=f'ch{i}')

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_xlim(0, sample_rate / 2)
    ax3.legend(fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 4. BODE: PHASE RESPONSE
    # ════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title('Bode: Phase Response (per channel)', fontsize=11)

    for i in range(N):
        freq_hz, _, phase_rad = compute_bode(lambdas[i], C_weights[i], sample_rate)
        ax4.plot(freq_hz, np.degrees(phase_rad), color=colors[i],
                 linewidth=1.0, alpha=0.8)

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Phase (degrees)')
    ax4.set_xlim(0, sample_rate / 2)
    ax4.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 5. IMPULSE RESPONSES
    # ════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.set_title('Impulse Responses (per eigenvalue channel)', fontsize=11)

    for i in range(N):
        t_ir, h_ir = compute_impulse_response(lambdas[i], C_weights[i], n_steps=400)
        ax5.plot(t_ir, h_ir, color=colors[i], linewidth=1.0, alpha=0.7,
                 label=f'ch{i} |l|={np.abs(lambdas[i]):.3f} ang={np.degrees(np.angle(lambdas[i])):.0f}deg')

    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Amplitude')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 6. COMBINED IMPULSE (sum of all channels)
    # ════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.set_title('Combined SSM Impulse Response (all channels summed)', fontsize=11)

    combined_ir = np.zeros(400)
    for i in range(N):
        _, h_ir = compute_impulse_response(lambdas[i], C_weights[i], n_steps=400)
        combined_ir += h_ir

    ax6.plot(combined_ir, 'k-', linewidth=1.0)
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Amplitude')
    ax6.grid(True, alpha=0.2)

    # Also show its spectrum
    ax6b = ax6.twinx()
    spec = np.abs(np.fft.rfft(combined_ir))
    freqs = np.fft.rfftfreq(len(combined_ir), d=1.0 / sample_rate)
    ax6b.plot(freqs[:len(freqs)//4], 20 * np.log10(spec[:len(freqs)//4] + 1e-10),
              'r-', linewidth=0.7, alpha=0.5)
    ax6b.set_ylabel('|H(f)| dB', color='red')
    ax6b.tick_params(axis='y', labelcolor='red')

    # ════════════════════════════════════════════════════════════
    # 7. INPUT vs RECONSTRUCTION
    # ════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.set_title('Input vs Reconstruction (time domain)', fontsize=11)

    t_ms = np.arange(len(x_np)) / sample_rate * 1000
    ax7.plot(t_ms, x_np, 'b-', linewidth=0.8, alpha=0.7, label='Input')
    ax7.plot(t_ms, xhat_np, 'r-', linewidth=0.8, alpha=0.7, label='Reconstruction')
    snr = compute_snr(x_hat, x)
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Amplitude')
    ax7.legend(fontsize=8)
    ax7.set_title(f'Input vs Reconstruction -- SNR: {snr:.1f} dB', fontsize=11)
    ax7.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 8. INPUT vs RECONSTRUCTION (frequency domain)
    # ════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.set_title('Input vs Reconstruction (spectrum)', fontsize=11)

    X = np.fft.rfft(x_np)
    X_hat = np.fft.rfft(xhat_np)
    freqs_sig = np.fft.rfftfreq(len(x_np), d=1.0 / sample_rate)

    ax8.plot(freqs_sig, 20 * np.log10(np.abs(X) + 1e-10), 'b-',
             linewidth=0.8, alpha=0.7, label='Input')
    ax8.plot(freqs_sig, 20 * np.log10(np.abs(X_hat) + 1e-10), 'r-',
             linewidth=0.8, alpha=0.7, label='Reconstruction')

    spec_snr = compute_spectral_snr(x_hat, x)
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('Magnitude (dB)')
    ax8.set_title(f'Spectrum -- Spectral SNR: {spec_snr:.1f} dB', fontsize=11)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 9. STATE VECTOR (pre/post quantization)
    # ════════════════════════════════════════════════════════════
    ax9 = fig.add_subplot(gs[3, 0])
    ax9.set_title('Latent State: z vs z_q', fontsize=11)

    bar_width = 0.35
    indices = np.arange(len(z_np))
    ax9.bar(indices - bar_width / 2, z_np, bar_width, alpha=0.7, label='z (continuous)', color='tab:blue')
    ax9.bar(indices + bar_width / 2, zq_np, bar_width, alpha=0.7, label='z_q (quantized)', color='tab:orange')
    ax9.set_xlabel('Dimension')
    ax9.set_ylabel('Value')
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 10. EIGENVALUE PROPERTIES TABLE
    # ════════════════════════════════════════════════════════════
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.set_title('Decoder Eigenvalue Properties', fontsize=11)
    ax10.axis('off')

    table_data = []
    headers = ['Ch', '|l|', 'angle (deg)', 'f (Hz)', 't_half (steps)', '|C|']
    for i in range(N):
        r = np.abs(lambdas[i])
        theta = np.angle(lambdas[i])
        freq_hz = theta * sample_rate / (2 * np.pi)
        halflife = -np.log(2) / np.log(r + 1e-10)
        c_mag = np.abs(C_weights[i])
        table_data.append([
            f'{i}',
            f'{r:.4f}',
            f'{np.degrees(theta):.1f}',
            f'{freq_hz:.0f}',
            f'{halflife:.1f}',
            f'{c_mag:.4f}',
        ])

    table = ax10.table(cellText=table_data, colLabels=headers,
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # ════════════════════════════════════════════════════════════
    # 11. CHANNEL ENERGY DECOMPOSITION
    # ════════════════════════════════════════════════════════════
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.set_title('Channel Energy (decoder output)', fontsize=11)

    # Decompose the reconstruction into per-channel contributions
    decoder = model.decoder
    if hasattr(decoder, 'analytical'):
        decoder = decoder.analytical

    with torch.no_grad():
        zq_tensor = z_q.unsqueeze(0) if z_q.dim() == 1 else z_q[:1]
        h0_re = zq_tensor[:, :model.d_state]
        h0_im = zq_tensor[:, model.d_state:]
        h0 = torch.complex(h0_re, h0_im)
        lambdas_t = decoder.get_eigenvalues()
        C_t = torch.complex(decoder.C[:, 0], decoder.C[:, 1])

    channel_energies = []
    for i in range(N):
        t_arr = torch.arange(model.segment_length, dtype=torch.float32)
        channel_signal = (C_t[i].conj() * h0[0, i] * lambdas_t[i] ** t_arr).real.numpy()
        energy = np.sum(channel_signal ** 2)
        channel_energies.append(energy)

    channel_energies = np.array(channel_energies)
    ax11.bar(range(N), channel_energies, color=colors, edgecolor='black', linewidth=0.5)
    ax11.set_xlabel('Channel')
    ax11.set_ylabel('Energy')
    ax11.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════
    # 12. ENCODER STEP SIZES & EIGENVALUE MAGNITUDES
    # ════════════════════════════════════════════════════════════
    ax12 = fig.add_subplot(gs[3, 3])
    ax12.set_title('Encoder: Discrete |lambda_bar| & Step Sizes', fontsize=11)

    for info_layer in enc_eigs:
        layer_idx = info_layer['layer']
        lam_d = info_layer['lambda_discrete']
        col = enc_colors[layer_idx % len(enc_colors)]

        mags = np.abs(lam_d)
        ax12.scatter(range(len(mags)), mags, c=col, s=20, alpha=0.7,
                     label=f'L{layer_idx} |lambda_bar|')

    ax12.set_xlabel('Channel index')
    ax12.set_ylabel('|lambda_bar| (discrete)')
    ax12.legend(fontsize=7)
    ax12.grid(True, alpha=0.2)
    ax12.set_ylim(0, 1.1)
    ax12.axhline(1.0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

    # ── Save / Show ──
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")

    plt.show()
    plt.close()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S5 Codec Diagnostics')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--d_state', type=int, default=4)
    parser.add_argument('--decoder', type=str, default='analytical')
    parser.add_argument('--segment_length', type=int, default=1600)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_quantization_levels', type=int, default=16)
    parser.add_argument('--save', type=str, default='codec/diagnostics.png',
                        help='Path to save the plot')
    parser.add_argument('--phase', type=int, default=1,
                        help='Dataset phase for test signal')
    args = parser.parse_args()

    # ── Build or load model ──
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        model_cfg = cfg.get('model', {})

        model = S5Codec(
            d_state=model_cfg.get('d_state', args.d_state),
            segment_length=cfg.get('data', {}).get('segment_length', args.segment_length),
            n_encoder_layers=model_cfg.get('n_encoder_layers', 2),
            n_decoder_layers=model_cfg.get('n_decoder_layers', 1),
            d_hidden=model_cfg.get('d_hidden', 16),
            n_quantization_levels=model_cfg.get('n_quantization_levels', args.n_quantization_levels),
            decoder_type=model_cfg.get('decoder_type', args.decoder),
        )
        model.load_state_dict(ckpt['model_state'])
        segment_length = cfg.get('data', {}).get('segment_length', args.segment_length)
        phase = cfg.get('data', {}).get('phase', args.phase)
        sample_rate = cfg.get('data', {}).get('sample_rate', args.sample_rate)

        print(f"  Loaded epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('best_val_loss', '?')}, "
              f"val_snr={ckpt.get('val_snr', '?')}")
    else:
        print("No checkpoint -- using random initialization.")
        model = S5Codec(
            d_state=args.d_state,
            segment_length=args.segment_length,
            n_encoder_layers=2,
            n_decoder_layers=1,
            d_hidden=16,
            n_quantization_levels=args.n_quantization_levels,
            decoder_type=args.decoder,
        )
        segment_length = args.segment_length
        phase = args.phase
        sample_rate = args.sample_rate

    model.eval()

    # ── Generate test signal ──
    dataset = SyntheticSignalDataset(
        n_samples=8, segment_length=segment_length,
        sample_rate=sample_rate, phase=phase, seed=42,
    )
    x, _ = dataset[0]
    x = x.unsqueeze(0)  # (1, T, 1)

    # ── Forward pass ──
    with torch.no_grad():
        x_hat, quant_loss, info = model(x)
        z = model.encoder(x)
        z_q, _, _ = model.quantizer(z)

    # ── Print summary ──
    snr = compute_snr(x_hat, x)
    spec_snr = compute_spectral_snr(x_hat, x)
    n_params = sum(p.numel() for p in model.parameters())
    
    # Decoder eigenvalue analysis
    eig_info = extract_decoder_eigenvalues(model)

    print(f"\n{'='*50}")
    print(f"  S5 Codec Diagnostic Summary")
    print(f"{'='*50}")
    print(f"  Parameters:       {n_params:,}")
    print(f"  d_state:          {model.d_state}")
    print(f"  Segment length:   {segment_length}")
    print(f"  Quant levels:     {model.quantizer.n_levels}")
    print(f"  Bits/segment:     {info['bits_per_segment']:.0f}")
    print(f"  Bits/sample:      {info['bits_per_sample']:.4f}")
    print(f"")
    print(f"  Input range:      [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Input power:      {(x**2).mean():.6f}")
    print(f"  Output range:     [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    print(f"  Output power:     {(x_hat**2).mean():.6f}")
    print(f"")
    print(f"  Encoder z range:  [{z.min():.4f}, {z.max():.4f}]")
    print(f"  Encoder z std:    {z.std():.4f}")
    print(f"  Quantized z_q:    [{z_q.min():.4f}, {z_q.max():.4f}]")
    print(f"  Quantized z_q std:{z_q.std():.4f}")
    print(f"  Quant distortion: {info['quant_distortion']:.6f}")
    print(f"")
    print(f"  Decoder |λ| range: [{eig_info['magnitudes'].min():.4f}, {eig_info['magnitudes'].max():.4f}]")
    print(f"  Decoder |C| range: [{eig_info['C_magnitudes'].min():.4f}, {eig_info['C_magnitudes'].max():.4f}]")
    print(f"  Decoder |C| mean:  {eig_info['C_magnitudes'].mean():.4f}")
    print(f"")
    print(f"  SNR:              {snr:.2f} dB")
    print(f"  Spectral SNR:     {spec_snr:.2f} dB")
    print(f"{'='*50}\n")
    
    # ══════════════════════════════════════════════════════════
    # CRITICAL TEST: Decoder with known state
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("  DECODER TEST: Known State Input")
    print("="*50)
    
    with torch.no_grad():
        # Test 1: All zeros state → should output near-zero
        z_zeros = torch.zeros(1, 2 * model.d_state)
        x_from_zeros = model.decoder(z_zeros)
        power_from_zeros = (x_from_zeros ** 2).mean().item()
        print(f"  Zero state → output power: {power_from_zeros:.6f}")
        
        # Test 2: All ones state → should output something
        z_ones = torch.ones(1, 2 * model.d_state)
        x_from_ones = model.decoder(z_ones)
        power_from_ones = (x_from_ones ** 2).mean().item()
        print(f"  Ones state → output power: {power_from_ones:.6f}")
        
        # Test 3: Single channel excitation (channel 0 only)
        z_single = torch.zeros(1, 2 * model.d_state)
        z_single[0, 0] = 1.0  # Real part of first eigenvalue
        x_from_single = model.decoder(z_single)
        power_from_single = (x_from_single ** 2).mean().item()
        print(f"  Single channel (ch0 re=1) → output power: {power_from_single:.6f}")
        
        # Test 4: What's the actual encoded state producing?
        print(f"\n  Actual encoder output z:")
        print(f"    Range: [{z.min():.4f}, {z.max():.4f}]")
        print(f"    Mean:  {z.mean():.4f}")
        print(f"    Std:   {z.std():.4f}")
        print(f"    Nonzero dims: {(z.abs() > 0.01).sum().item()} / {z.numel()}")
        
        # Test 5: Bypass quantization to isolate the problem
        x_hat_no_quant, _, info_no_quant = model(x, bypass_quantization=True)
        snr_no_quant = compute_snr(x_hat_no_quant, x)
        print(f"\n  Bypass quantization test:")
        print(f"    SNR without quantization: {snr_no_quant:.2f} dB")
        print(f"    SNR with quantization:    {snr:.2f} dB")
        print(f"    → Quantization cost:      {snr_no_quant - snr:.2f} dB")
        
        if snr_no_quant < 0:
            print(f"\n  ⚠️  WARNING: Even without quantization, SNR is negative!")
            print(f"      This means the encoder-decoder pair hasn't learned to reconstruct.")
            print(f"      Possible causes:")
            print(f"        - Encoder outputs are too small (check z std)")
            print(f"        - Decoder C weights are too small (check |C| mean)")
            print(f"        - Skip connections in encoder dominating (encoder not learning)")
    
    print("="*50 + "\n")

    # ── Plot ──
    plot_diagnostics(
        model, x, x_hat, z, z_q, info,
        sample_rate=sample_rate,
        save_path=args.save,
    )
