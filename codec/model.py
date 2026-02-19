"""
S5 Dynamics Codec — learned compression via state-space bottleneck.

Architecture:
    Encoder:  S5 processes input segment → final hidden state h(T) ∈ ℂ^N
    Bottleneck: Scalar quantization — each dimension independently quantized
                to n_levels. Transmitted as a bit vector.
    Decoder:  (a) Analytical: autonomous SSM rollout y(t) = Re[C · Λ^t · h₀]
              (b) Learned: analytical + S5 refinement conditioned on state
              (c) S5: pure S5 decoder — state vector broadcast across time,
                  S5 generates waveform. Symmetric to encoder.

The compressed representation IS the quantized SSM state.
Bitrate = 2 * d_state * bits_per_dim / segment_length  bits/sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent paths so we can import the existing SSM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ssm'))

from ssm.src.model.ssm import SSM, as_complex, discretize_zoh
from ssm.src.model.sequence_layer import SequenceLayer


# ============================================================
# Scalar Quantizer — independent per-dimension quantization
# ============================================================

class ScalarQuantizer(nn.Module):
    """
    Uniform scalar quantization with straight-through gradient.
    
    Each dimension of the input is independently quantized to one of
    n_levels uniform levels in a learned [min, max] range.
    
    The "bitstream" is just the integer level per dimension:
        bits_per_segment = n_dims * ceil(log2(n_levels))
    
    No codebook collapse possible — every combination is reachable.
    """
    def __init__(self, n_dims: int, n_levels: int = 16):
        super().__init__()
        self.n_dims = n_dims
        self.n_levels = n_levels
        self.bits_per_dim = np.ceil(np.log2(n_levels))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D) continuous latent vectors (assumed roughly in [-1, 1] after tanh)
        Returns:
            z_q: (B, D) quantized vectors (straight-through)
            quant_loss: scalar quantization distortion (for monitoring, not backprop'd)
            indices: (B, D) integer level indices
            info: dict
        """
        # z is already in [-1, 1] from tanh in encoder
        # Map to [0, n_levels-1], round, map back
        z_scaled = (z + 1.0) / 2.0 * (self.n_levels - 1)  # [0, n_levels-1]
        z_rounded = torch.round(z_scaled)                    # integer levels
        z_rounded = z_rounded.clamp(0, self.n_levels - 1)    # safety
        
        # Map back to [-1, 1]
        z_q = z_rounded / (self.n_levels - 1) * 2.0 - 1.0
        
        # Straight-through: forward uses quantized, backward uses continuous
        z_q_st = z + (z_q - z).detach()
        
        # Quantization distortion (for logging)
        quant_loss = F.mse_loss(z_q, z)
        
        indices = z_rounded.long()
        
        return z_q_st, quant_loss, indices

    def bits_per_segment(self):
        return self.n_dims * self.bits_per_dim


# ============================================================
# Encoder: S5 processes signal → final state = compressed repr.
# ============================================================

class S5Encoder(nn.Module):
    """
    Runs an S5 stack over the input signal and extracts the final
    hidden state as the compressed representation.
    
    Input:  (B, T, 1) signal segment
    Output: (B, 2*d_state) real-valued state vector (re/im concatenated)
    """
    def __init__(self, d_state: int = 64, n_layers: int = 2, d_hidden: int = 16):
        super().__init__()
        self.d_state = d_state
        self.d_hidden = d_hidden

        # Lift input to hidden dimension
        self.input_proj = nn.Linear(1, d_hidden)

        # S5 encoder stack
        self.layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_hidden,
                d_out=d_hidden,
                d_state=d_state,
                act='gelu',
                norm=True,
                norm_type='ln',
            )
            for _ in range(n_layers)
        ])

        # Project to state-sized bottleneck
        # Output dim = 2 * d_state (real + imag parts)
        self.state_proj = nn.Linear(d_hidden, 2 * d_state)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, 1) input signal
        Returns:
            state: (B, 2*d_state) compressed representation in [-1, 1]
        """
        # Project input
        h = self.input_proj(x)  # (B, T, d_hidden)

        # Run through S5 layers
        for layer in self.layers:
            h = layer(h)  # (B, T, d_hidden)

        # Take final timestep as compressed representation
        h_final = h[:, -1, :]  # (B, d_hidden)

        # Project to bottleneck size, bound to [-1, 1] for scalar quantization
        state = torch.tanh(self.state_proj(h_final))  # (B, 2*d_state)

        return state


# ============================================================
# Analytical Decoder: SSM rollout from initial state
# ============================================================

class AnalyticalDecoder(nn.Module):
    """
    Decoder that reconstructs signal by rolling out a diagonal SSM
    from the initial state. No learned recurrence — just:

        h_i(t) = λ_i^t · h_i(0)
        y(t) = Re[ C · h(t) ]

    This maps directly to analog hardware (bank of biquad filters).
    
    The eigenvalues and C matrix are learnable parameters.
    """
    def __init__(self, d_state: int = 64, segment_length: int = 1600):
        super().__init__()
        self.d_state = d_state
        self.segment_length = segment_length

        # Learnable eigenvalues in (log_magnitude, angle) parameterization
        #
        # We parameterize |λ| = 1 - softplus(neg_log_one_minus_r)
        # so that |λ| < 1 is strictly enforced and gradient doesn't just
        # push everything to |λ|=1.
        #
        # Init: spread magnitudes across [0.95, 0.9999] (long-lived modes)
        # At 44.1kHz with seg=1000: |λ|=0.95 → halflife=14 samples (ok)
        #                           |λ|=0.9999 → halflife=6931 samples (persistent)
        # Need most modes to survive the full segment length.
        target_r = torch.linspace(0.95, 0.9999, d_state)
        # neg_log_one_minus_r such that 1 - softplus(x) = target_r  →  softplus(x) = 1 - target_r
        # softplus(x) = log(1 + exp(x)), so x = log(exp(1-r) - 1)
        init_neg_log = torch.log(torch.exp(1.0 - target_r) - 1.0)
        self.neg_log_one_minus_r = nn.Parameter(init_neg_log)
        self.angle = nn.Parameter(torch.linspace(0.1, np.pi - 0.1, d_state))

        # Output projection C: complex, stored as (d_state, 2)
        # Output y(t) = Re[Σ C_i · λ^t · h0_i] sums d_state modes.
        # For unit-scale output: std(C) = 1/sqrt(d_state)
        # Then Var(y) = d_state * std(C)^2 * std(h)^2 ≈ 1
        c_std = 1.0 / (d_state ** 0.5)
        C_real = torch.randn(d_state) * c_std
        C_imag = torch.randn(d_state) * c_std
        self.C = nn.Parameter(torch.stack([C_real, C_imag], dim=-1))  # (d_state, 2)

        # Learnable output scale and bias — lets the decoder match target amplitude
        # without needing C weights to be exactly right at init.
        # In analog hardware this is just a final op-amp gain stage.
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def get_eigenvalues(self):
        """Compute complex eigenvalues from parameterization.
        
        |λ| = 1 - softplus(neg_log_one_minus_r), strictly < 1
        angle = abs(angle), frequency ≥ 0
        λ = r · e^(jθ)
        """
        one_minus_r = F.softplus(self.neg_log_one_minus_r).clamp(min=1e-4, max=0.999)
        r = 1.0 - one_minus_r  # magnitude ∈ (0.001, 0.9999)
        theta = torch.abs(self.angle)       # frequency ≥ 0
        # λ = r · e^(jθ)
        lambdas = torch.complex(r * torch.cos(theta), r * torch.sin(theta))
        return lambdas

    def forward(self, initial_state: torch.Tensor):
        """
        Args:
            initial_state: (B, 2*d_state) — real-valued (re, im concatenated)
        Returns:
            reconstruction: (B, T, 1) signal
        """
        B = initial_state.shape[0]

        # Split into complex state
        h0_real = initial_state[:, :self.d_state]  # (B, N)
        h0_imag = initial_state[:, self.d_state:]  # (B, N)
        h0 = torch.complex(h0_real, h0_imag)       # (B, N)

        # Get eigenvalues
        lambdas = self.get_eigenvalues()  # (N,)

        # Compute λ^t for all timesteps: (N, T)
        t = torch.arange(self.segment_length, device=initial_state.device, dtype=torch.float32)
        lambda_powers = lambdas.unsqueeze(-1) ** t.unsqueeze(0)  # (N, T)

        # State evolution: h(t) = λ^t · h(0) → (B, N, T)
        states = h0.unsqueeze(-1) * lambda_powers.unsqueeze(0)  # (B, N, T)

        # Output: y(t) = scale * Re[C^H · h(t)] + bias
        C_complex = torch.complex(self.C[:, 0], self.C[:, 1])  # (N,)
        y = torch.einsum('n,bnt->bt', C_complex.conj(), states).real  # (B, T)
        y = self.output_scale * y + self.output_bias

        return y.unsqueeze(-1)  # (B, T, 1)

    def get_eigenvalue_info(self):
        """Return interpretable eigenvalue diagnostics."""
        with torch.no_grad():
            lambdas = self.get_eigenvalues()
            magnitudes = torch.abs(lambdas)
            angles = torch.angle(lambdas)
            halflives = -np.log(2) / torch.log(magnitudes + 1e-10)
            return {
                'magnitudes': magnitudes.cpu().numpy(),
                'angles': angles.cpu().numpy(),
                'halflives': halflives.cpu().numpy(),
                'frequencies_normalized': (angles / np.pi).cpu().numpy(),
                'output_scale': self.output_scale.item(),
                'output_bias': self.output_bias.item(),
            }


# ============================================================
# S5 Decoder: pure S5 decoding — symmetric to encoder
# ============================================================

class S5Decoder(nn.Module):
    """
    Pure S5 decoder — no analytical rollout. Symmetric to encoder.
    
    The quantized state vector is broadcast across time as a constant
    input sequence. The S5 stack generates temporal structure via its
    own learned dynamics (eigenvalues, C matrices, skip connections).
    
    This is much more expressive than the analytical decoder because:
    - S5 layers have skip connections → can represent arbitrary mappings
    - Multiple layers compose to approximate complex waveforms
    - Each layer has its OWN learned SSM dynamics (not tied to encoder)
    - The GELU activations break linearity → can approximate any waveform
    
    The encoder's job is just to compress the signal into a compact
    state vector. The decoder's job is to expand it back. Symmetric.
    """
    def __init__(self, d_state: int = 64, segment_length: int = 1600,
                 d_hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.d_state = d_state
        self.segment_length = segment_length

        # Project state vector (2*d_state) to hidden dim
        self.input_proj = nn.Linear(2 * d_state, d_hidden)

        # S5 decoder stack — generates temporal structure
        self.layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_hidden,
                d_out=d_hidden,
                d_state=d_state,
                act='gelu',
                norm=True,
                norm_type='ln',
            )
            for _ in range(n_layers)
        ])

        # Project to output waveform
        self.output_proj = nn.Linear(d_hidden, 1)

    def forward(self, initial_state: torch.Tensor):
        """
        Args:
            initial_state: (B, 2*d_state) — quantized state vector
        Returns:
            reconstruction: (B, T, 1) signal
        """
        B = initial_state.shape[0]
        T = self.segment_length

        # Broadcast state across time: (B, 2*d_state) → (B, T, 2*d_state)
        state_seq = initial_state.unsqueeze(1).expand(B, T, -1)

        # Project to hidden dim
        h = self.input_proj(state_seq)  # (B, T, d_hidden)

        # S5 stack generates temporal variation from constant input
        for layer in self.layers:
            h = layer(h)  # (B, T, d_hidden)

        # Project to waveform
        y = self.output_proj(h)  # (B, T, 1)

        return y

    def get_eigenvalue_info(self):
        """
        Return diagnostics from the S5 layers' own SSM parameters.
        These are the decoder's learned dynamics — different from the
        analytical decoder's eigenvalues but still informative.
        """
        info = {'decoder_type': 's5'}
        for i, layer in enumerate(self.layers):
            mimo_ssm = layer.s5  # the MIMOSSM module inside SequenceLayer
            ssm = mimo_ssm.seq   # the actual SSM inside MIMOSSM
            with torch.no_grad():
                # Compute Delta (discretization step size)
                Delta = ssm.step_scale * torch.exp(ssm.log_step)
                
                # Get the SSM's discretized eigenvalues
                Lambda_bar, _ = discretize_zoh(
                    as_complex(ssm.Lambda),
                    as_complex(ssm.B),
                    as_complex(ssm.B_bias),
                    Delta,
                    ssm.input_bias
                )
                mags = torch.abs(Lambda_bar)
                angles = torch.angle(Lambda_bar)
                halflives = -np.log(2) / torch.log(mags.clamp(min=1e-10))
                info[f'layer_{i}_magnitudes'] = mags.cpu().numpy()
                info[f'layer_{i}_angles'] = angles.cpu().numpy()
                info[f'layer_{i}_halflives'] = halflives.cpu().numpy()
        return info


# ============================================================
# Learned Decoder: S5 block refines analytical reconstruction
# ============================================================

class LearnedDecoder(nn.Module):
    """
    Two-stage decoder:
    1. Analytical rollout (sum of decaying sinusoids)
    2. Learned S5 refinement conditioned on BOTH the analytical
       output AND the state vector h0
    
    The state vector is broadcast across time as a conditioning signal,
    giving the S5 refinement access to information that the analytical
    decoder couldn't express (transients, noise, etc.)
    """
    def __init__(self, d_state: int = 64, segment_length: int = 1600,
                 d_hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.d_state = d_state
        self.analytical = AnalyticalDecoder(d_state, segment_length)

        # Project analytical output (1) + state conditioning (2*d_state) → d_hidden
        self.refine_proj = nn.Linear(1 + 2 * d_state, d_hidden)
        self.refine_layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_hidden,
                d_out=d_hidden,
                d_state=d_state // 2,
                act='gelu',
                norm=True,
                norm_type='ln',
            )
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_hidden, 1)

    def forward(self, initial_state: torch.Tensor):
        """
        Args:
            initial_state: (B, 2*d_state)
        Returns:
            reconstruction: (B, T, 1)
        """
        B = initial_state.shape[0]
        T = self.analytical.segment_length

        # Stage 1: analytical rollout
        y_analytical = self.analytical(initial_state)  # (B, T, 1)

        # Stage 2: refine with S5, conditioned on state vector
        # Broadcast state across time: (B, 2*d_state) → (B, T, 2*d_state)
        state_cond = initial_state.unsqueeze(1).expand(B, T, -1)
        
        # Concatenate: [analytical_output, state_conditioning]
        refine_input = torch.cat([y_analytical, state_cond], dim=-1)  # (B, T, 1+2*d_state)
        
        h = self.refine_proj(refine_input)  # (B, T, d_hidden)
        for layer in self.refine_layers:
            h = layer(h)
        y_residual = self.output_proj(h)  # (B, T, 1)

        return y_analytical + y_residual

    def get_eigenvalue_info(self):
        return self.analytical.get_eigenvalue_info()


# ============================================================
# Full Codec
# ============================================================

class S5Codec(nn.Module):
    """
    Full S5-based signal codec.
    
    Signal → Encoder (S5) → Scalar Quantization → Decoder (analytical, learned, or s5) → Reconstruction
    
    The compressed bitstream is the quantized state vector (integer levels per dim).
    Bitrate = 2 * d_state * bits_per_dim / segment_length  bits/sample
    """
    def __init__(self,
                 d_state: int = 64,
                 segment_length: int = 1600,
                 n_encoder_layers: int = 2,
                 n_decoder_layers: int = 2,
                 d_hidden: int = 16,
                 n_quantization_levels: int = 16,
                 decoder_type: str = 's5',  # 'analytical', 'learned', or 's5'
                 # Legacy VQ params — ignored but accepted for config compat
                 n_vq_stages: int = 2,
                 n_vq_codes: int = 256,
                 commitment_cost: float = 0.25,
                 ):
        super().__init__()
        self.d_state = d_state
        self.segment_length = segment_length
        self.decoder_type = decoder_type

        # Encoder
        self.encoder = S5Encoder(
            d_state=d_state,
            n_layers=n_encoder_layers,
            d_hidden=d_hidden,
        )

        # Bottleneck — scalar quantization
        self.quantizer = ScalarQuantizer(
            n_dims=2 * d_state,
            n_levels=n_quantization_levels,
        )

        # Decoder
        if decoder_type == 'analytical':
            self.decoder = AnalyticalDecoder(d_state, segment_length)
        elif decoder_type == 'learned':
            self.decoder = LearnedDecoder(
                d_state, segment_length,
                d_hidden=d_hidden,
                n_layers=n_decoder_layers,
            )
        elif decoder_type == 's5':
            self.decoder = S5Decoder(
                d_state, segment_length,
                d_hidden=d_hidden,
                n_layers=n_decoder_layers,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, x: torch.Tensor, bypass_quantization: bool = False):
        """
        Args:
            x: (B, T, 1) input signal segment
            bypass_quantization: if True, skip quantization (for warmup)
        Returns:
            x_hat: (B, T, 1) reconstructed signal
            quant_loss: scalar quantization distortion (for monitoring)
            info: dict with diagnostics
        """
        # Encode → z ∈ [-1, 1]^D
        z = self.encoder(x)  # (B, 2*d_state)

        if bypass_quantization:
            z_q = z
            quant_loss = torch.tensor(0.0, device=z.device)
            indices = torch.zeros_like(z, dtype=torch.long)
        else:
            z_q, quant_loss, indices = self.quantizer(z)

        # Decode
        x_hat = self.decoder(z_q)  # (B, T, 1)

        info = {
            'quant_distortion': quant_loss.item(),
            'indices': indices,
            'z_pre_quant': z,
            'z_post_quant': z_q,
            'bits_per_segment': self.quantizer.bits_per_segment(),
            'bits_per_sample': self.quantizer.bits_per_segment() / self.segment_length,
            'bypass_quantization': bypass_quantization,
        }

        return x_hat, quant_loss, info

    def encode(self, x: torch.Tensor):
        """Encode-only for inference. Returns integer level indices."""
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor):
        """Decode from quantization indices (receiver side)."""
        # Map integer levels back to [-1, 1]
        z_q = indices.float() / (self.quantizer.n_levels - 1) * 2.0 - 1.0
        return self.decoder(z_q)

    def get_eigenvalue_info(self):
        """Return decoder eigenvalue diagnostics."""
        return self.decoder.get_eigenvalue_info()
