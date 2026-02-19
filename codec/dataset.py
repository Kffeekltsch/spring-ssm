"""
Synthetic signal dataset for the S5 Codec experiment.

Generates signals from multiple classes covering the space of
"things an SSM should be good at compressing":

Phase 1: Single sinusoids (atomic test case)
Phase 2: Sum of sinusoids (superposition)
Phase 3: Decaying sinusoids (eigenvalue magnitude matters)
Phase 4: Full mix (chirps, steps, noise, AM/FM, real-world-like)

Each signal type tests a specific SSM capability.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple


class SyntheticSignalDataset(Dataset):
    """
    Generates normalized signal segments for autoencoder training.
    Returns (signal, signal) pairs — input = target.
    
    Compatible with the existing spring-ssm DataLoader pattern:
    returns (x, x) shaped as (T, 1).
    """
    def __init__(self,
                 n_samples: int = 10000,
                 segment_length: int = 1600,
                 sample_rate: int = 16000,
                 phase: int = 4,
                 seed: int = 42):
        """
        Args:
            n_samples: number of segments to generate
            segment_length: samples per segment
            sample_rate: for frequency calculations
            phase: curriculum phase (1-4), controls signal complexity
            seed: random seed for reproducibility
        """
        super().__init__()
        self.n_samples = n_samples
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.phase = phase

        # Pre-generate all signals for fast training
        rng = np.random.RandomState(seed)
        self.signals = []
        self.labels = []

        for i in range(n_samples):
            signal, label = self._generate_signal(rng)
            self.signals.append(signal)
            self.labels.append(label)

        self.signals = np.stack(self.signals)  # (n_samples, segment_length)

    def _generate_signal(self, rng) -> Tuple[np.ndarray, str]:
        """Generate one signal segment based on curriculum phase."""
        t = np.linspace(0, self.segment_length / self.sample_rate,
                        self.segment_length, endpoint=False)

        if self.phase == 1:
            # Phase 1: single sinusoid — if SSM can't do this, nothing works
            return self._single_sine(t, rng)

        elif self.phase == 2:
            # Phase 2: sum of sinusoids — test superposition
            choice = rng.randint(0, 2)
            if choice == 0:
                return self._single_sine(t, rng)
            else:
                return self._sum_of_sines(t, rng)

        elif self.phase == 3:
            # Phase 3: add decaying sinusoids — eigenvalue magnitudes matter
            choice = rng.randint(0, 3)
            if choice == 0:
                return self._single_sine(t, rng)
            elif choice == 1:
                return self._sum_of_sines(t, rng)
            else:
                return self._decaying_sines(t, rng)

        else:
            # Phase 4: full complexity
            choice = rng.randint(0, 7)
            generators = [
                self._single_sine,
                self._sum_of_sines,
                self._decaying_sines,
                self._chirp,
                self._brownian,
                self._step_function,
                self._am_fm,
            ]
            return generators[choice](t, rng)

    def _single_sine(self, t, rng):
        # Narrower frequency range to match d_state capacity
        freq = rng.uniform(100, 2000)  # Speech-like range
        amp = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        return self._normalize(signal), 'single_sine'

    def _sum_of_sines(self, t, rng):
        n_sines = rng.randint(2, 12)
        signal = np.zeros_like(t)
        for _ in range(n_sines):
            freq = rng.uniform(50, self.sample_rate / 2 * 0.8)
            amp = rng.exponential(0.3)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        return self._normalize(signal), 'sum_of_sines'

    def _decaying_sines(self, t, rng):
        n_comp = rng.randint(2, 10)
        signal = np.zeros_like(t)
        for _ in range(n_comp):
            freq = rng.uniform(50, self.sample_rate / 2 * 0.7)
            decay = rng.uniform(1.0, 80.0)
            amp = rng.uniform(0.3, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)
        return self._normalize(signal), 'decaying_sines'

    def _chirp(self, t, rng):
        f0 = rng.uniform(50, 500)
        f1 = rng.uniform(500, self.sample_rate / 2 * 0.8)
        if rng.rand() > 0.5:
            f0, f1 = f1, f0  # down-chirp
        T = t[-1]
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * T))
        signal = np.sin(phase)
        # Optional envelope
        if rng.rand() > 0.5:
            env_decay = rng.uniform(0.5, 5.0)
            signal *= np.exp(-env_decay * t)
        return self._normalize(signal), 'chirp'

    def _brownian(self, t, rng):
        increments = rng.randn(len(t)) * 0.01
        signal = np.cumsum(increments)
        return self._normalize(signal), 'brownian'

    def _step_function(self, t, rng):
        n_steps = rng.randint(2, 8)
        positions = np.sort(rng.uniform(t[0], t[-1], n_steps))
        levels = rng.randn(n_steps + 1)
        signal = np.zeros_like(t)
        # Smooth steps
        sharpness = rng.uniform(20, 200)
        for i, pos in enumerate(positions):
            signal += (levels[i + 1] - levels[i]) / (1 + np.exp(-sharpness * (t - pos)))
        signal += levels[0]
        return self._normalize(signal), 'step'

    def _am_fm(self, t, rng):
        carrier_freq = rng.uniform(200, self.sample_rate / 2 * 0.5)
        mod_freq = rng.uniform(1, 50)
        mod_depth = rng.uniform(0.2, 0.9)
        # AM
        envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        # Optional FM
        fm_depth = rng.uniform(0, 50) if rng.rand() > 0.5 else 0
        phase = 2 * np.pi * carrier_freq * t + fm_depth * np.sin(2 * np.pi * mod_freq * 0.7 * t)
        signal = envelope * np.sin(phase)
        return self._normalize(signal), 'am_fm'

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] range."""
        peak = np.max(np.abs(signal))
        if peak > 1e-7:
            signal = signal / peak
        return signal.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns (signal, signal) — autoencoder format.
        Shape: (T, 1) to match existing spring-ssm convention.
        """
        signal = torch.from_numpy(self.signals[idx]).float()
        signal = signal.unsqueeze(-1)  # (T, 1)
        return signal, signal


def create_dataloaders(
    n_train: int = 8000,
    n_val: int = 1000,
    n_test: int = 1000,
    segment_length: int = 1600,
    sample_rate: int = 16000,
    phase: int = 4,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Create train/val/test dataloaders for the codec experiment."""
    train_set = SyntheticSignalDataset(
        n_samples=n_train, segment_length=segment_length,
        sample_rate=sample_rate, phase=phase, seed=42,
    )
    val_set = SyntheticSignalDataset(
        n_samples=n_val, segment_length=segment_length,
        sample_rate=sample_rate, phase=phase, seed=1337,
    )
    test_set = SyntheticSignalDataset(
        n_samples=n_test, segment_length=segment_length,
        sample_rate=sample_rate, phase=phase, seed=9999,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
