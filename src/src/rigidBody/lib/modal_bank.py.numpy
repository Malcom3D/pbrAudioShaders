# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from typing import Optional, Tuple

class ModalBank:
    """
    Fully vectorized modal bank implementation using coupled-form structure
    for maximum numerical stability and performance.
    """
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, t60s: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        self.num_modes = len(frequencies)

        # Store parameters
        self.frequencies = np.array(frequencies, dtype=np.float32)
        self.gains = np.array(gains, dtype=np.float32)
        self.t60s = np.array(t60s, dtype=np.float32)

        # Coupled-form coefficients
        self.r = np.zeros(self.num_modes, dtype=np.float32)  # radius (decay)
        self.c = np.zeros(self.num_modes, dtype=np.float32)  # cos(theta)
        self.s = np.zeros(self.num_modes, dtype=np.float32)  # sin(theta)
        self.g = np.zeros(self.num_modes, dtype=np.float32)  # gain scaling

        # Coupled-form state variables (u1, u2)
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)

        # Calculate initial coefficients
        self._calculate_all_coefficients()

    def _calculate_all_coefficients(self):
        """Calculate coupled-form coefficients for all modes"""
        # Angular frequency
        omega = 2.0 * np.pi * self.frequencies / self.sample_rate

        # Calculate decay factor r from T60
        with np.errstate(divide='ignore', invalid='ignore'):
            # r = exp(-π * bandwidth / sample_rate)
            # bandwidth = 2.2 / (T60 * 2π) = 0.35 / T60
            bandwidth = 0.35 / self.t60s
            bandwidth[self.t60s <= 0] = 0.0

            # Decay per sample
            decay_per_sample = np.pi * bandwidth / self.sample_rate
            self.r = np.exp(-decay_per_sample)
            self.r[np.isnan(self.r)] = 0.0
            self.r[self.t60s <= 0] = 0.0

        # Coupled-form rotation matrix elements
        self.c = self.r * np.cos(omega)  # r * cos(θ)
        self.s = self.r * np.sin(omega)  # r * sin(θ)

        # Gain scaling for unity peak gain at resonance
        # For coupled-form, output is taken from u2, so we scale by sin(θ)
        # Additional scaling by (1 - r) ensures proper gain
#        with np.errstate(divide='ignore', invalid='ignore'):
#            # Peak gain of coupled-form resonator: 1/(1 - r) at resonance
#            peak_gain = 1.0 / (1.0 - self.r)
#            peak_gain[np.isinf(peak_gain)] = 1.0
#
#            # Desired gain is self.gains, so scale accordingly
#            self.g = self.gains * self.s / peak_gain
#
        # Alternative simpler gain calculation:
        self.g = self.gains * (1.0 - self.r) * self.s

    def process(self, excitation: float) -> float:
        """
        Process one sample through all modes using coupled-form structure.

        Coupled-form equations:
        u1[n] = c * u1[n-1] - s * u2[n-1] + g * x[n]
        u2[n] = s * u1[n-1] + c * u2[n-1]
        y[n] = u2[n]
        """
        # Vectorized coupled-form update
        u1_new = self.c * self.u1 - self.s * self.u2 + self.g * excitation
        u2_new = self.s * self.u1 + self.c * self.u2

        # Update states
        self.u1 = u1_new
        self.u2 = u2_new

        # Sum outputs from all modes (output is u2)
        output = np.sum(u2_new)

        return output

    def reset(self):
        """Reset all state variables to zero"""
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)

    def update_frequency(self, mode_idx: int, frequency: float):
        """Update frequency for a specific mode"""
        self.frequencies[mode_idx] = frequency
        self._update_mode_coefficients(mode_idx)

    def update_gain(self, mode_idx: int, gain: float):
        """Update gain for a specific mode"""
        self.gains[mode_idx] = gain
        self._update_mode_coefficients(mode_idx)

    def update_t60(self, mode_idx: int, t60: float):
        """Update T60 for a specific mode"""
        self.t60s[mode_idx] = t60
        self._update_mode_coefficients(mode_idx)

    def _update_mode_coefficients(self, mode_idx: int):
        """Update coefficients for a single mode"""
        # Recalculate for this specific mode
        omega = 2.0 * np.pi * self.frequencies[mode_idx] / self.sample_rate

        # Calculate decay factor
        if self.t60s[mode_idx] > 0:
            bandwidth = 0.35 / self.t60s[mode_idx]
            decay_per_sample = np.pi * bandwidth / self.sample_rate
            self.r[mode_idx] = np.exp(-decay_per_sample)
        else:
            self.r[mode_idx] = 0.0

        # Update coupled-form coefficients
        self.c[mode_idx] = self.r[mode_idx] * np.cos(omega)
        self.s[mode_idx] = self.r[mode_idx] * np.sin(omega)

        # Update gain scaling
        if self.r[mode_idx] < 1.0:
            peak_gain = 1.0 / (1.0 - self.r[mode_idx])
            self.g[mode_idx] = self.gains[mode_idx] * self.s[mode_idx] / peak_gain
        else:
            self.g[mode_idx] = 0.0
