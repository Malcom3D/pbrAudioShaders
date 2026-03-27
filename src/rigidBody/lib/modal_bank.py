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
from numba import float32, int32
from numba.experimental import jitclass
from typing import Optional, Tuple

# Define the spec for jitclass
spec = [
    ('sample_rate', int32),
    ('num_modes', int32),
    ('frequencies', float32[:]),
    ('gains', float32[:]),
    ('t60s', float32[:]),
    ('r', float32[:]),
    ('c', float32[:]),
    ('s', float32[:]),
    ('g', float32[:]),
    ('u1', float32[:]),
    ('u2', float32[:]),
]

@jitclass(spec)
class ModalBank:
    """
    Fully vectorized modal bank implementation using coupled-form structure
    for maximum numerical stability and performance.
    Numba-accelerated version.
    """
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, 
                 t60s: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        self.num_modes = len(frequencies)

        # Store parameters
        self.frequencies = np.array(frequencies, dtype=np.float32)
        self.gains = np.array(gains, dtype=np.float32)
        self.t60s = np.array(t60s, dtype=np.float32)

        # Coupled-form coefficients
        self.r = np.zeros(self.num_modes, dtype=np.float32)
        self.c = np.zeros(self.num_modes, dtype=np.float32)
        self.s = np.zeros(self.num_modes, dtype=np.float32)
        self.g = np.zeros(self.num_modes, dtype=np.float32)

        # Coupled-form state variables
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)

        # Calculate initial coefficients
        self._calculate_all_coefficients()

    def _calculate_all_coefficients(self):
        """Calculate coupled-form coefficients for all modes"""
        # Angular frequency
        omega = 2.0 * np.pi * self.frequencies / self.sample_rate

        # Calculate decay factor r from T60
        bandwidth = np.zeros(self.num_modes, dtype=np.float32)
        for i in range(self.num_modes):
            if self.t60s[i] > 0:
                bandwidth[i] = 0.35 / self.t60s[i]
            else:
                bandwidth[i] = 0.0

        # Decay per sample
        decay_per_sample = np.pi * bandwidth / self.sample_rate
        
        # Handle potential NaN/inf values
        for i in range(self.num_modes):
            if decay_per_sample[i] < 100:  # Avoid overflow
                self.r[i] = np.exp(-decay_per_sample[i])
            else:
                self.r[i] = 0.0
            
            if self.t60s[i] <= 0:
                self.r[i] = 0.0

        # Coupled-form rotation matrix elements
        for i in range(self.num_modes):
            self.c[i] = self.r[i] * np.cos(omega[i])
            self.s[i] = self.r[i] * np.sin(omega[i])
            
            # Gain scaling
            if self.r[i] < 1.0 and self.r[i] > 0:
                self.g[i] = self.gains[i] * (1.0 - self.r[i]) * self.s[i]
            else:
                self.g[i] = 0.0

    def process(self, excitation: float) -> float:
        """
        Process one sample through all modes using coupled-form structure.
        """
        output = 0.0
        
        # Vectorized update - manually unrolled for Numba
        for i in range(self.num_modes):
            u1_new = self.c[i] * self.u1[i] - self.s[i] * self.u2[i] + self.g[i] * excitation
            u2_new = self.s[i] * self.u1[i] + self.c[i] * self.u2[i]
            
            self.u1[i] = u1_new
            self.u2[i] = u2_new
            
            output += u2_new
        
        return output

    def reset(self):
        """Reset all state variables to zero"""
        for i in range(self.num_modes):
            self.u1[i] = 0.0
            self.u2[i] = 0.0

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
        omega = 2.0 * np.pi * self.frequencies[mode_idx] / self.sample_rate

        # Calculate decay factor
        if self.t60s[mode_idx] > 0:
            bandwidth = 0.35 / self.t60s[mode_idx]
            decay_per_sample = np.pi * bandwidth / self.sample_rate
            if decay_per_sample < 100:
                self.r[mode_idx] = np.exp(-decay_per_sample)
            else:
                self.r[mode_idx] = 0.0
        else:
            self.r[mode_idx] = 0.0

        # Update coupled-form coefficients
        self.c[mode_idx] = self.r[mode_idx] * np.cos(omega)
        self.s[mode_idx] = self.r[mode_idx] * np.sin(omega)

        # Update gain scaling
        if self.r[mode_idx] < 1.0 and self.r[mode_idx] > 0:
            self.g[mode_idx] = self.gains[mode_idx] * (1.0 - self.r[mode_idx]) * self.s[mode_idx]
        else:
            self.g[mode_idx] = 0.0
