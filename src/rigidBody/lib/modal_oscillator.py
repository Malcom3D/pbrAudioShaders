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
import scipy.signal as signal
from typing import List, Tuple, Optional, Dict, Any

class ModalOscillator:
    """Single modal oscillator with damping and frequency"""
    def __init__(self, freq: float, decay: float, amplitude: float, sample_rate: int):
        """
        Initialize a modal oscillator
        
        Args:
            freq: Frequency in Hz
            decay: Decay time in seconds (time to decay to 1/e)
            amplitude: Initial amplitude
            sample_rate: output sample rate
        """
        self.freq = freq
        self.decay = decay
        self.amplitude = amplitude
        
        # State variables
        self.y = 0.0  # Current output
        self.y_prev = 0.0  # Previous output
        
        # Calculate filter coefficients
        self.calculate_coefficients(sample_rate)
    
    def calculate_coefficients(self, sample_rate: int):
        """Calculate digital filter coefficients for the oscillator"""
        self.sample_rate = sample_rate
        
        # Convert to digital domain
        omega = 2 * np.pi * self.freq / sample_rate
        sigma = 1.0 / (self.decay * sample_rate) if self.decay > 0 else 0
        
        # Pole radius and angle
        r = np.exp(-sigma)
        theta = omega
        
        # Complex conjugate poles
        self.b = [self.amplitude, 0]  # Numerator coefficients
        self.a = [1, -2 * r * np.cos(theta), r * r]  # Denominator coefficients

        # Initialize filter state
        self.zi = signal.lfilter_zi(self.b, self.a)

    def process(self, excitation: float) -> float:
        """Process one sample of excitation through the oscillator"""
        # Use lfilter for stability
        self.y, self.zi = signal.lfilter(self.b, self.a, [excitation], zi=self.zi)
        return self.y[0]

    def reset(self):
        """Reset oscillator state"""
        self.zi = signal.lfilter_zi(self.b, self.a)

class ModalBank:
    """Bank of modal oscillators for physical modeling synthesis"""
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, t60s: np.ndarray, sample_rate: int):
        """
        Initialize modal bank with list of (freq, decay, amplitude) tuples

        Args:
            modes: List of modal parameters (frequency, decay_time, amplitude)
        """
        self.oscillators = [ModalOscillator(frequencies[idx], gains[idx], t60s[idx], sample_rate) for idx in range(len(frequencies))]
        self.sample_rate = sample_rate

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate for all oscillators"""
        self.sample_rate = sample_rate
        for osc in self.oscillators:
            osc.calculate_coefficients(sample_rate)

    def process(self, excitation: float) -> float:
        """Process excitation through all oscillators and sum outputs"""
        output = 0.0
        for osc in self.oscillators:
            output += osc.process(excitation)
        return output

    def process_buffer(self, excitation_buffer: np.ndarray) -> np.ndarray:
        """Process a buffer of excitation samples"""
        output = np.zeros_like(excitation_buffer)
        for i, exc in enumerate(excitation_buffer):
            output[i] = self.process(exc)
        return output

    def reset(self):
        """Reset all oscillators"""
        for osc in self.oscillators:
            osc.reset()
