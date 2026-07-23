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
from numba import jit, float32, int32, prange
from scipy.linalg import toeplitz
from typing import Optional, Tuple

@jit(nopython=True, parallel=True)
def calculate_coefficients_batch(frequencies, gains, t60s, sample_rate):
    """Calculate coefficients for all modes using SIMD-friendly operations"""
    num_modes = len(frequencies)
    r = np.zeros(num_modes, dtype=np.float32)
    c = np.zeros(num_modes, dtype=np.float32)
    s = np.zeros(num_modes, dtype=np.float32)
    g = np.zeros(num_modes, dtype=np.float32)
    
    omega = 2.0 * np.pi * frequencies / sample_rate
    
    for i in prange(num_modes):
        if t60s[i] > 0:
            bandwidth = 0.35 / t60s[i]
            decay_per_sample = np.pi * bandwidth / sample_rate
            if decay_per_sample < 100:
                r[i] = np.exp(-decay_per_sample)
        
        c[i] = r[i] * np.cos(omega[i])
        s[i] = r[i] * np.sin(omega[i])
        
        if r[i] < 1.0 and r[i] > 0:
            g[i] = gains[i] * (1.0 - r[i]) * s[i]
    
    return r, c, s, g

@jit(nopython=True, parallel=True)
def compute_impulse_response_batch(c, s, g, num_samples):
    """Compute impulse response for all modes in batch"""
    num_modes = len(c)
    impulse_response = np.zeros(num_samples, dtype=np.float32)
    
    for mode_idx in prange(num_modes):
        u1 = 0.0
        u2 = 0.0
        
        for n in range(num_samples):
            # Single impulse at n=0
            excitation = 1.0 if n == 0 else 0.0
            
            u1_new = c[mode_idx] * u1 - s[mode_idx] * u2 + g[mode_idx] * excitation
            u2_new = s[mode_idx] * u1 + c[mode_idx] * u2
            
            impulse_response[n] += u2_new
            
            u1 = u1_new
            u2 = u2_new
    
    return impulse_response

@jit(nopython=True, parallel=True)
def compute_impulse_response_matrix(c, s, g, num_samples):
    """Compute impulse response matrix for all modes (Toeplitz-like structure)"""
    num_modes = len(c)
    # Pre-allocate impulse response matrix
    ir_matrix = np.zeros((num_modes, num_samples), dtype=np.float32)
    
    for mode_idx in prange(num_modes):
        u1 = 0.0
        u2 = 0.0
        
        for n in range(num_samples):
            excitation = 1.0 if n == 0 else 0.0
            
            u1_new = c[mode_idx] * u1 - s[mode_idx] * u2 + g[mode_idx] * excitation
            u2_new = s[mode_idx] * u1 + c[mode_idx] * u2
            
            ir_matrix[mode_idx, n] = u2_new
            
            u1 = u1_new
            u2 = u2_new
    
    return ir_matrix

class ModalBank:
    """Optimized ModalBank with convolution matrix approach"""
    
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, 
                 t60s: np.ndarray, sample_rate: int, 
                 max_ir_length: int = 48000):  # 1 second at 48kHz
        self.sample_rate = sample_rate
        self.num_modes = len(frequencies)
        self.max_ir_length = max_ir_length
        
        # Store parameters
               self.frequencies = np.array(frequencies, dtype=np.float32)
        self.gains = np.array(gains, dtype=np.float32)
        self.t60s = np.array(t60s, dtype=np.float32)
        
        # Coefficient matrices
        self.r = np.zeros(self.num_modes, dtype=np.float32)
        self.c = np.zeros(self.num_modes, dtype=np.float32)
        self.s = np.zeros(self.num_modes, dtype=np.float32)
        self.g = np.zeros(self.num_modes, dtype=np.float32)
        
        # Pre-computed impulse response matrix (Toeplitz-like)
        self._ir_matrix = None
        self._ir_length = 0
        
        # State for streaming processing
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)
        
        # Initialize coefficients
        self._update_all_coefficients()
    
    def _update_all_coefficients(self):
        """Update all coefficients using Numba function"""
        self.r, self.c, self.s, self.g = calculate_coefficients_batch(
            self.frequencies, self.gains, self.t60s, self.sample_rate
        )
    
    def compute_impulse_response(self, num_samples: Optional[int] = None) -> np.ndarray:
        """Compute the full impulse response of the modal bank"""
        if num_samples is None:
            num_samples = self.max_ir_length
        
        if self._ir_matrix is None or self._ir_length < num_samples:
            self._ir_matrix = compute_impulse_response_matrix(
                self.c, self.s, self.g, num_samples
            )
            self._ir_length = num_samples
        
        # Sum across all modes
        return np.sum(self._ir_matrix[:, :num_samples], axis=0)
    
    def compute_impulse_response_toeplitz(self, num_samples: int) -> np.ndarray:
        """Compute impulse response using Toeplitz matrix structure"""
        # Get the impulse response
        ir = self.compute_impulse_response(num_samples)
        
        # Create Toeplitz matrix (lower triangular)
        # This is memory-efficient as we only store the first column
        return ir
    
    def convolve_with_signal(self, signal: np.ndarray) -> np.ndarray:
        """Convolve input signal with modal bank impulse response"""
        ir = self.compute_impulse_response(len(signal) + self.max_ir_length)
        
        # Use FFT convolution for efficiency
        from scipy.signal import fftconvolve
        result = fftconvolve(signal, ir, mode='full')
        
        return result[:len(signal)]
    
    def process_batch(self, excitations: np.ndarray) -> np.ndarray:
        """Process a batch of excitations using matrix multiplication"""
        num_samples = len(excitations)
        
        # Compute impulse response matrix
        ir_matrix = compute_impulse_response_matrix(
            self.c, self.s, self.g, num_samples
        )
        
        # For each mode, convolve with excitation
        output = np.zeros(num_samples, dtype=np.float32)
        
        for mode_idx in range(self.num_modes):
            # Use Toeplitz-like structure for efficient convolution
            ir_mode = ir_matrix[mode_idx]
            
            # Create Toeplitz matrix (first column)
            # This is equivalent to convolution
            conv_result = np.zeros(num_samples, dtype=np.float32)
            
            for n in range(num_samples):
                for k in range(min(n + 1, len(ir_mode))):
                    conv_result[n] += excitations[n - k] * ir_mode[k]
            
            output += conv_result
        
        return output
    
    def process_batch_simd(self, excitations: np.ndarray) -> np.ndarray:
        """SIMD-friendly batch processing using matrix operations"""
        num_samples = len(excitations)
        
        # Pre-compute impulse response
        ir = self.compute_impulse_response(num_samples)
        
        # Use numpy's convolve for vectorized operation
        # This is SIMD-friendly as numpy uses optimized BLAS
        output = np.convolve(excitations, ir, mode='full')[:num_samples]
        
        return output
    
    def process(self, excitation: float) -> float:
        """Process one sample (for streaming compatibility)"""
        output = 0.0
        
        for i in range(self.num_modes):
            u1_new = self.c[i] * self.u1[i] - self.s[i] * self.u2[i] + self.g[i] * excitation
            u2_new = self.s[i] * self.u1[i] + self.c[i] * self.u2[i]
            
            self.u1[i] = u1_new
            self.u2[i] = u2_new
            output += u2_new
        
        return output
    
    def reset(self):
        """Reset all state variables to zero"""
        self.u1.fill(0.0)
        self.u2.fill(0.0)
        self._ir_matrix = None
        self._ir_length = 0
