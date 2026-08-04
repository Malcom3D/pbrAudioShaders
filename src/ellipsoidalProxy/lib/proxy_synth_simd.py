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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager

from .proxy_ir_table import ProxyIRTable

@dataclass
class ProxySynthSIMD:
    """
    Lightweight physically-based synthesizer for proxy meshes.
    
    Uses precomputed IR tables and FFT convolution for fast synthesis.
    All operations are vectorized using numpy SIMD operations.
    """
    
    entity_manager: EntityManager
    ir_table: ProxyIRTable = None
    
    # Processing parameters
    fft_size: int = 16384  # FFT size for convolution
    hop_size: int = 4096   # Hop size for overlap-add
    
    def __post_init__(self):
        if self.ir_table is None:
            self.ir_table = ProxyIRTable(self.entity_manager)
        
        # Pre-allocate FFT buffers
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.complex64)
        self._overlap_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        # Pre-compute FFT of IRs for each size step and contact type
        self._precompute_ir_ffts()
    
    def _precompute_ir_ffts(self):
        """Pre-compute FFT of all IRs for faster convolution."""
        n_sizes = self.ir_table.n_size_steps
        n_types = 4
        n_bands = self.ir_table.n_frequency_bands
        
        # Pre-compute FFTs: (n_sizes, n_types, n_bands, fft_size/2+1)
        self._ir_ffts = np.zeros(
            (n_sizes, n_types, n_bands, self.fft_size // 2 + 1),
            dtype=np.complex64
        )
        
        for size_idx in range(n_sizes):
            for type_idx in range(n_types):
                for band_idx in range(n_bands):
                    ir = self.ir_table.ir_table[size_idx, type_idx, band_idx]
                    # Pad to FFT size
                    padded = np.zeros(self.fft_size)
                    ir_len = min(len(ir), self.fft_size)
                    padded[:ir_len] = ir[:ir_len]
                    self._ir_ffts[size_idx, type_idx, band_idx] = np.fft.rfft(padded)
    
    def process_impact(self, size_scale: float, force: float, 
                       duration: float = 0.1) -> np.ndarray:
        """
        Process impact event.
        
        Parameters:
        -----------
        size_scale : float
            Normalized size (0-1)
        force : float
            Impact force magnitude
        duration : float
            Impact duration in seconds
        
        Returns:
        --------
        np.ndarray : Synthesized audio
        """
        # Get IR for impact
        ir = self.ir_table.get_ir(size_scale, 0)
        
        # Generate impact excitation (Hertzian-like profile)
        n_samples = int(duration * self.ir_table.sample_rate)
        t = np.arange(n_samples) / self.ir_table.sample_rate
        
        # Asymmetric impact envelope
        rise_time = duration * 0.3
        decay_time = duration * 0.7
        
        envelope = np.zeros(n_samples)
        rise_samples = int(rise_time * self.ir_table.sample_rate)
        decay_samples = int(decay_time * self.ir_table.sample_rate)
        
        # Rise phase
        if rise_samples > 0:
            rise_env = np.sin(np.linspace(0, np.pi/2, rise_samples))**2
            envelope[:rise_samples] = rise_env
        
        # Decay phase
        if decay_samples > 0:
            decay_env = np.exp(-np.linspace(0, 5, decay_samples))
            envelope[rise_samples:rise_samples + decay_samples] = decay_env
        
        # Scale by force
        excitation = force * envelope
        
        # Convolve with IR (vectorized across frequency bands)
        output = self._convolve_with_ir(excitation, ir)
        
        return output
    
    def process_continuous(self, size_scale: float, contact_type: int,
                           force: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Process continuous contact (sliding, scraping, rolling).
        
        Parameters:
        -----------
        size_scale : float
            Normalized size (0-1)
        contact_type : int
            Contact type (1=sliding, 2=scraping, 3=rolling)
        force : np.ndarray
            Force signal over time
        velocity : np.ndarray
            Velocity signal over time
        
        Returns:
        --------
        np.ndarray : Synthesized audio
        """
        n_samples = len(force)
        
        # Get IR for contact type
        ir = self.ir_table.get_ir(size_scale, contact_type)
        
        # Generate excitation based on contact type
        if contact_type == 1:  # Sliding
            excitation = self._generate_sliding_excitation(force, velocity)
        elif contact_type == 2:  # Scraping
            excitation = self._generate_scraping_excitation(force, velocity)
        else:  # Rolling
            excitation = self._generate_rolling_excitation(force, velocity, size_scale)
        
        # Convolve with IR using overlap-add
        output = self._overlap_add_convolve(excitation, ir)
        
        return output
    
    def _generate_sliding_excitation(self, force: np.ndarray, 
                                     velocity: np.ndarray) -> np.ndarray:
        """Generate sliding excitation signal."""
        n_samples = len(force)
        
        # White noise modulated by force and velocity
        noise = np.random.randn(n_samples)
        
        # Amplitude modulation
        amplitude = np.sqrt(np.abs(force) * np.abs(velocity))
        amplitude = amplitude / (np.max(amplitude) + 1e-10)
        
        # Frequency modulation based on velocity
        base_freq = 500 + 2000 * np.abs(velocity) / (np.max(np.abs(velocity)) + 1e-10)
        
        # Generate modulated noise
        t = np.arange(n_samples) / self.ir_table.sample_rate
        phase = 2 * np.pi * np.cumsum(base_freq) / self.ir_table.sample_rate
        
        excitation = noise * amplitude * np.sin(phase)
        
        return excitation
    
    def _generate_scraping_excitation(self, force: np.ndarray,
                                      velocity: np.ndarray) -> np.ndarray:
        """Generate scraping excitation signal."""
        n_samples = len(force)
        
               # Bandpass noise with higher frequency content
        noise = np.random.randn(n_samples)
        
        # Apply simple highpass filter (first difference)
        noise = np.diff(noise, prepend=0)
        
        # Amplitude modulation
        amplitude = np.abs(force) * np.abs(velocity)
        amplitude = amplitude / (np.max(amplitude) + 1e-10)
        
        # Add transient spikes
        n_spikes = int(n_samples / (self.ir_table.sample_rate * 0.05))  # ~20 spikes/sec
        spike_positions = np.random.choice(n_samples, min(n_spikes, n_samples), replace=False)
        
        excitation = noise * amplitude
        for pos in spike_positions:
            spike_len = min(50, n_samples - pos)
            if spike_len > 0:
                spike = np.exp(-np.arange(spike_len) / 10) * np.random.randn()
                excitation[pos:pos + spike_len] += spike * amplitude[pos]
        
        return excitation
    
    def _generate_rolling_excitation(self, force: np.ndarray,
                                     velocity: np.ndarray,
                                     size_scale: float) -> np.ndarray:
        """Generate rolling excitation signal."""
        n_samples = len(force)
        
        # Pulse rate based on size and velocity
        base_rate = 5.0 + 20.0 * (1 - size_scale)  # Smaller = faster
        pulse_rate = base_rate * np.abs(velocity) / (np.max(np.abs(velocity)) + 1e-10)
        
        # Generate pulse train
        t = np.arange(n_samples) / self.ir_table.sample_rate
        pulse_phase = np.cumsum(pulse_rate) / self.ir_table.sample_rate
        
        # Gaussian pulses
        pulse_width = 0.005  # 5ms
        excitation = np.exp(-((np.mod(pulse_phase, 1.0) - 0.5) / pulse_width)**2)
        
        # Modulate by force
        amplitude = np.abs(force) / (np.max(np.abs(force)) + 1e-10)
        excitation *= amplitude
        
        # Add some noise
        excitation += 0.1 * np.random.randn(n_samples) * amplitude
        
        return excitation
    
    def _convolve_with_ir(self, signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Convolve signal with IR using FFT.
        
        ir shape: (n_frequency_bands, ir_length)
        """
        n_samples = len(signal)
        n_bands = ir.shape[0]
        
        # Pad signal to FFT size
        padded_signal = np.zeros(self.fft_size)
        padded_signal[:min(n_samples, self.fft_size)] = signal[:min(n_samples, self.fft_size)]
        
        # FFT of signal
        signal_fft = np.fft.rfft(padded_signal)
        
        # Convolve with each frequency band
        output = np.zeros(n_samples + self.fft_size, dtype=np.float32)
        
        for band_idx in range(n_bands):
            # Get IR FFT for this band
            # Interpolate IR FFT based on size (already done in get_ir)
            ir_fft = np.fft.rfft(ir[band_idx], n=self.fft_size)
            
            # Multiply in frequency domain
            result_fft = signal_fft * ir_fft
            
            # Inverse FFT
            result = np.fft.irfft(result_fft, n=self.fft_size)
            
            # Add to output
            output[:self.fft_size] += result
        
        # Trim to signal length
        output = output[:n_samples]
        
        return output
    
    def _overlap_add_convolve(self, signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Overlap-add convolution for long signals.
        
        This is more memory efficient for continuous contact.
        """
        n_samples = len(signal)
        n_bands = ir.shape[0]
        
        # Initialize output
        output = np.zeros(n_samples + self.fft_size, dtype=np.float32)
        
        # Process in blocks
        n n_blocks = int(np.ceil(n_samples / self.hop_size))
        
        for block_idx in range(n_blocks):
            start = block_idx * self.hop_size
            end = min(start + self.hop_size, n_samples)
            block_len = end - start
            
            if block_len <= 0:
                continue
            
            # Extract block
            block = signal[start:end]
            
            # Pad block
            padded_block = np.zeros(self.fft_size)
            padded_block[:block_len] = block
            
            # FFT of block
            block_fft = np.fft.rfft(padded_block)
            
            # Convolve with each frequency band
            for band_idx in range(n_bands):
                # Get IR for this band
                ir_band = ir[band_idx]
                
                # Pad IR
                padded_ir = np.zeros(self.fft_size)
                ir_len = min(len(ir_band), self.fft_size)
                padded_ir[:ir_len] = ir_band[:ir_len]
                
                # FFT of IR
                ir_fft = np.fft.rfft(padded_ir)
                
                # Multiply in frequency domain
                result_fft = block_fft * ir_fft
                
                # Inverse FFT
                result = np.fft.irfft(result_fft, n=self.fft_size)
                
                # Overlap-add
                output[start:start + self.fft_size] += result
        
        # Trim to signal length
        output = output[:n_samples]
        
        return output
    
    def process_mixed(self, size_scale: float, forces: Dict[int, np.ndarray],
                      velocities: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Process mixed contact (multiple contact types simultaneously).
        
        Parameters:
        -----------
        size_scale : float
            Normalized size (0-1)
        forces : Dict[int, np.ndarray]
            Forces for each contact type
        velocities : Dict[int, np.ndarray]
            Velocities for each contact type
        
        Returns:
        --------
        np.ndarray : Mixed synthesized audio
        """
        n_samples = max(len(f) for f in forces.values()) if forces else 0
        
        if n_samples == 0:
            return np.zeros(0)
        
        # Initialize output
        output = np.zeros(n_samples)
        
        # Process each contact type
        for contact_type, force in forces.items():
            if contact_type in velocities:
                velocity = velocities[contact_type]
                
                # Process this contact type
                if contact_type == 0:  # Impact
                    # Impact is instantaneous, use peak force
                    peak__force = np.max(np.abs(force)) if len(force) > 0 else 0
                    if peak_force > 0:
                        impact_output = self.process_impact(size_scale, peak_force)
                        # Align impact at the position of peak force
                        peak_idx = np.argmax(np.abs(force))
                        start = max(0, peak_idx - len(impact_output) // 2)
                        end = min(n_samples, start + len(impact_output))
                        if end > start:
                            output[start:end] += impact_output[:end - start]
                else:
                    # Continuous contact
                    contact_output = self.process_continuous(
                        size_scale, contact_type, force, velocity
                    )
                    output += contact_output
        
        return output

