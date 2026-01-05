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

"""
Fast Acceleration Noise implementation for small debris based on:
"Faster Acceleration Noise for Small Debris"
https://www.cs.cornell.edu/projects/Sound/proxy/FasterAccelerationNoise_SCA2012.pdf
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal
from scipy.interpolate import interp1d
import trimesh

from ..lib.impacts_data import ImpactEvent, ObjectContact, ContactType

@dataclass
class FastAccelerationNoiseGenerator:
    """Fast acceleration noise generator optimized for small debris"""
    
    # Audio parameters
    sample_rate: int = 48000
    bit_depth: int = 32
    
    # Fast acceleration noise parameters
    num_octave_bands: int = 8  # Number of octave bands
    overlap_factor: float = 0.5  # Overlap between bands
    noise_memory: float = 0.1  # Noise memory in seconds
    acceleration_threshold: float = 0.1  # Minimum acceleration to trigger noise
    
    # Debris-specific parameters
    max_debris_size: float = 0.1  # Maximum size for debris (m)
    min_debris_mass: float = 0.001  # Minimum mass for debris (kg)
    
    # Precomputed filters for speed
    _filters: Dict = field(default_factory=dict)
    _noise_buffers: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Precompute filters for faster processing"""
        self._precompute_filters()
    
    def _precompute_filters(self):
        """Precompute bandpass filters for each octave band"""
        center_freqs = []
        f_low = 50.0  # Start at 50Hz
        
        for i in range(self.num_octave_bands):
            f_high = f_low * 2.0  # Octave bandwidth
            center_freq = np.sqrt(f_low * f_high)
            center_freqs.append(center_freq)
            
            # Create bandpass filter
            nyquist = self.sample_rate / 2
            low_cut = f_low / nyquist
            high_cut = f_high / nyquist
            
            # Use butterworth filter for smooth response
            b, a = signal.butter(
                4,  # 4th order
                [low_cut, high_cut],
                btype='band'
            )
            
            self._filters[center_freq] = (b, a)
            f_low = f_high  # Move to next octave
    
    def generate_fast_acceleration_noise(self, event: ImpactEvent, obj_idx: int,
                                        trajectories: Dict, meshes: Dict,
                                        is_debris: bool = False) -> np.ndarray:
        """
        Generate fast acceleration noise using the paper's optimized method.
        
        Args:
            event: The contact event
            obj_idx: Object index
            trajectories: Object trajectory data
            meshes: Object mesh data
            is_debris: Whether the object is considered debris
            
        Returns:
            Audio signal as numpy array
        """
        # Get object trajectory
        if obj_idx not in trajectories:
            return np.zeros(int(self.sample_rate * event.get_contact_audio_duration()))
        
        traj = trajectories[obj_idx]
        
        # Extract acceleration data
        start_idx = np.searchsorted(traj['times'], event.start_time)
        end_idx = np.searchsorted(traj['times'], event.end_time)
        
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, len(traj['times']) - 1)
        
        times = traj['times'][start_idx:end_idx_idx]
        accelerations = traj['accelerations'][start_idx:end_idx]
        
        if len(accelerations) < 2:
            return np.zeros(int(self.sample_rate * event.get_contact_audio_duration()))
        
        # Calculate acceleration magnitude and jerk (derivativeivative of acceleration)
        accel_magnitude = np.linalg.norm(accelerations, axis=1)
        
        # Calculate jerk (rate of change of acceleration)
        if len(accel_magnitude) > 1:
            dt = np.diff(times)
            jerk = np.diff(accel_magnitude) / dt
            jerk = np.concatenate([[jerk[0]], jerk])  # Pad for same length
        else:
            jerk = np.zeros_like(accel_magnitude)
        
        # Interpolate to audio sample rate
        duration = event.get_contact_audio_duration()
        audio_samples = int(self.sample_rate * duration)
        audio_times = np.linspace(0, duration, audio_samples)
        
        if len(times) > 1:
            # Interpolate acceleration and jerk
            accel_interp = interp1d(
                times - times[0], 
                accel_magnitude, 
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            jerk_interp = interp1d(
                times - times[0],
                jerk,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            accel_signal = accel_interp(audio_times)
            jerk_signal = jerk_interp(audio_times)
        else:
            accel_signal = np.full_like(audio_times, accel_magnitude[0])
            jerk_signal = np.zeros_like(audio_times)
        
        # Generate noise using fast method
        if is_debris:
            noise_signal = self._generate_debris_noise(
                accel_signal, jerk_signal, audio_times,
                event, obj_idx, meshes.get(obj_idx, {})
            )
        else:
            noise_signal = self._generate_general_noise(
                accel_signal, jerk_signal, audio_times,
                event, obj_idx, meshes.get(obj_idx, {})
            )
        
        return noise_signal
    
    def _generate_general_noise(self, accel_signal: np.ndarray, jerk_signal: np.ndarray,
                               times: np.ndarray, event: ImpactEvent,
                               obj_idx: int, mesh_data: Dict) -> np.ndarray:
        """
        Generate noise for general objects using the paper's method.
        
        Uses precomputed filters and noise buffers for speed.
        """
        num_samples = len(accel_signal)
        
        # Initialize output signal
        output_signal = np.zeros(num_samples)
        
        # Get object properties
        obj_contact = event.get_object_contact(obj_idx)
        if obj_contact:
            material_hardness = obj_contact.material_hardness
        else:
            material_hardness = 1.0
        
        # Calculate overall modulation from acceleration and jerk
        # Normalize signals
        accel_norm = (accel_signal - np.min(accel_signal)) / \
                    (np.max(accel_signal) - np.min(accel_signal) + 1e-10)
        jerk_norm = (jerk_signal - np.min(jerk_signal)) / \
                   (np.max(jerk_signal) - np.min(jerk_signal) + 1e-10)
        
        # Combined modulation (acceleration + jerk for high-frequency content)
        modulation = 0.7 * accel_norm + 0.3 * jerk_norm
        
        # Process each octave band
        for center_freq, (b, a) in self._filters.items():
            # Get or create noise buffer for this band
            buffer_key = (obj_idx, center_freq)
            if buffer_key not in self._noise_buffers:
                # Initialize noise buffer with memory
                buffer_size = int(self.noise_memory * self.sample_rate)
                self._noise_buffers[buffer_key] = np.random.randn(buffer_size)
            
            noise_buffer = self._noise_buffers[buffer_key]
            
            # Generate band noise by filtering buffer
            # Use circular buffer for continuous noise
            band_noise = np.zeros(num_samples)
            buffer_pos = 0
            
            for i in range(num_samples):
                # Get noise sample from buffer
                noise_sample = noise_buffer[buffer_pos]
                
                # Update buffer position
                buffer_pos = (buffer_pos + 1) % len(noise_buffer)
                
                # Apply bandpass filter (simplified - using pre-filtered noise would be faster)
                # For speed, we'll apply filter to the entire buffer periodically
                band_noise[i] = noise_sample
            
            # Apply bandpass filter to the entire segment
            band_noise = signal.lfilter(b, a, band_noise)
            
            # Calculate band gain
            # Higher frequencies get less gain (1/f roll-off)
            freq_gain = 1.0 / (1.0 + (center_freq / 1000.0))
            
            # Material-dependent gain
            material_gain = material_hardness
            
            # Apply modulation
            modulated_band = band_noise * modulation * freq_gain * material_gain * 0.05
            
            # Add to output
            output_signal += modulated_band
        
        # Apply envelope
        output_signal = self._apply_fast_envelope(output_signal, event, obj_idx)
        
        # Normalize
        max_val = np.max(np.abs(output_signal))
        if max_val > 1e-10:
            output_signal = output_signal / max_val * 0.8
        
        return output_signal
    
    def _generate_debris_noise(self, accel_signal: np.ndarray, jerk_signal: np.ndarray,
                              times: np.ndarray, event: ImpactEvent,
                              obj_idx: int, mesh_data: Dict) -> np.ndarray:
        """
        Generate noise specifically for small debris.
        
        Uses simplified model with fewer bands and optimized processing.
        """
        num_samples = len(accel_signal)
        
        # For debris, use only 4 octave bands for speed
        debris_bands = 4
        center_freqs = list(self._filters.keys())[:debris_bands]
        
        # Initialize output
        output_signal = np.zeros(num_samples)
        
        # Debris-specific modulation
        # Debris tends to have sharper, more impulsive sounds
        accel_norm = (accel_signal - np.min(accel_signal)) / \
                    (np.max(accel_signal) - np.min(accel_signal) + 1e-10)
        
        # Emphasize transients for debris
        modulation = accel_norm ** 2  # Square for sharper response
        
        # Process selected bands
        for center_freq in center_freqs:
            b, a = self._filters[center_freq]
            
            # Generate noise for this band
            # For speed, use simpler noise generation
            white_noise = np.random.randn(num_samples)
            band_noise = signal.lfilter(b, a, white_noise)
            
            # Debris-specific gains
            # Smaller debris -> higher frequency content
            if 'mesh' in mesh_data:
                mesh = mesh_data['mesh']
                size_estimate = np.max(mesh.bounds[1] - mesh.bounds[0])
                size_factor = min(1.0, size_estimate / self.max_debris_size)
            else:
                size_factor = 0.5
            
            # Frequency adjustment based on size
            size_gain = 1.0 + (1.0 - size_factor) * 2.0  # Smaller = brighter
            
            # Apply modulation and gains
            band_gain = size_gain * 0.1
            modulated_band = band_noise * modulation * band_gain
            
            output_signal += modulated_band
        
        # Apply sharp envelope for debris
        output_signal = self._apply_debris_envelope(output_signal, event)
        
        # Normalize
        max_val = np.max(np.abs(output_signal))
        if max_val > 1e-10:
            output_signal = output_signal / max_val * 0.8
        
        return output_signal
    
    def _apply_fast_envelope(self, signal: np.ndarray, event: ImpactEvent,
                            obj_idx: int) -> np.ndarray:
        """Apply optimized envelope for fast processing"""
        num_samples = len(signal)
        t = np.linspace(0, event.get_contact_audio_duration(), num_samples)
        
        # Simple envelope based on contact type
        if event.dominant_contact_type == ContactType.IMPACT:
            # Exponential decay
            decay_time = min(0.1, event.duration)
            decay_samples = int(decay_time * self.sample_rate)
            
            if decay_samples > 0:
                decay = np.exp(-5.0 * np.linspace(0, 1, decay_samples))
                if len(decay) < num_samples:
                    envelope = np.ones(num_samples)
                    envelope[:decay_samples] = decay
                else:
                    envelope = decay[:num_samples]
            else:
                envelope = np.ones(num_samples)
                
        else:
            # Simple fade in/out
            fade_samples = int(min(0.05, event.get_contact_audio_duration() * 0.1) * self.sample_rate)
            fade_samples = min(fade_samples, num_samples // 2)
            
            envelope = np.ones(num_samples)
            if fade_samples > 0:
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return signal * envelope
    
    def _apply_debris_envelope(self, signal: np.ndarray, event: ImpactEvent) -> np.ndarray:
        """Apply envelope optimized for debris sounds"""
        num_samples = len(signal)
        
        # Debris typically has sharp attacks and quick decays
        attack_time = 0.0005  # Very fast attack
        decay_time = min(0.05, event.duration)  # Quick decay
        
        attack_samples = int(attack_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        
        envelope = np.zeros(num_samples)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if decay_samples > attack_samples:
            decay_start = attack_samples
            decay_end = min(decay_samples, num_samples)
            decay_length = decay_end - decay_start
            
            if decay_length > 0:
                envelope[decay_start:decay_end] = np.exp(-20.0 * np.linspace(0, 1, decay_length))
        
        return signal * envelope
    
    def is_debris(self, mesh_data: Dict, mass: float) -> bool:
        """
        Determine if an object should be treated as debris.
        
        Based on size and mass thresholds.
        """
        if 'mesh' not in mesh_data:
            return False
        
        mesh = mesh_data['mesh']
        
        # Check size
        bounds = mesh.bounds
        size = np.max(bounds[1] - bounds[0])
        
        if size > self.max_debris_size:
            return False
        
        # Check mass
        if mass < self.min_debris_mass:
            return True
        
        # Additional heuristic: high surface area to volume ratio
        if hasattr(mesh, 'area') and hasattr(mesh, 'volume'):
            if mesh.volume > 0:
                ratio = mesh.area / mesh.volume
                if ratio > 100.0:  # High ratio indicates complex, debris-like shape
                    return True
        
        return False
