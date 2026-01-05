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
Acceleration Noise implementation based on:
"Acceleration Noise: A Probabilistic Approach to Sound Synthesis for Rigid-Body Animations"
https://graphics.stanford.edu/courses/cs448z-21-spring/stuff/PAN_typoFix.pdf
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal
from scipy.interpolate import interp1d
import trimesh

from ..lib.impacts_data import ImpactEvent, ObjectContact, ContactType

@dataclass
class AccelerationNoiseGenerator:
    """Generates acceleration noise for rigid body animations"""
    
    # Audio parameters
    sample_rate: int = 48000
    bit_depth: int = 32
    
    # Acceleration noise parameters
    min_frequency: float = 20.0  # Hz
    max_frequency: float = 20000.0  # Hz
    num_bands: int = 100  # Number of frequency bands
    q_factor: float = 10.0  # Q factor for bandpass filters
    noise_floor_db: float = -60.0  # Noise floor in dB
    
    # Physical parameters
    air_density: float = 1.225  # kg kg/m³
    speed_of_sound: float = 343.0  # m/s
    
    def generate_acceleration_noise(self, event: ImpactEvent, obj_idx: int, 
                                   trajectories: Dict, meshes: Dict) -> np.ndarray:
        """
        Generate acceleration noise for an object during a contact event.
        
        Args:
            event: The contact event
            obj_idx: Object index
            trajectories: Object trajectory data
            meshes: Object mesh data
            
        Returns:
            Audio signal as numpy array
        """
        # Get object trajectory
        if obj_idx not in trajectories:
            return np.zeros(int(self.sample_rate * event.get_contact_audio_duration()))
        
        traj = trajectories[obj_idx]
        
        # Extract acceleration data for the event duration
        start_idx = np.searchsorted(traj['times'], event.start_time)
        end_idx = np.searchsorted(traj['times'], event.end_time)
        
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, len(traj['times']) - 1)
        
        # Get acceleration samples
        times = traj['times'][start_idx:end_idx]
        accelerations = traj['accelerations'][start_idx:end_idx]
        
        if len(accelerations) < 2:
            return np.zeros(int(self.sample_rate * event.get_contact_audio_duration()))
        
        # Calculate acceleration magnitude
        accel_magnitude = np.linalg.norm(accelerations, axis=1)
        
        # Interpolate to audio sample rate
        duration = event.get_contact_audio_duration()
        audio_samples = int(self.sample_rate * duration)
        audio_times = np.linspace(0, duration, audio_samples)
        
        # Interpolate acceleration magnitude
        if len(times) > 1:
            accel_interp = interp1d(
                times - times[0], 
                accel_magnitude, 
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            accel_signal = accel_interp(audio_times)
        else:
            accel_signal = np.full_like(audio_times, accel_magnitude[0])
        
        # Generate acceleration noise
        noise_signal = self._generate_noise_from_acceleration(
            accel_signal, 
            audio_times,
            event,
            obj_idx,
            meshes.get(obj_idx, {})
        )
        
        return noise_signal
    
    def _generate_noise_from_acceleration(self, accel_signal: np.ndarray, 
                                         times: np.ndarray,
                                         event: ImpactEvent,
                                         obj_idx: int,
                                         mesh_data: Dict) -> np.ndarray:
        """
        Generate noise signal from acceleration using the paper's method.
        
        The method uses:
        1. Acceleration magnitude to modulate noise amplitude
        2. Frequency bands to distribute energy
        3. Object properties to shape spectrum
        """
        num_samples = len(accel_signal)
        
        # Create frequency bands (logarithmic spacing)
        freq_bands = np.logspace(
            np.log10(self.min_frequency),
            np.log10(self.max_frequency),
            self.num_bands
        )
        
        # Initialize output signal
        output_signal = np.zeros(num_samples)
        
        # Get object properties for spectral shaping
        obj_contact = event.get_object_contact(obj_idx)
        if obj_contact:
            surface_roughness = obj_contact.surface_roughness
            material_hardness = obj_contact.material_hardness
        else:
            surface_roughness = 0.001
            material_hardness = 1.0
        
        # Generate noise for each frequency band
        for i, f0 in enumerate(freq_bands):
            # Create bandpass filter
            b, a = signal.iirfilter(
                4,  # 4th order filter
                [f0 * 0.9 / (self.sample_rate/2), f0 * 1.1 / (self.sample_rate/2)],
                btype='bandpass',
                ftype='butter'
            )
            
            # Generate white noise
            white_noise = np.random.randn(num_samples)
            
            # Apply bandpass filter
            band_noise = signal.lfilter(b, a, white_noise)
            
            # Calculate band gain based on:
            # 1. Frequency (higher frequencies typically have less energy)
            # 2. Acceleration magnitude
            # 3. Object properties
            
            # Frequency-dependent gain (1/f roll-off)
            freq_gain = 1.0 / (1.0 + (f0 / 1000.0))  # -6dB/octave approx
            
            # Acceleration-dependent modulation
            # Normalize acceleration signal
            accel_norm = (accel_signal - np.min(accel_signal)) / \
                        (np.max(accel_signal) - np.min(accel_signal) + 1e-10)
            
            # Material-dependent gain
            material_gain = material_hardness * (1.0 + surface_roughness * 100.0)
            
            # Combine gains
            band_gain = freq_gain * material_gain * 0.1  # Overall scaling
            
            # Apply amplitude modulation based on acceleration
            modulated_band = band_noise * (0.5 + 0.5 * accel_norm) * band_gain
            
            # Add to output
            output_signal += modulated_band
        
        # Apply overall envelope based on event
        output_signal = self._apply_event_envelope(output_signal, event, obj_idx)
        
        # Normalize
        max_val = np.max(np.abs(output_signal))
        if max_val > 1e-10:
            output_signal = output_signal / max_val * 0.8
        
        return output_signal
    
    def _apply_event_envelope(self, signal: np.ndarray, event: ImpactEvent, 
                             obj_idx: int) -> np.ndarray:
        """Apply temporal envelope based on event characteristics"""
        num_samples = len(signal)
        t = np.linspace(0, event.get__contact_audio_duration(), num_samples)
        
        # Create envelope based on contact type
        if event.dominant_contact_type == ContactType.IMPACT:
            # Short, sharp envelope for impacts
            attack_time = 0.001
            decay_time = min(0.1, event.duration)
            
            envelope = np.zeros_like(t)
            attack_samples = int(attack_time * self.sample_rate)
            decay_samples = int(decay_time * self.sample_rate)
            
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            if decay_samples > attack_samples:
                decay_start = attack_samples
                decay_length = decay_samples - attack_samples
                envelope[decay_start:decay_samples] = np.exp(-10.0 * np.linspace(0, 1, decay_length))
            
        elif event.dominant_contact_type == ContactType.SCRAPING:
            # Follows the contact duration
            attack_time = min(0.01, event.duration * 0.1)
            release_time = min(0.01, event.duration * 0.1)
            
            envelope = np.ones_like(t)
            
            if attack_time > 0:
                attack_samples = int(attack_time * self.sample_rate)
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            if release_time > 0:
                release_samples = int(release_time * self.sample_rate)
                release_start = num_samples - release_samples
                envelope[release_start:] = np.linspace(1, 0, release_samples)
            
        else:  # ROLLING or MIXED
            # Smooth envelope
            fade_time = min(0.05, event.get_contact_audio_duration() * 0.1)
            fade_samples = int(fade_time * self.sample_rate)
            
            envelope = np.ones_like(t)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return signal * envelope
