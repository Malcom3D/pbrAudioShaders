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

import os
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import soundfile as sf

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _parse_lib
from rigidBody import ModalBank, ConnectedBuffer

from .fracture_data import FractureEvent, FractureType

@dataclass
class FractureSynth:
    """
    Synthesize fracture sounds based on the FractureSound paper.
    
    Implements:
    - Crack propagation sound (filtered noise)
    - Fragment separation sounds (modal synthesis)
    - Energy release characteristics
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = config.system.sample_rate
        self.fracture_modal_path = f"{config.system.cache_path}/fracture_modal"
        self.fracture_audio_dir = f"{config.system.cache_path}/fracture_audio"
        os.makedirs(self.fracture_audio_dir, exist_ok=True)
    
    def compute(self, event: FractureEvent) -> None:
        """
        Compute fracture sound for a fracture event.
        
        Parameters:
        -----------
        event : FractureEvent
            The fracture event to synthesize
        """
        config = self.entity_manager.get('config')
        
        # Get fragment objects
        fragment1_obj = None
        fragment2_obj = None
        
        for obj in config.objects:
            if obj.idx == event.fragment1_idx:
                fragment1_obj = obj
            elif obj.idx == event.fragment2_idx:
                fragment2_obj = obj
        
        if not fragment1_obj or not fragment2_obj:
            raise ValueError("Fragments not found")
        
        # Load fracture modal models
        modal1 = self._load_fracture_modal(fragment1_obj.name)
        modal2 = self._load_fracture_modal(fragment2_obj.name)
        
        # Calculate fracture energy
        if event.fracture_energy is None:
            event.fracture_energy = self._compute_fracture_energy(event)
        
        # Synthesize crack propagation sound
        crack_sound = self._synthesize_crack_propagation(event)
        
        # Synthesize fragment separation sounds
        separation_sound1 = self._synthesize_fragment_separation(event, fragment1_obj, modal1)
        separation_sound2 = self._synthesize_fragment_separation(event, fragment2_obj, modal2)
        
        # Combine sounds (crack + both fragments)
        total_duration = int(event.crack_duration * self.sample_rate) + self.sample_rate  # Add 1 second for decay
        combined = np.zeros(total_duration)
        
        # Add crack sound (starts at fracture)
        crack_len = len(crack_sound)
        combined[:crack_len] += crack_sound
        
        # Add fragment separation sounds (slightly delayed - crack propagates then fragments separate)
        delay_samples = int(0.001 * self.sample_rate)  # 1ms delay
        sep1_len = len(separation_sound1)
        sep2_len = len(separation_sound2)
        
        if sep1_len > 0:
            start = min(delay_samples, total_duration - sep1_len)
            combined[start:start+sep1_len] += separation_sound1[:min(sep1_len, total_duration-start)]
        
        if sep2_len > 0:
            start = min(delay_samples, total_duration - sep2_len)
            combined[start:start+sep2_len] += separation_sound2[:min(sep2_len, total_duration-start)]
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.9
        
        # Save to file
        output_file = f"{self.fracture_audio_dir}/fracture_{event.original_obj_idx}_{event.fragment1_idx}_{event.fragment2_idx}.wav"
        sf.write(output_file, combined, self.sample_rate)
        
        print(f"Fracture sound saved to {output_file}")
    
    def _load_fracture_modal(self, obj_name: str) -> Dict[str, Any]:
        """Load fracture modal model."""
        lib_file = f"{self.fracture_modal_path}/{obj_name}_fracture.lib"
        
        if not os.path.exists(lib_file):
            # Fall back to original modal model
            lib_file = f"{self.fracture_modal_path}/{obj_name}.lib"
        
        return _parse_lib(lib_file)
    
    def _compute_fracture_energy(self, event: FractureEvent) -> float:
        """
        Compute energy released during fracture.
        
        Based on forces at fracture time and material properties.
        """
        total_energy = 0.0
        
        # Sum energy from collisions at fracture
        for collision in event.collisions:
            if hasattr(collision, 'distances') and collision.distances is not None:
                # Approximate energy from penetration
                if isinstance(collision.distances, np.ndarray):
                    energy = np.sum(np.abs(collision.distances)) * 1000  # Scale factor
                    total_energy += energy
        
        # Sum energy from forces
        for force_seq in event.forces:
            if hasattr(force_seq, 'normal_force_magnitude'):
                try:
                    # Get force magnitude at fracture frame
                    force_mag = force_seq.normal_force_magnitude(event.frame)
                    total_energy += force_mag * 0.001  # Force * small displacement
                except:
                    pass
        
        # Ensure minimum energy
        total_energy = max(total_energy, 0.1)
        
        return total_energy
    
    def _synthesize_crack_propagation(self, event: FractureEvent) -> np.ndarray:
        """
        Synthesize crack propagation sound.
        
        Implements the crack propagation model from the FractureSound paper:
        - Broadband noise shaped by crack velocity
        - Amplitude modulation based on crack front
        - Frequency content related to material properties
        """
        config = self.entity_manager.get('config')
        
        # Get material properties from original object
        original_obj = None
        for obj in config.objects:
            if obj.idx == event.original_obj_idx:
                original_obj = obj
                break
        
        if not original_obj:
            return np.zeros(0)
        
        # Crack duration in samples
        n_samples = int(event.crack_duration * self.sample_rate)
        
        if n_samples <= 0:
            return np.zeros(0)
        
        # Generate crack sound
        crack_sound = np.zeros(n_samples)
        
        # Time array
        t = np.arange(n_samples) / self.sample_rate
        
        # Crack velocity profile (accelerates then decelerates)
        velocity_profile = 4 * t * (1 - t / event.crack_duration) / event.crack_duration
        
        # Generate broadband noise
        noise = np.random.randn(n_samples)
        
        # Apply time-varying filter based on crack velocity
        # Higher velocity = higher frequency content
        from scipy import signal
        
        # Design filter bank
        nyquist = self.sample_rate / 2
        
        # Crack sound has energy from low to high frequencies
        # but spectral content shifts with velocity
        
        filtered_noise = np.zeros(n_samples)
        
        # Use moving average filter with velocity-dependent window
        for i in range(n_samples):
            # Window size inversely related to velocity
            vel = velocity_profile[i]
            window_size = int(max(3, 20 * (1 - vel)))
            
            start_idx = max(0, i - window_size)
            end_idx = min(n_samples, i + window_size + 1)
            
            if end_idx > start_idx:
                filtered_noise[i] = np.mean(noise[start_idx:end_idx])
        
        # Amplitude envelope based on fracture energy
        energy_factor = np.log10(event.fracture_energy + 1) / 10
        amplitude = energy_factor * velocity_profile
        
        # Add initial impact spike
        spike = np.zeros(n_samples)
        spike_len = min(100, n_samples)
        spike[:spike_len] = np.exp(-np.arange(spike_len) / 20) * energy_factor * 2
        
        # Combine
        crack_sound = filtered_noise * amplitude + spike
        
        # Apply smooth envelope
        from scipy.signal import windows
        envelope = windows.tukey(n_samples, alpha=0.2)
        crack_sound *= envelope
        
        return crack_sound
    
    def _synthesize_fragment_separation(self, event: FractureEvent, fragment_obj: Any, modal_data: Dict[str, Any]) -> np.ndarray:
        """
        Synthesize sound of fragment separation using modal synthesis.
        
        The fragments are excited by the fracture event and radiate sound
        as they separate.
        """
        # Get fragment velocity at separation
        fragment_velocity = np.array([0, 0, 0])
        if len(event.fragment_velocities) >= 2:
            if event.fragment1_idx == fragment_obj.idx:
                fragment_velocity = event.fragment_velocities[0]
            else:
                fragment_velocity = event.fragment_velocities[1]
        
        # Estimate excitation force from fracture energy
        if fragment_obj.acoustic_shader.density:
            # Force based on momentum change
            mass = fragment_obj.acoustic_shader.density * 0.001  # Approximate
            impulse = mass * np.linalg.norm(fragment_velocity)
            excitation_force = impulse / 0.001  # Force over 1ms
        else:
            excitation_force = event.fracture_energy * 10
        
        # Create modal bank for this fragment
        n_vertices = 1  # Simplified - use average response
        banks = []
        
        for i in range(n_vertices):
            bank = ModalBank(
                frequencies=modal_data['frequencies'],
                gains=modal_data['gains'][i] if i < len(modal_data['gains']) else modal_data['gains'][0],
                t60s=modal_data['t60s'],
                sample_rate=self.sample_rate
            )
            banks.append(bank)
        
        # Generate sound
        duration = 1.0  # 1 second of sound
        n_samples = int(duration * self.sample_rate)
        output = np.zeros(n_samples)
        
        # Apply excitation (impulse at beginning)
        for i in range(min(10, n_samples)):
            excitation = excitation_force * np.exp(-i / 5)  # Decaying impulse
            
            for bank in banks:
                output[i] += bank.process(excitation)
        
        # Continue free decay
        for i in range(10, n_samples):
            for bank in banks:
                output[i] += bank.process(0)
        
        # Apply amplitude envelope (fragments radiate as they move apart)
        t = np.arange(n_samples) / self.sample_rate
        distance = np.linalg.norm(fragment_velocity) * t
        # Sound pressure decreases with distance (1/r)
        distance_factor = 1.0 / (1.0 + distance)
        
        output *= distance_factor
        
        return output
