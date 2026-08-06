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
import numpy as np np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from .proxy_ir_table import ProxyIRTable
from .proxy_eq import ProxyEqualizer

@dataclass
class ProxySynth:
    """
    Complete lightweight physically-based synthesizer for proxy meshes.
    
    Features:
    - Precomputed IR table with size interpolation
    - SIMD-optimized FFT convolution
    - Dynamic frequency equalization
    - Support for impact, sliding, scraping, rolling
    - Mixed contact type processing
    - CollisionData-based excitation for more detailed synthesis
    """
    
    entity_manager: EntityManager
    
    # Components
    ir_table: ProxyIRTable = None
    equalizer: ProxyEqualizer = None
    
    # Processing parameters
    sample_rate: int = 48000
    fft_size: int = 16384
    hop_size: int = 4096
    
    # Output
    output_dir: str = None
    
    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.sample_rate = int(config.system.sample_rate)
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        self.sfps = ( fps / fps_base ) * subframes # subframes per seconds
        
        # Initialize components
        if self.ir_table is None:
            self.ir_table = ProxyIRTable(self.entity_manager)
        
        if self.equalizer is None:
            self.equalizer = ProxyEqualizer(sample_rate=self.sample_rate)
        
        # Set output directory
        if self.output_dir is None:
            self.output_dir = f"{config.system.cache_path}/proxy_audio"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pre-compute FFT of IRs for fast convolution
        self._precompute_ir_ffts()
        
        # Load collision data for this object
        self._load_collision_data()
    
    def _precompute_ir_ffts(self):
        """Pre-compute FFT of all IRs for faster convolution."""
        n_sizes = self.ir_table.n_size_steps
        n_types = 4
        n_bands = self.ir_table.n_frequency_bands
        
        # Pre-compute FFTs: (n_sizes, n_types, n_bands, fft_size/2+1)
        self._ir_ffts = np.zeros((n_sizes, n_types, n_bands, self.fft_size // 2 + 1), dtype=np.complex64)
        
        for size_idx in range(n_sizes):
            for type_idx in range(n_types):
                for band_idx in range(n_bands):
                    ir = self.ir_table.ir_table_table[size_idx, type_idx, band_idx]
                    # Pad to FFT size
                    padded = np.zeros(self.fft_size)
                    ir_len = min(len(ir), self.fft_size)
                    padded[:ir_len] = ir[:ir_len]
                    self._ir_ffts[size_idx, type_idx, band_idx] = np.fft.rfft(padded)
    
    def _load_collision_data(self):
        """Load collision data from entity manager."""
        self.collisions = self.entity_manager.get('collisions')
        self.forces = self.entity_manager.get('forces')
        self.trajectories = self.entity_manager.get('trajectories')
    
    def compute(self, obj_idx: int) -> None:
        """
        Compute proxy synth for an object using CollisionData.
        
        Parameters:
        -----------
        obj_idx : int
            Object index
        """
        config = self.entity_manager.get('config')
        
        # Find object config
        config_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                config_obj = obj
                break
        
        if config_obj is None or config_obj.proxy_type is False:
            return
        
        # Get size scale for this object
        size_scale = self._compute_size_scale(config_obj)
        
        # Find all collisions involving this object
        obj_collisions = []
        for c_idx in self.collisions.keys():
            collision = self.collisions[c_idx]
            if collision.obj1_idx == obj_idx or collision.obj2_idx == obj_idx:
                obj_collisions.append(collision)
        
        if not obj_collisions:
            debug_print(f"No collisions found for {config_obj.name}")
            return
        
        # Get trajectory for this object
        trajectory = None
        for t_idx in self.trajectories.keys():
            if self.trajectories[t_idx].obj_idx == obj_idx:
                trajectory = self.trajectories[t_idx]
                break
        
        if trajectory is None:
            return
        
        # Get total duration from trajectory
        frames = trajectory.get_x()
        if len(frames) > 0:
            total_samples = int(frames[-1])
        else:
            total_samples = 0
        
        # Initialize output tracks
        impact_track = np.zeros(total_samples, dtype=np.float32)
        sliding_track = np.zeros(total_samples, dtype=np.float32)
        scraping_track = np.zeros(total_samples, dtype=np.float32)
        rolling_track = np.zeros(total_samples, dtype=np.float32)
        
        # Process each collision event
        for collision in obj_collisions:
            # Get the other object index
            other_obj_idx = collision.obj2_idx if collision.obj1_idx == obj_idx else collision.obj1_idx
            
            # Get force data for this collision pair
            force_data = self._get_force_data(obj_idx, other_obj_idx)
            
            if force_data is None:
                continue
            
            # Process based on collision type
            if collision.type.value == 'impact':
                # Process impact event
                impact_audio = self._process_impact_from_collision(collision, force_data, trajectory, size_scale)
                
                # Add to impact track
                start_sample = int(collision.frame)
                end_sample = min(start_sample + len(impact_audio), total_samples)
                if end_sample > start_sample:
                    impact_track[start_sample:end_sample] += impact_audio[:end_sample - start_sample]
            
            elif collision.type.value == 'contact':
                # Process continuous contact
                contact_audio = self._process_contact_from_collision(collision, force_data, trajectory, size_scale)
                
                # Add to appropriate track based on contact type
                start_sample = int(collision.frame)
                end_sample = min(start_sample + len(contact_audio), total_samples)
                if end_sample > start_sample:
                    # Determine contact type from force data
                    contact_type = self._get_contact_type(force_data, collision.frame)
                    
                    if contact_type == 2:  # Scraping
                        scraping_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample]
                    elif contact_type == 3:  # Sliding
                        sliding_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample]
                    elif contact_type == 4:  # Rolling
                        rolling_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample]
                    else:  # Mixed or other
                        # Split between tracks based on force characteristics
                        split = self._splitsplit_mixed_contact(force_data, collision.frame)
                        scraping_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample] * split['scraping']
                        sliding_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample] * split['sliding']
                        rolling_track[start_sample:end_sample] += contact_audio[:end_sample - start_sample] * split['rolling']
        
        # Apply equalization
        impact_track = self.equalizer.apply_equalization(impact_track, 0)
        sliding_track = self.equalizer.apply_equalization(sliding_track, 1)
        scraping_track = self.equalizer.apply_equalization(scraping_track, 2)
        rolling_track = self.equalizer.apply_equalization(rolling_track, 3)
        
        # Mix all tracks
        mixed = impact_track + sliding_track + scraping_track + rolling_track
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.9
        
        # Save output
        self._save_audio(config_obj, mixed)
    
    def _get_force_data(self, obj_idx: int, other_obj_idx: int) -> Any:
        """Get force data for a collision pair."""
        for f_idx in self.forces.keys():
            force = self.forces[f_idx]
            if (force.obj_idx == obj_idx and force.other_obj_idx == other_obj_idx) or \
               (force.obj_idx == other_obj_idx and force.other_obj_idx == obj_idx):
                return force
        return None
    
    def _get_contact_type(self, force_data: Any, frame: float) -> int:
        """Get contact type from force data at a specific frame."""
        try:
            return int(force_data.get_contact_type(frame))
        except:
            return 0
    
    def _process_impact_from_collision(self, collision: Any, force_data: Any, 
                                        trajectory: Any, size_scale: float) -> np.ndarray:
        """
        Process impact event using CollisionData for detailed excitation.
        
        Uses collision impulse_range and force data for more accurate impact synthesis.
        """
        # Get impact parameters from collision data
        impact_duration = collision.impulse_range / self.sample_rate  # Convert samples to seconds
        if impact_duration <= 0:
            impact_duration = 0.01  # Default 10ms
        
        # Get force at impact frame
        frame = collision.frame
        try:
            force_mag = force_data.get_normal_force_magnitude(frame)
        except:
            force_mag = 1.0
        
        # Get velocity at impact
        try:
            velocity = trajectory.get_velocity(frame)
            velocity_mag = np.linalg.norm(velocity)
        except:
            velocity_mag = 1.0
        
        # Get IR for impact
        ir = self.ir_table.get_ir(size_scale, 0)
        
        # Generate impact excitation using collision data
        n_samples = int(impact_duration * self.sample_rate)
        if n_samples <= 0:
            n_samples = int(0.01 * self.sample_rate)  # Minimum 10ms
        
        t = np.arange(n_samples) / self.sample_rate
        
        # Use collision impulse_range for more accurate envelope
        # The impulse_range represents the duration of the impact impulse
        rise_time = min(impact_duration * 0.3, 0.005)  # Max 5ms rise
        decay_time = impact_duration - rise_time
        
        envelope = np.zeros(n_samples)
        rise_samples = int(rise_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        
        # Rise phase (Hertzian-like)
        if rise_samples > 0:
            rise_env = np.sin(np.linspace(0, np.pi/2, rise_samples))**2
            envelope[:rise_samples] = rise_env
        
        # Decay phase (exponential based on collision threshold)
        if decay_samples > 0:
            # Use collision threshold for decay rate
            decay_rate = 3.0 + collision.threshold * 2.0 if collision.threshold else 3.0
            decay_env = np.exp(-np.linspace(0, decay_rate, decay_samples))
            envelope[rise_samples:rise_samples + decay_samples] = decay_env
        
        # Scale by force and velocity
        excitation = force_mag * velocity_mag * envelope
        
        # Convolve with IR using FFT
        output = self._fft_convolve(excitation, ir)
        
        return output
    
    def def _process_contact_from_collision(self, collision: Any, force_data: Any, trajectory: Any, size_scale: float) -> np.ndarray:
        """
        Process continuous contact using CollisionData for detailed excitation.
        
        Uses collision frame_range and distance data for more accurate synthesis.
        """
        # Get contact duration from collision data
        contact_duration = collision.frame_range / self.sample_rate
        if contact_duration <= 0:
            contact_duration = 0.1  # Default 100ms
        
        n_samples = int(contact_duration * self.sample_rate)
        if n_samples <= 0:
            n_samples = 1
        
        # Get contact type
        contact_type = self._get_contact_type(force_data, collision.frame)
        
        # Get IR for contact type
        ir = self.ir_table.get_ir(size_scale, contact_type)
        
        # Generate excitation based on contact type and collision data
        if contact_type == 2:  # Scraping
            excitation = self._generate_scraping_excitation_from_collision(n_samples, force_data, trajectory, collision)
        elif contact_type == 3:  # Sliding
            excitation = self._generate_sliding_excitation_from_collision(n_samples, force_data, trajectory, collision)
        elif contact_type == 4:  # Rolling
            excitation = self._generate_rolling_excitation_from_collision(n_samples, force_data, trajectory, collision, size_scale)
        else:  # Mixed or other
            excitation = self._generate_mixed_excitation_from_collision(n_samples, force_data, trajectory, collision, size_scale)
        
        # Convolve with IR
        output = self._fft_convolve(excitation, ir)
        
        return output
    
    def _generate_sliding_excitation_from_collision(self, n_samples: int, force_data: Any, trajectory: Any, collision: Any) -> np.ndarray:
        """Generate sliding excitation using collision data."""
        # Get force and velocity profiles over the contact duration
        start_frame = collision.frame
        end_frame = collision.frame + collision.frame_range
        
        # Generate time array
        t = np.arange(n_samples) / self.sample_rate
        
        # Get force and velocity at each sample
        force_profile = np.zeros(n_samples)
        velocity_profile = np.zeros(n_samples)
        
        for i in range(n_samples):
            frame = = start_frame + (i / n_samples) * (end_frame - start_frame)
            try:
                force_profile[i] = force_data.get_normal_force_magnitude(frame)
                velocity = trajectory.get_velocity(frame)
                velocity_profile[i] = np.linalg.norm(velocity)
            except:
                force_profile[i] = 1.0
                velocity_profile[i] = 1.0
        
        # White noise
        noise = np.random.randn(n_samples)
        
        # Amplitude modulation based on force and velocity
        amplitude = np.sqrt(np.abs(force_profile) * np.abs(velocity_profile))
        amplitude = amplitude / (np.max(amplitude) + 1e-10)
        
        # Frequency modulation based on velocity
        base_freq = 500 + 2000 * np.abs(velocity_profile) / (np.max(np.abs(velocity_profile)) + 1e-10)
        
        # Generate modulated noise
        phase = 2 * np.pi * np.cumsum(base_freq) / self.sample_rate
        
        excitation = noise * amplitude * np.sin(phase)
        
        # Apply collision-based envelope
        envelope = self._get_collision_envelope(n_samples, collision)
        excitation *= envelope
        
        return excitation
    
    def _generate_scraping_excitation_from_collisionision(self, n_samples: int, force_data: Any, trajectory: Any, collision: Any) -> np.ndarray:
        """Generate scraping excitation using collision data."""
        # Get force and velocity profiles
        start_frame = collision.frame
        end_frame = collision.frame + collision.frame_range
        
        # Generate time array
        t = np.arange(n_samples) / self.sample_rate
        
        # Get force and velocity at each sample
        force_profile = np.zeros(n_samples)
        velocity_profile = np.zeros(n_samples)
        
        for i in range(n_samples):
            frame = start_frame + (i / n_samples) * (end_frame - start_frame)
            try:
                force_profile[i] = force_data.get_normal_force_magnitude(frame)
                velocity = trajectory.get_velocity(frame)
                velocity_profile[i] = np.linalg.norm(velocity)
            except:
                force_profile[i] = 1.0
                velocity_profile[i] = 1.0
        
        # Bandpass noise with higher frequency content
        noise = np.random.randn(n_samples)
        
        # Apply simple highpass filter (first difference)
        noise = np.diff(noise, prepend=0)
        
        # Amplitude modulation
        amplitude = np.abs(force_profile) * np.abs(velocity_profile)
        amplitude = amplitude / (np.max(amplitude) + 1e-10)
        
        # Add transient spikes based on collision distance data
        if collision.distances is not None and len(collision.distances) > 0:
            # Use distance variations to create spikes
            distance_profile = np.interp(
                np.linspace(0, 1, n_samples),
                np.linspace(0, 1, len(collision.distances)),
                collision.distances
            )
            
            # Detect spikes where distance changes rapidly
            distance_diff = np.abs(np.diff(distance_profile, prepend=distance_profile[0]))
            spike_mask = distance_diff_diff > np.mean(distance_diff) * 2
            
            # Add spikes
            for i in range(n_samples):
                if spike_mask[i] and i > 0 and i < n_samples - 1:
                    spike_len = min(50, n_samples - i)
                    spike = np.exp(-np.arange(spike_len) / 10) * np.random.randn()
                    excitation[i:i + spike_len] += spike * amplitude[i]
        
        excitation = noise * amplitude
        
        # Apply collision-based envelope
        envelope = self._get_collision_envelope(n_samples, collision)
        excitation *= envelope
        
        return excitation
    
    def _generate_rolling_excitation_from_collision(self, n_samples: int, force_data: Any, trajectory: Any, collision: Any, size_scale: float) -> np.ndarray:
        """Generate rolling excitation using collision data."""
        # Get angular velocity profile
        start_frame = collision.frame
        end_frame = collision.frame + collision.frame_range
        
        # Generate time array
        t = np.arange(n_samples) / self.sample_rate
        
        # Get force and angular velocity at each sample
        force_profile = np.zeros(n_samples)
        angular_velocity_profile = np.zeros(n_samples)
        
        for i in range(n_samples):
            frame = start_frame + (i / n_samples) * (end_frame - start_frame)
            try:
                force_profile[i] = force_data.get_normal_force_magnitude(frame)
                               angular_velocity = trajectory.get_angular_velocity(frame)
                angular_velocity_profile[i] = np.linalg.norm(angular_velocity)
            except:
                force_profile[i] = 1.0
                angular_velocity_profile[i] = 1.0
        
        # Pulse rate based on size and angular velocity
        base_rate = 5.0 + 20.0 * (1 - size_scale)
        pulse_rate = base_rate * np.abs(angular_velocity_profile) / (np.max(np.abs(angular_velocity_profile)) + 1e-10)
        
        # Generate pulse train
        pulse_phase = np.cumsum(pulse_rate) / self.sample_rate
        
        # Gaussian pulses
        pulse_width = 0.005
        excitation = np.exp(-((np.mod(pulse_phase, 1.0) - 0.5) / pulse_width)**2)
        
        # Modulate by force
        amplitude = np.abs(force_profile) / (np.max(np.abs(force_profile)) + 1e-10)
        excitation *= amplitude
        
        # Add some noise
        excitation += 0.1 * np.random.randn(n_samples) * amplitude
        
        # Apply collision-based envelope
        envelope = self._get_collision_envelope(n_samples, collision)
        excitation *= envelope
        
        return excitation
    
    def _generate_mixed_excitation_from_collision(self, n_samples: int, force_data: Any, trajectory: Any, collision: Any, size_scale: float) -> np.ndarray:
        """Generate mixed excitation using collision data."""
        # Get contact type at different points in the collision
        start_frame = collision.frame
        end_frame = collision.frame + collision.frame_range
        
        # Sample contact types throughout the collision
        contact_types = []
        for i in range(min(10, n_samples)):
            frame = start_frame + (i / min(10, n_samples)) * (end_frame - start_frame)
            contact_types.append(self._get_contact_type(force_data, frame))
        
        # Determine dominant contact types
        unique_types = set(contact_types)
        
        # Generate excitation based on dominant types
        if 4 in unique_types:  # Rolling component
            rolling_excitation = self._generate_rolling_excitation_from_collision(n_samples, force_data, trajectory, collision, size_scale)
        else:
            rolling_excitation = np.zeros(n_samples)
        
        if 3 in unique_types:  # Sliding component
            sliding_excitation = self._generate_sliding_excitation_from_collision(n_samples, force_data, trajectory, collision)
        else:
            sliding_excitation = np.zeros(n_samples)
        
        if 2 in unique_types:  # Scraping component
            scraping_excitation = self._generate_scraping_excitation_from_collision(n_samples, force_data, trajectory, collision)
        else:
            scraping_excitation = np.zeros(n_samples)
        
        # Combine excitations
        excitation = rolling_excitation + sliding_excitation + scraping_excitation
        
        return excitation
    
    def _get_collision_envelope(self, n_samples: int, collision: Any) -> np.ndarray:
        """Generate envelope based on collision data."""
        # Use collision distance data for envelope
        if collision.distances is not None and len(collision.distances) > 0:
            # Interpolate distances to match n_samples
            distance_profile = np.interp(
                np.linspace(0, 1, n_samples),
                np.linspace(0, 1, len(collision.distances)),
                collision.distances
            )
            
            # Convert distance to envelope (closer = louder)
            envelope = 1.0 / (1.0 + distance_profile * 10.0)
            
            # Normalize
            envelope = envelope / np.max(envelope)
        else:
            # Use Tukey window as fallback
            from scipy.signal import windows
            envelope = windows.tukey(n_samples, alpha=0.3)
        
        return envelope
    
    def _split_mixed_contact(self, force_data: Any, frame: float) -> Dict[str, float]:
        """Split mixed contact into component ratios."""
        # Get force components
        try:
            normal_force = force_data.get_normal_force_magnitude(frame)
            tangential_force = force_data.get_tangential_force_magnitude(frame)
        except:
            normal_force = 1.0
            tangential_force = 1.0
        
        # Get velocities
        try:
            relative_velocity = np.linalg.norm(force_data.get_relative_velocity(frame))
            tangential_velocity = np.linalg.norm(force_data.get_tangential_velocity(frame))
        except:
            relative_velocity = 1.0
            tangential_velocity = 1.0
        
        # Calculate ratios based on force and velocity characteristics
        total = normal_force + tangential_forceforce + 1e-10
        
        # Rolling ratio (based on tangential velocity relative to normal force)
        rolling_ratio = min(0.5, tangential_velocity / (normal_force + 1e-10) * 0.1)
        
        # Scraping ratio (based on tangential force)
        scraping_ratio = min(0.5, tangential_force / total * 0.3)
        
        # Sliding ratio (remaining)
        sliding_ratio = max(0.0, 1.0 - rolling_ratio - scraping_ratio)
        
        return {
            'rolling': rolling_ratio,
            'scraping': scraping_ratio,
            'sliding': sliding_ratio
        }
    
    def _compute_size_scale(self, config_obj: Any) -> float:
        """Compute normalized size scale (0-1) for an object."""
        # Load proxy mesh to compute size
        from pbrAudioCommon import _load_mesh
        try:
            vertices, _, _ = _load_mesh(config_obj, 0, use_proxy_path=True)
            if len(vertices) > 0:
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                size = np.linalg.norm(max_coords - min_coords)
                
                # Normalize to 0-1 range
                size_range = self.ir_table.max_size - self.ir_table.min_size
                if size_range > 0:
                    size_scale = (size - self.ir_table.min_size) / size_range
                else:
                    size_scale = 0.5
                
                return np.clip(size_scale, 0, 1)
        except:
            pass
        
        return 0.5  # Default
    
    def _fft_convolve(self, signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        FFT-based convolution with SIMD optimization.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        ir : np.ndarray
            Impulse response (n_frequency_bands, ir_length)
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        n_samples = len(signal)
        n_bands = ir.shape[0]
        
        # Pad signal to FFT size
        padded_signal = np.zeros(self.fft_size)
        padded_signal[:min(n_samples, self.fft_size)] = signal[:min(n_samples, self.fft_size)]
        
        # FFT of signal
        signal_fft = np.fft.rfft(padded_signal)
        
        # Initialize output
        output = np.zeros(self.fft_size, dtype=np.float32)
        
        # Convolve with each frequency band
        for band_idx in range(n_bands):
            # Get IR for this band
            ir_band = ir[band_idx]
            
            # Pad IR
            padded_ir = np.zeros(self.fft_size)
            ir_len = min(len(ir_band), self.fft_size)
            padded_ir[:ir_len] = ir_band[:ir_len]
            
            # FFT of IR
            ir_fffft = np.fft.rfft(padded_ir)
            
            # Multiply in frequency domain
            result_fft = signal_fft * ir_fft
            
            # Inverse FFT
            result = np.fft.irfft(result_fft, n=self.fft_size)
            
            # Add to output
            output += result
        
        # Trim to signal length
        output = output[:n_samples]
        
        return output
    
    def _save_audio(self, config_obj: Any, audio: np.ndarray) -> None:
        """Save synthesized audio to file."""
        output_file = f"{self.output_dir}/{config_obj.name}_proxy.wav"
        sf.write(output_file, audio, self.sample_rate, subtype='FLOAT')
        debug_print(f"Saved proxy audio to {output_file}")
