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


import os, sys
import json
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal
from dataclasses import dataclass 
from typing import List, Dict, Tuple, Optional, Any

from ..core.entity_manager import EntityManager
from ..lib.force_data import ContactType
from ..lib.hertzian_contact import HertzianContact

@dataclass
class ForceSynth:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        os.makedirs(self.collisions_dir, exist_ok=True)
        self.audio_force_dir = f"{config.system.cache_path}/audio_force"
        os.makedirs(self.audio_force_dir, exist_ok=True)
        
        # Overlap-add parameters
        self.overlap_size = int(0.005 * config.system.sample_rate)  # 5ms overlap
        self.window = np.hanning(2 * self.overlap_size)  # Hanning window for smoothing

    def _apply_overlap_add(self, track: np.ndarray, signal_to_add: np.ndarray, 
                          start_idx: int, end_idx: int) -> np.ndarray:
        """
        Apply overlap-add processing to smoothly blend a signal into a track.
        
        Parameters:
        -----------
        track : np.ndarray
            The main track to add the signal to
        signal_to_add : np.ndarray
            The signal segment to add
        start_idx : int
            Start index in the track
        end_idx : int
            End index in the track
            
        Returns:
        --------
        np.ndarray : Updated track with smooth transitions
        """
        signal_len = len(signal_to_add)
        
        # Create fade-in window
        fade_in_len = min(self.overlap_size, signal_len // 2)
        if fade_in_len > 0:
            fade_in = np.linspace(0, 1, fade_in_len)
            signal_to_add[:fade_in_len] *= fade_in
        
        # Create fade-out window
        fade_out_len = min(self.overlap_size, signal_len // 2)
        if fade_out_len > 0:
            fade_out = np.linspace(1, 0, fade_out_len)
            signal_to_add[-fade_out_len:] *= fade_out
        
        # Add signal to track
        actual_end = min(end_idx, len(track))
        actual_len = actual_end - start_idx
        if actual_len > 0:
            track[start_idx:actual_end] += signal_to_add[:actual_len]
        
        return track

    def _create_smooth_window(self, n_samples: int) -> np.ndarray:
        """
        Create a smooth window for envelope shaping with cosine-tapered edges.
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples in the window
            
        Returns:
        --------
        np.ndarray : Smooth window array
        """
        if n_samples <= 2 * self.overlap_size:
            # For very short signals, use full hanning
            return np.hanning(n_samples)
        
        window = np.ones(n_samples)
        
        # Apply fade-in
        fade_in = np.linspace(0, 1, self.overlap_size)
        window[:self.overlap_size] = fade_in
        
        # Apply fade-out
        fade_out = np.linspace(1, 0, self.overlap_size)
        window[-self.overlap_size:] = fade_out
        
        # Apply cosine-tapered edges for smoother transitions
        taper_size = min(self.overlap_size // 4, n_samples //  8)
        if taper_size > 0:
            taper = np.cos(np.linspace(0, np.pi/2, taper_size))**2
            window[:taper_size] *= taper
            window[-taper_size:] *= taper[::-1]
        
        return window

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        collision_margin = config.system.collision_margin
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
        spsf = sample_rate / sfps # Samples Per SubFrame

        collisions = []
        collision_data = self.entity_manager.get('collisions')
        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                if config_obj.static:
                    # exit: obj_idx are static
                    return
                elif not config_obj.static:
                    for c_idx in collision_data.keys():
                        if (collision_data[c_idx].obj1_idx == obj_idx or collision_data[c_idx].obj2_idx == obj_idx) and collision_data[c_idx].valid:
                            collisions.append(collision_data[c_idx])
                    trajectories = self.entity_manager.get('trajectories')
                    for t_idx in trajectories.keys():
                        if trajectories[t_idx].obj_idx == obj_idx:
                            trajectory = trajectories[t_idx]
                            forces = self.entity_manager.get('forces')

        fracture_frame = None
        if not config_obj.fractured == False:
            fracture_frame = config_obj.fractured
            fracture_frame *= sample_rate / sfps

        is_shard_frame = None
        if not config_obj.is_shard == False:
            is_shard_frame = config_obj.is_shard
            is_shard_frame *= sample_rate / sfps

        # Calculate total duration in samples
        frames = trajectory.get_x()
        total_samples = int(trajectory.get_x()[-1])

        # Init tracks
        impact_track = np.zeros(total_samples)
        sliding_track = np.zeros(total_samples)
        scraping_track = np.zeros(total_samples)
        rolling_track = np.zeros(total_samples)
        sliding_sound = np.zeros(total_samples)
        scraping_sound = np.zeros(total_samples)
        rolling_sound = np.zeros(total_samples)
        non_collision_track = np.zeros(total_samples)
        coupling_strength_track = np.zeros(total_samples)

        synthesized_track = self._create_empty_tracks(total_samples)
        for sample_idx in frames:
            if (fracture_frame == None or sample_idx <= fracture_frame) and (is_shard_frame == None or is_shard_frame <= sample_idx):
#                # Synthesize non-collision forces (air resistance, etc etc.)
#                for f_idx in forces.keys():
#                    if forces[f_idx].obj_idx == obj_idx:
#                        force = forces[f_idx]
#                        synthesized_track = self._synthesize_non_collision(force, config_obj, sample_idx, total_samples, sample_rate)
                # Check if this frame contains a collision
                for collision in collisions:
                    if collision.frame == sample_idx:
                        other_obj_idx = collision.obj2_idx if collision.obj1_idx == obj_idx else collision.obj1_idx
                        for conf_obj in config.objects:
                            if conf_obj.idx == other_obj_idx:
                                other_config_obj = conf_obj 
                        for f_idx in forces.keys():
                            if forces[f_idx].obj_idx == obj_idx and forces[f_idx].other_obj_idx == other_obj_idx:
                                force = forces[f_idx]
                        if sample_idx >= force.frames[0] and sample_idx <= force.frames[-1]:
                            if collision.type.value == 'impact':
                                # Synthesize impact sound using Hertzian model
                                synthesized_track = self._synthesize_impact(force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                            elif collision.type.value == 'contact':
                                # Synthesize contact sound
                                synthesized_track = self._synthesize_contact(trajectory, force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate, sfps, spsf)
                                # Synthesize impact sound using Hertzian model
                                synthesized_impact_track = self._synthesize_impact(force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                                for key in synthesized_impact_track.keys():
                                    synthesized_track[key] += synthesized_impact_track[key]

                    # Add to tracks with overlap-add processing
                    impact_track = self._apply_overlap_add(impact_track, synthesized_track['impact'], int(sample_idx), int(sample_idx + len(synthesized_track['impact'])))
                    sliding_track = self._apply_overlap_add(sliding_track, synthesized synthesized_track['sliding'], int(sample_idx), int(sample_idx + len(synthesized_track['sliding'])))
                    scraping_track = self._apply_overlap_add(scraping_track, synthesized_track['scraping'], int(sample_idx), int(sample_idx + len(synthesized_track['scraping'])))
                    rolling_track = self._apply_overlap_add(rolling_track, synthesized_track['rolling'], int(sample_idx), int(sample_idx + len(synthesized_track['rolling'])))
                    non_collision_track = self._apply_overlap_add(non_collision_track, synthesized_track['non_collision'], int(sample_idx), int(sample_idx + len(synthesized_track['non_collision'])))
                    coupling_strength_track = self._apply_overlap_add(coupling_strength_track, synthesized_track['coupling_strength'], int(sample_idx), int(sample_idx + len(synthesized_track['coupling_strength'])))
                    sliding_sound = self._apply_overlap_add(sliding_sound, synthesized_track['sliding_sound'], int(sample_idx), int(sample_idx + len(synthesized_track['sliding_sound'])))
                    scraping_sound = self._apply_overlap_add(scraping_sound, synthesized_track['scraping_sound'], int(sample_idx), int(sample_idx + len(synthesized_track['scraping_sound'])))
                    rolling_sound = self._apply_overlap_add(rolling_sound, synthesized_track['rolling_sound'], int(sample_idx), int(sample_idx + len(synthesized_track['rolling_sound'])))

        # Save tracks
        tracks = {
            'impact': impact_track,
            'sliding': sliding_track,
            'scraping': scraping_track,
            'rolling': rolling_track,
            'sliding_sound': sliding_sound,
            'scraping_sound': scraping_sound,
            'rolling_sound': rolling_sound,
            'non_collision': non_collision_track,
            'coupling_strength': coupling_strength_track
        }
        
        self._save_tracks(config_obj, tracks, total_samples, int(sample_rate))

    def _synthesize_impact(self, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int) -> Dict[str, Any]:
        """Synthesize Hertzian impact audio-force with smooth envelope."""
        # Hertzian impact parameters
        if config_obj.stochastic_variation:
            normal_force_mag = np.linalg.norm(force.get_stochastic_normal_force(sample_idx))
        else:
            normal_force_mag = force.get_normal_force_magnitude(sample_idx)
        relative_velocity_mag = np.linalg.norm(force.get_relative_velocity(sample_idx))
        
        # Get impact duration using Hertzian theory
        # R_eff = Effective radius, E_star = Effective Young's modulus, impact_duration in seconds
        impact_duration = force.get_impact_duration(sample_idx)

        # Calculate impulse
        impulse = normal_force_mag * impact_duration

        # impact_duration in samples@sample_rate
        total_impact_sample = int(sample_rate * impact_duration)

        # Generate impact envelope (Hertzian asymmetric force profile)
        rise_sample = int(total_impact_sample / 2) + 1
        decay_sample = int(total_impact_sample / 2)
        t_rise = np.linspace(0, 0.5, max(int(rise_sample), 2))
        t_decay = np.linspace(0.5, 1, max(int(decay_sample), 2))
        t = np.concatenate((t_rise, t_decay[1:]))

        # Hertzian force profile: F(t) = F_max * (1 - (t/T)^(3/2))
        # Using qualitative correct form 1 − cos(2πt/τ ) for 0 ≤ t ≤ τ,
        # with τ the total duration as in van den Doel FoleyAutomatic
        F_max = normal_force_mag
        force_envelope = F_max * (1 - np.cos(2 * np.pi * t))/2
        
        # Normalize to match total impulse
        actual_impulse = np.trapz(force_envelope, t)
        if actual_impulse > 0:
            force_envelope = force_envelope * (impulse / actual_impulse)
        
        # Apply smooth window to the envelope
        smooth_window = self._create_smooth_window(len(force_envelope))
        force_envelope *= smooth_window
        
        # Create impact signal and coupling strength signal
        impact_samples, coupling_strength = (np.zeros(total_samples) for _ in range(2))
        
        # Calculate start and end indices with overlap consideration
        start_idx = max(0, int(sample_idx - rise_sample))
        end_idx = min(total_samples, start_idx + len(force_envelope))
        actual_len = end_idx - start_idx
        
        if actual_len > 0:
            # Apply overlap-add for the impact signal
            impact_segment = np.zeros(actual_len)
            impact_segment[:min(len(force_envelope), actual_len)] = force_envelope[:min(len(force_envelope), actual_len)]
            impact_track = np.zeros(total_samples)
            impact_track = self._apply_overlap_add(impact_track, impact_segment, start_idx, end_idx)
            impact_samples = impact_track
            
            # Coupling strength with smooth transition
            coupling_segment = np.full(actual_len, force.get_coupling_strength(sample_idx))
            coupling_track = np.zeros(total_samples)
            coupling_track = self._apply_overlap_add(coupling_track, coupling_segment, start_idx, end_idx)
            coupling_strength = coupling_track

        # Create tracks
        result = {
            'impact': impact_samples,
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'sliding_sound': np.zeros(total_samples),
            'scraping_sound': np.zeros(total_samples),
            'rolling_sound': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples),
            'coupling_strength': coupling_strength
        }
        
        return result

    def _synthesize_contact(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int, sfps: float, spsf: float):
        """Synthesize contact audio-force (scraping, sliding, rolling) with overlap-add."""
        
        if force.get_contact_type(sample_idx) in [ContactType.STATIC.value, ContactType.NO_CONTACT.value]:
            return self._create_empty_tracks(total_samples)
        elif force.get_contact_type(sample_idx) == ContactType.ROLLING.value:
            return self._synthesize_rolling(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
        elif force.get_contact_type(sample_idx) == ContactType.SLIDING.value:
            return self._synthesize_sliding(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
        elif force.get_contact_type(sample_idx) == ContactType.SCRAPING.value:
            return self._synthesize_scraping(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
        elif force.get_contact_type(sample_idx) == ContactType.MIXED.value:
            return self._synthesize_mixed(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)

    def _synthesize_mixed(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize mixed contact audio-forces with overlap-add (rolling with scraping or sliding or static)."""
        mixed_tracks = self._create_empty_tracks(total_samples)
        rolling_tracks = self._create_empty_tracks(total_samples)
        scraping_tracks = self._create_empty_tracks(total_samples)
        sliding_tracks = self._create_empty_tracks(total_samples)

        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if trajectories[t_idx].obj_idx == other_config_obj.idx:
                other_trajectory = trajectories[t_idx]

        tangential_velocity = force.get_tangential_velocity(sample_idx)
        tangential_velocity = np.linalg.norm(tangential_velocity)
        tangential_force_magnitude = force.get_tangential_force_magnitude(sample_idx)
        relative_velocity = force.get_relative_velocity(sample_idx)
        relative_velocity = np.linalg.norm(relative_velocity)
        normal_force_magnitude = force.get_normal_force_magnitude(sample_idx)

        vertices1 = trajectory.get_vertices(sample_idx)
        vertices2 = other_trajectory.get_vertices(sample_idx)
        omega1 = trajectory.get_angular_velocity(sample_idx)
        omega2 = other_trajectory.get_angular_velocity(sample_idx)

        # Get material properties
        roughness1 = config_obj.acoustic_shader.roughness
        roughness2 = other_config_obj.acoustic_shader.roughness
        friction1 = config_obj.acoustic_shader.friction
        friction2 = other_config_obj.acoustic_shader.friction

        # Analyzes mixed contact HertzianContact lib.
        hertzian_contact = HertzianContact(self.entity_manager)
        mixed_factor = hertzian_contact.get_mixed_factor(relative_velocity, tangential_velocity, normal_force_magnitude, tangential_force_magnitude, omega1, omega2, roughness1, roughness2, friction1, friction2, vertices1, vertices2)
        
        if mixed_factor['static_factor'] == 1:
            tmp_config_obj = config_obj
            config_obj = other_config_obj
            other_config_obj = tmp_config_obj
            trajectory = other_trajectory
            if mixed_factor['sliding_factor'] == 1:
                sliding_tracks = self._synthesize_sliding(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
            elif mixed_factor['scraping_factor'] == 1:
                scraping_tracks = self._synthesize_scraping(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
        else:
            mixed_tracks = self._synthesize_rolling(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
            if mixed_factor['sliding_factor'] > 0:
                sliding_tracks = self._synthesize_sliding(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
            elif mixed_factor['scraping_factor'] > 0:
                scraping_tracks = self._synthesize_scraping(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)

        # Apply overlap-add for mixed factors
        for key in mixed_tracks.keys():
            mixed_tracks[key] += rolling_tracks[key] * mixed_factor['rolling_factor']
            mixed_tracks[key] += sliding_tracks[key] * mixed_factor['sliding_factor']
            mixed_tracks[key] += scraping_tracks[key] * mixed_factor['scraping_factor']

        return mixed_tracks
        
    def _synthesize_non_collision(self, force: Any, config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize non-collision audio-forces (air resistance, etc.)."""
        # Create tracks
        non_collision = np.zeros(total_samples)
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'sliding_sound': np.zeros(total_samples),
            'scraping_sound': np.zeros(total_samples),
            'rolling_sound': np.zeros(total_samples),
            'non_collision': non_collision,
            'coupling_strength': np.zeros(total_samples)
        }

        return result

    def _synthesize_scraping(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize scraping sound using fractal noise with resonant filter and overlap-add."""
    
        # Get material properties
        roughness = config_obj.acoustic_shader.roughness
        other_roughness = other_config_obj.acoustic_shader.roughness
    
        # Estimate Rq = Ra * 1.1 (RMS roughness from average roughness)
        Rq = roughness * 1.1
        other_Rq = other_roughness * 1.1
        Rq_effective = math.sqrt(Rq**2 + other_Rq**2)
        roughness_rms = Rq_effective / 1.1
    
        # Normalize roughness to 0-1 range for parameter control

        B = (roughness_rms - 0.00001) / (0.99999 - 0.00001)
    
        # Get contact duration in samples
        n_samples = int(collision.frame_range)
    
        # Collect force and velocity data over the contact duration
        total_noise, coupling_strength = (np.zeros(total_samples) for _ in range(2))
        contact_velocities, normal_forces, tangential_forces = (np.zeros(n_samples) for _ in range(3))
        for s_idx in range(n_samples):
            current_idx = int(sample_idx) + s_idx
            if config_obj.stochastic_variation:
                normal_forces[s_idx] = np.linalg.norm(force.get_stochastic_normal_force(current_idx))
                tangential_forces[s_idx] = np.linalg.norm(force.get_stochastic_tangential_force(current_idx))
            else:
                normal_forces[s_idx] = np.linalg.norm(force.get_normal_force(current_idx))
                tangential_forces[s_idx] = np.linalg.norm(force.get_tangential_force(current_idx))
            contact_velocities[s_idx] = np.linalg.norm(force.get_relative_velocity(current_idx))
            coupling_strength[current_idx] = force.get_coupling_strength(current_idx)
    
        # Frequency array for FFT
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    
        # Pre-calculate fractal frequency response (1/f^B noise)
        mask = freqs != 0
        fractal_response = np.ones_like(freqs, dtype=complex)
        fractal_response[mask] = (2 * np.pi * np.abs(freqs[mask])) ** (B/2)
    
        # Generate multiple noise layers for richer scraping sound
        num_layers = 3
        layer_weights = [0.6, 0.3, 0.1]  # Weight for each layer
    
        for layer in range(num_layers):
            # Generate white noise
            white_noise = np.random.randn(n_samples)
        
            # Apply fractal spectrum
            noise_fft = np.fft.fft(white_noise) * fractal_response
        
            # Design resonant filters for scraping
            # Higher frequencies for scraping compared to sliding
            f0_base = 2000 + 8000 * np.abs(contact_velocities)
            f0 = f0_base * (1 + 0.5 * layer)  # Different center frequency for each layer

            f0 = np.clip(f0, 100, sample_rate/2 - 1)
        
            # Q factor depends on roughness - smoother surfaces have higher Q
            Q = 5.0 / (roughness_rms + 1e-6)
            Q = np.clip(Q, 0.5, 50.0)
        
            # Apply resonant filter to each sample
            resonant_noise = np.zeros(n_samples)
            for s_idx in range(n_samples):
                current_idx = int(sample_idx + s_idx)
                # Design and apply resonant filter
                b, a = signal.iirpeak(f0[s_idx], Q, fs=sample_rate)
                w, h = signal.freqz(b, a, worN=n_samples, fs=sample_rate)
            
                # Apply filter in frequency domain
                layer_noise_fft = noise_fft * h
                layer_noise = np.real(np.fft.ifft(layer_noise_fft))
                resonant_noise[s_idx] = layer_noise[s_idx]
        
            # Apply amplitude modulation based on forces and velocities
            # Scraping has stronger modulation with tangential force
            amplitude = np.sqrt(np.abs(contact_velocities) * tangential_forces)
        
            # Add amplitude modulation from surface irregularities
            irregularity_freq = 50 + 200 * roughness_rms  # Hz
            t = np.arange(n_samples) / sample_rate
            irregularity_mod = 1 + 0.3 * np.sin(2 * np.pi * irregularity_freq * t)
        
            # Combine amplitude factors
            total_amplitude = amplitude * irregularity_mod
        
            # Add this layer to total noise with overlap-add
            start_idx = int(sample_idx)
            stop_idx = start_idx + n_samples
            layer_segment = np.zeros(n_samples)
            layer_segment[:n_samples] = layer_weights[layer] * resonant_noise * total_amplitude
            total_noise = self._apply_overlap_add(total_noise, layer_segment, start_idx, stop_idx)
    
        # Add transient spikes for scraping events (sudden catches/releases)
        if n_samples > 100:
            num_spikes = int(3 + 5 * roughness_rms)
            spike_positions = np.random.choice(n_samples - 20, num_spikes, replace=False)
        
            for spike_pos in spike_positions:
                spike_width = int(5 + 10 * np.random.rand())
                spike_start = int(sample_idx) + max(0, spike_pos - spike_width//2)
                spike_end = int(sample_idx) + min(n_samples, spike_pos + spike_width//2)
                spike_len = spike_end - spike_start
            
                # Create spike envelope
                spike_env = signal.windows.tukey(spike_len, alpha=0.3)
            
                # Create spike signal (impulse-like)
                spike_signal = np.random.randn(spike_len) * spike_env
            
                # Scale spike by tangential force at that moment
                spike_scale = tangential_forces[spike_pos] / np.max(tangential_forces + 1e-10)
                
                # Apply overlap-add for spike
                spike_segment = spike_signal * spike_scale * 2.0
                total_noise = self._apply_overlaplap_add(total_noise, spike_segment, spike_start, spike_end)
    
        scraping_sound = total_noise

        """Generate scraping vibration signal with overlap-add."""
        # Base frequency depends on velocity and roughness
        base_freq = 2000 + 5000 * np.mean(contact_velocities) * Rq_effective
        
        # Create modulated noise
        t = np.arange(n_samples) / sample_rate
        noise = np.random.randn(n_samples) * 0.1
        
        # Amplitude modulation based on tangential force
        amplitude_env = tangential_forces / np.max(tangential_forces + 1e-10)
        
        # Frequency modulation based on velocity
        freq_mod = 1.0 + 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz modulation
        
        # Generate scraping_signal
        scraping_signal = np.zeros(n_samples)
        for i in range(n_samples):
            freq = base_freq * freq_mod[i]
            if freq < sample_rate / 2:  # Nyquist limit
                scraping_signal[i] = np.sin(2 * np.pi * freq * t[i]) * amplitude_env[i]
        
        # Add noise component
        scraping_signal += noise * 0.3
        
        # Apply smooth envelope
        smooth_window = self._create_smooth_window(n_samples)
        scraping_signal *= smooth_window
        
        # Add scraping_signal to empty track with overlap-add
        start_idx = int(sample_idx)
        stop_idx = start_idx + n_samples
        scraping_vibration = np.zeros(total_samples)
        scraping_vibration = self._apply_overlap_add(scraping_vibration, scraping_signal, start_idx, stop_idx)

        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': scraping_vibration,
            'rolling': np.zeros(total_samples),
            'sliding_sound': np.zeros(total_samples),
            'scraping_sound': scraping_sound,
            'rolling_sound': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples),
            'coupling_strength': coupling_strength
        }
    
        return result

    def _synthesize_sliding(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize sliding sound using fractal noise with resonant filter and overlap-add."""

        # Root Mean Square Roughness
        roughness = config_obj.acoustic_shader.roughness
        other_roughness = config_obj.acoustic_shader.roughness
        # Estimate Rq = Ra * 1.1
        Rq = roughness * 1.1
        other_Rq = other_roughness * 1.1
        Rq_effective = math.sqrt(Rq**2 + other_Rq**2)
        roughness_rms = Rq_effective / 1.1
        B = (roughness_rms - 0.00001) / (0.99999 - 0.00001)

        n_samples = int(collision.frame_range)
        total_noise, coupling_strength = (np.zeros(total_samples) for _ in range(2))
        contact_velocities, normal_forces = (np.zeros(n_samples) for _ in range(2))
        for s_idx in range(n_samples):
            current_idx = int(sample_idx) + s_idx
            if config_obj.stochastic_variation:
                normal_forces[s_idx] = np.linalg.norm(force.get_stochastic_normal_force(current_idx))
            else:
                normal_forces[s_idx] = np.linalg.norm(force.get_normal_force(current_idx))
            contact_velocities[s_idx] = np.linalg.norm(force.get_relative_velocity(current_idx))
            coupling_strength[current_idx] = force.get_coupling_strength(current_idx)
    
        # Frequency array for FFT
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    
        # Pre-calculate fractal frequency response
        mask = freqs != 0
        fractal_response = np.ones_like(freqs, dtype=complex)
        fractal_response[mask] = (2 * np.pi * np.abs(freqs[mask])) ** (B/2)

        # Generate white noise
        white_noise = np.random.randn(n_samples)
        
        # Apply fractal spectrum
        noise_fft = np.fft.fft(white_noise) * fractal_response
        
        # Design resonant filter for this contact
        f0 = 1000 + 5000 * np.abs(contact_velocities)
        f0 = np.clip(f0, 20, sample_rate/2 - 1)
        
        Q = 10.0 / (roughness_rms + 1e-6)
        Q = np.clip(Q, 0.1, 100.0)

        resonant_noises = np.zeros(n_samples)
        for s_idx in range(n_samples):
            current_idx = int(sample_idx + s_idx)
            # Design and apply resonant filter
            b, a = signal.iirpeak(f0[s_idx], Q, fs=sample_rate)
            w, h = signal.freqz(b, a, worN=n_samples, fs=sample_rate)
        
            resonant_noise_fft = noise_fft * h
            resonant_noise = np.real(np.fft.ifft(resonant_noise_fft))
            resonant_noises[s_idx] = resonant_noise[s_idx]

        # Apply amplitude scaling with overlap-add
        start_idx = int(sample_idx)
        stop_idx = start_idx + n_samples
        amplitude = np.sqrt(np.abs(contact_velocities) * normal_forces)
        
        # Create smooth segment
        sliding_segment = resonant_noises * amplitude
        smooth_window = self._create_smooth_window(n_samples)
        sliding_segment *= smooth_window
        
        total_noise = self._apply_overlap_add(total_noise, sliding_segment, start_idx, stop_idx)

        sliding_sound = total_noise

        """Generate sliding vibration signal with overlap-add."""
        # Base frequency depends on velocity

        base_freq = 500 + 2000 * np.mean(contact_velocities)

        # Create modulated signal
        t = np.arange(n_samples) / sample_rate
        noise = np.random.randn(n_samples) * 0.05

        # Amplitude modulation based on normal force
        amplitude_env = normal_forces / np.max(normal_forces + 1e-10)

        # Generate sliding_signal
        sliding_signal = np.zeros(n_samples)
        for i in range(n_samples):
            # Slight frequency variation
            freq_variation = 1.0 + 0.1 * np.sin(2 * np.pi * 20 * t[i])
            freq = base_freq * freq_variation

            if freq < sample_rate / 2:  # Nyquist limit
                sliding_signal[i] = np.sin(2 * np.pi * freq * t[i]) * amplitude_env[i]

        # Add noise component (less than scraping)
        sliding_signal += noise * 0.1

        # Apply smooth envelope
        smooth_window = self._create_smooth_window(n_samples)
        sliding_signal *= smooth_window

        # Add sliding_signal to empty track with overlap-add
        sliding_vibration = np.zeros(total_samples)
        sliding_vibration = self._apply_overlap_add(sliding_vibration, sliding_signal, start_idx, stop_idx)

        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': sliding_vibration,
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'sliding_sound': sliding_sound,
            'scraping_sound': np.zeros(total_samples),
            'rolling_sound': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples),
            'coupling_strength': coupling_strength
        }

        return result

    def _synthesize_rolling(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize rolling sound using Poisson pulse sequence filtered by second-order resonant filter with overlap-add."""
        # Extract parameters
        n_samples = int(collision.frame_range)
        rolling_vibration, rolling_signal, coupling_strength = (np.zeros(total_samples) for _ in range(3))
        normal_forces, tangential_velocities, angular_velocities, contact_velocity = (np.zeros(n_samples) for _ in range(4))
        
        for s_idx in range(n_samples):
            current_idx = int(sample_idx) + s_idx
            if config_obj.stochastic_variation:
                normal_forces[s_idx] = np.linalg.norm(force.get_stochastic_normal_force(current_idx))
            else:
                normal_forces[s_idx] = np.linalg.norm(force.get_normal_force(current_idx))
            tangential_velocities[s_idx] = np.linalg.norm(force.get_tangential_velocity(current_idx))
            angular_velocities[s_idx] = np.linalg.norm(trajectory.get_angular_velocity(current_idx))
            contact_velocity[s_idx] = np.linalg.norm(force.get_relative_velocity(current_idx))
            coupling_strength[current_idx] = force.get_coupling_strength(current_idx)

        # Time array
        t = np.arange(n_samples) / sample_rate

        # Get material properties for irregularity estimation
        roughness = config_obj.acoustic_shader.roughness
        other_roughness = other_config_obj.acoustic_shader.roughness

        # Estimate ovality/irregularity from roughness
        # Ovality factor: 0 = perfect sphere, 1 = highly irregular
        # TODO: use rolling_radius delta in (t)
        ovality = 0.1 + 0.9 * (roughness + other_roughness) / 2.0
        ovality = np.clip(ovality, 0.1, 0.9)

        # Base pulse rate from angular velocity (revolutions per second)
        # Each revolution causes multiple impacts due to surface irregularities
        avg_angular_velocity = np.mean(np.abs(angular_velocities))
        base_pulse_rate = avg_angular_velocity / (2 * np.pi)  # Convert rad/s to Hz

        # Scale by ovality: more irregular = more pulses per revolution
        pulse_rate = base_pulse_rate * (1.0 + 10.0 * ovality)
        pulse_rate = np.clip(pulse_rate, 1.0, 1000.0)

        # Generate Poisson pulse sequence
        # Average pulse rate λ (pulses per second) depends on angular velocity
        # and surface irregularity (ovality)
        poisson_pulses = self._generate_poisson_pulse_sequence(n_samples=n_samples, sample_rate=sample_rate, pulse_rate=pulse_rate, amplitude_env=normal_forces / np.max(normal_forces + 1e-10))

        # Design spectral envelope S(w) = 1 / sqrt((w - p)² + d²)
        # where p is center frequency, d is damping

        # Center frequency p depends on material properties and normal force
        # Higher normal force = higher frequency (stiffer contact)
        avg_normal_force = np.mean(normal_forces)
        p_base = 50.0 + 200.0 * avg_normal_force * 0.001  # Hz
        p_variation = 0.2 * p_base

        # Damping d depends on material damping and velocity
        # Higher velocity = more damping
        avg_velocity = np.mean(tangential_velocities)
        d_base = 10.0 + 20.0 * avg_velocity
        d_variation = 0.3 * d_base

        # Time-varying parameters for richer sound
        p = p_base + p_variation * np.sin(2 * np.pi * 0.5 * t)
        d = d_base + d_variation * np.sin(2 * np.pi * 0.3 * t)

        # Apply resonant filter to Poisson pulses with overlap-add
        filtered_pulses = np.zeros(n_samples)
        for s_idx in range(n_samples):
            # Create IIR filter that approximates the spectral envelope
            # Using a bandpass filter with specific Q
            Q = p[s_idx] / (2 * d[s_idx]) if d[s_idx] > 0 else 10.0
            nyquist = sample_rate / 2
            center_normalized = p[s_idx] / nyquist
        
            # Design resonant filter
            b, a = signal.iirpeak(center_normalized, Q, fs=sample_rate)

            # Apply filter (using convolution in frequency domain for efficiency)
            # We'll apply to a window around the current sample
            window_size = min(1024, n_samples - s_idx)
            if window_size > 10:
                window_signal = poisson_pulses[s_idx:s_idx + window_size]
                filtered_window = signal.lfilter(b, a, window_signal)
                filtered_pulses[s_idx] = filtered_window[0]

        # Amplitude modulation
        # Modulation frequency proportional to relative speed/angular velocity
        # Modulation depth proportional to ovality

        # Base modulation frequency (Hz)
        mod_freq_base = 1.0 + 5.0 * avg_angular_velocity / (2 * np.pi)

        # Add some variation
        mod_freq = mod_freq_base * (1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t))

        # Modulation depth: 0 = no modulation, 1 = full modulation
        mod_depth = 0.3 + 0.7 * ovality

        # Create amplitude modulation signal
        mod_signal = 1.0 - mod_depth + mod_depth * np.sin(2 * np.pi * mod_freq * t)

        # Apply amplitude modulation
        modulated_signal = filtered_pulses * mod_signal

        # Amplitude scales with tangential velocity (rolling speed)
        velocity_scale = tangential_velocities / np.max(tangential_velocities + 1e-10)
        velocity_scale = np.clip(velocity_scale, 0.1, 1.0)

        # Apply velocity scaling
        rolling_sound_segment = modulated_signal * velocity_scale

        # Add some filtered noise for texture (surface roughness effects)
        noise_level = 0.1 * ovality
        texture_noise = np.random.randn(n_samples) * noise_level

        # Apply same resonant filter to noise
        filtered_noise = np.zeros(n_samples)
        for s_idx in range(n_samples):
            if s_idx % 100 == 0: # Update filter less frequently for efficiency
                nyquist = sample_rate / 2
                center_normalized = p[s_idx] / nyquist
                Q = p[s_idx] / (2 * d[s_idx]) if d[s_idx] > 0 else 10.0
                Q = np.clip(Q, 0.5, 100.0)
                b, a = signal.iirpeak(center_normalized, Q, fs=sample_rate)
        
            window_size = min(512, n_samples - s_idx)
            if window_size > 10:
                window_noise = texture_noise[s_idx:s_idx + window_size]
                filtered_window = signal.lfilter(b, a, window_noise)
                filtered_noise[s_idx] = filtered_window[0]

        # Combine pulse signal and noise
        rolling_sound_segment = rolling_sound_segment + 0.3 * filtered_noise

        # Apply smooth envelope
        smooth_window = self._create_smooth_window(n_samples)
        rolling_sound_segment *= smooth_window

        # Add rolling_sound to track with overlap-add
        start_idx = int(sample_idx)
        stop_idx = start_idx + n_samples
        rolling_signal = self._apply_overlap_add(rolling_signal, rolling_sound_segment, start_idx, stop_idx)

        # Generate vibration signal with harmonic content related to rolling frequency and overlap-add

        vibration_signal = self._generate_rolling_vibration(n_samples=n_samples, sample_rate=sample_rate, angular_velocities=angular_velocities, normal_forces=normal_forces, ovality=ovality)
        
        # Apply smooth envelope to vibration signal as well
        vibration_signal *= smooth_window
        rolling_vibration = self._apply_overlap_add(rolling_vibration, vibration_signal, start_idx, stop_idx)

        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': rolling_signal,
            'sliding_sound': np.zeros(total_samples),
            'scraping_sound': np.zeros(total_samples),
            'rolling_sound': rolling_vibration,
            'non_collision': np.zeros(total_samples),
            'coupling_strength': coupling_strength
        }
        
        return result

    def _generate_poisson_pulse_sequence(self, n_samples: int, sample_rate: int, pulse_rate: float, amplitude_env: np.ndarray) -> np.ndarray:
        """
        Generate a Poisson pulse sequence with time-varying amplitude.
        Pulses are shaped with smooth windows to avoid clicks.
    
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        sample_rate : int
            Sample rate in Hz
        pulse_rate : float
            Average pulse rate in Hz (λ parameter)
        amplitude_env : np.ndarray
            Amplitude envelope for pulses
    
        Returns:
        --------
        np.ndarray : Poisson pulse sequence
        """
        pulse_sequence = np.zeros(n_samples)
        current_sample = 0
        
        while current_sample < n_samples:
            interval_seconds = np.random.exponential(1.0 / max(pulse_rate, 0.001))
            interval_samples = int(interval_seconds * sample_rate)
            current_sample += interval_samples
        
            if current_sample < n_samples:
                pulse_amplitude = amplitude_env[min(current_sample, len(amplitude_env)-1)]
            
                # Create short pulse with smooth shape (Tukey window)
                pulse_width = min(5, n_samples - current_sample)
                if pulse_width > 1:
                    pulse_shape = signal.windows.tukey(pulse_width, alpha=0.7)  # More taper for smoother edges
                    for i in range(pulse_width):
                        idx = current_sample + i
                        if idx < n_samples:
                            pulse_sequence[idx] += pulse_amplitude * pulse_shape[i] * np.random.randn()
                else:
                    # Single sample pulse - use a small impulse
                    pulse_sequence[current_sample] += pulse_amplitude * 0.1 * np.random.randn()
    
        return pulse_sequence

    def _generate_rolling_vibration(self, n_samples: int, sample_rate: int, angular_velocities: np.ndarray, normal_forces: np.ndarray, ovality: float) -> np.ndarray:
        """
        Generate vibration signal for resonance synthesis from rolling contact.
        Uses smooth envelopes to avoid clicks.
    
        Parameters:
        -----------
        n_samples : int
            Number of samples
        sample_rate : int
            Sample rate in Hz
        angular_velocities : np.ndarray
            Angular velocity over time
        normal_forces : np.ndarray
            Normal force over time
        ovality : float
            Ovality factor (0-1)
    
        Returns:
        --------
        np.ndarray : Vibration signal for modal excitation
        """
        t = np.arange(n_samples) / sample_rate
        base_freq = np.abs(angular_velocities) / (2 * np.pi)
        
        num_harmonics = int(3 + 7 * ovality)
        harmonics = np.arange(1, num_harmonics + 1)
        
        vibration = np.zeros(n_samples)
        amplitude_env = normal_forces / np.max(normal_forces + 1e-10)

        for i in range(n_samples):
            if base_freq[i] > 0.1:
                for harmonic in harmonics:
                    freq = base_freq[i] * harmonic
                    if freq < sample_rate / 2:
                        harmonic_amp = amplitude_env[i] / (harmonic ** (1.5 - ovality))
                        phase_mod = 0.1 * ovality * np.sin(2 * np.pi * 2.0 * t[i])
                        vibration[i] += harmonic_amp * np.sin(2 * np.pi * freq * t[i] + phase_mod)
    
        noise_level = 0.1 * ovality
        vibration += np.random.randn(n_samples) * noise_level * amplitude_env
    
        # Apply smooth envelope to avoid boundary clicks
        smooth_window = self._create_smooth_window(n_samples)
        vibration *= smooth_window
    
        return vibration

    def _create_empty_tracks(self, total_samples: int) -> Dict:
        """Create empty tracks for silent sections."""
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'sliding_sound': np.zeros(total_samples),
            'scraping_sound': np.zeros(total_samples),
            'rolling_sound': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples),
            'coupling_strength': np.zeros(total_samples)
        }
        
        return result

    def _save_tracks(self, config_obj: Any, tracks: Dict[str, np.ndarray], total_samples: int, sample_rate: int):
        """
        Save individual tracks as WAV files.
        Create a json multitrack project file (e.g., for Reaper, Ardour).
        """
        project_data = {
            'object_name': config_obj.name,
            'sample_rate': sample_rate,
            'duration': total_samples / sample_rate,
            'tracks': []
        }
        
        for track_name, track_data in tracks.items():
            track_file = f"{config_obj.name}_{track_name}.raw"
            wave_file = f"{self.audio_force_dir}/{track_file}"
            sf.write(wave_file, track_data, sample_rate, subtype='FLOAT')
            project_data['tracksracks'].append({
                'name': track_name,
                'file': track_file,
                'channels': 1,
                'position': 0.0,
                'volume': 1.0,
                'pan': 0.0
            })
            print(f"Saved {track_name} tracks to {self.audio_force_dir}")

        # Save project file
        json_file = f"{self.audio_force_dir}/{config_obj.name}.json"
        with open(json_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        print(f"Created multitrack project: {json_file}")

