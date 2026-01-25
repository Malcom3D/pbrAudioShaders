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
import json
import resampy
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal
from dataclasses import dataclass 
from typing import List, Dict, Tuple, Optional, Any

from ..core.entity_manager import EntityManager
from ..lib.hertzian_contact import HertzianContact
from ..lib.contact_classifier import ContactClassifier

@dataclass
class ForceSynth:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.hertzian_contact = HertzianContact(self.entity_manager)
        self.contact_classifier = ContactClassifier(window_size=10)
        self.output_dir = f"{config.system.cache_path}/audio-force"
        os.makedirs(self.output_dir, exist_ok=True)


    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        collision_margin = config.system.collision_margin
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        trajectories = self.entity_manager.get('trajectories')
        forces = self.entity_manager.get('forces')
        collision_data = self.entity_manager.get('collisions')
        collisions = []
        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                if not config_obj.static: 
                    for t_idx in trajectories.keys():
                        if trajectories[t_idx].obj_idx == obj_idx:
                            trajectory = trajectories[t_idx]
                    for f_idx in forces.keys():
                        if forces[f_idx].obj_idx == obj_idx:
                            force = forces[f_idx]
                    for c_idx in collision_data.keys():
                        if collision_data[c_idx].obj1_idx == obj_idx or collision_data[c_idx].obj2_idx == obj_idx:
                            collisions.append(collision_data[c_idx])

        # Calculate total duration in samples
        frames = force.frames
        total_samples = int(trajectory.get_x()[-1])

        # Init tracks
        impact_track = np.zeros(total_samples)
        sliding_track = np.zeros(total_samples)
        scraping_track = np.zeros(total_samples)
        rolling_track = np.zeros(total_samples)
        non_collision_track = np.zeros(total_samples)

        frame_samples = self._create_empty_tracks(total_samples)
        for sample_idx in frames:
#            # Synthesize non-collision forces (air resistance, etc etc.)
#            frame_samples = self._synthesize_non_collision(force, config_obj, sample_idx, total_samples, sample_rate)
            # Check if this frame contains a collision
            for collision in collisions:
                if collision.frame == sample_idx:
                    other_obj_idx = collision.obj2_idx if collision.obj1_idx == obj_idx else collision.obj1_idx
                    for conf_obj in config.objects:
                        if conf_obj.idx == other_obj_idx:
                            other_config_obj = conf_obj 
                    if collision.type.value == 'impact': 
                        # Synthesize impact sound using Hertzian model
                        frame_samples = self._synthesize_impact(force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                    elif collision.type.value == 'contact': 
                        # Synthesize impact sound using Hertzian model
                        impact_samples = self._synthesize_impact(force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                        # Synthesize contact sound
                        frame_samples = self._synthesize_contact(trajectory, force, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                        for key in frame_samples.keys():
                            frame_samples[key] += impact_samples[key]

                # Add to tracks
                impact_track += frame_samples['impact']
                sliding_track += frame_samples['sliding']
                scraping_track += frame_samples['scraping']
                rolling_track += frame_samples['rolling']
                non_collision_track += frame_samples['non_collision']
        
        # Save tracks
        tracks = {
            'impact': impact_track,
            'sliding': sliding_track,
            'scraping': scraping_track,
            'rolling': rolling_track,
            'non_collision': non_collision_track
        }
        
        self._save_tracks(config_obj, tracks, total_samples, sample_rate)

    def _synthesize_impact(self, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int) -> Dict[str, Any]:
        """Synthesize Hertzian impact audio-force."""
        # Hertzian impact parameters
        normal_force_mag = force.get_normal_force_magnitude(sample_idx)
        relative_velocity_mag = np.linalg.norm(force.get_relative_velocity(sample_idx))
        
        # Get impact duration using Hertzian theory
        # R_eff = Effective radius, E_star = Effective Young's modulus, impact_duration in seconds
        R_eff, E_star, impact_duration = self.hertzian_contact.compute(config_obj.idx, other_config_obj.idx, sample_idx)

        # Calculate impulse
        impulse = normal_force_mag * impact_duration

        # impact_duration in samples@sample_rate
        total_impact_sample = int(sample_rate*impact_duration)

        # Generate impact envelope (Hertzian force profile)
        rise_sample = int(total_impact_sample / 2) + 1
        decay_sample = int(total_impact_sample / 2)
        t_rise = np.linspace(0, 0.5, int(rise_sample))
        t_decay = np.linspace(0.5, 1, int(decay_sample))
        t = np.concatenate((t_rise, t_decay[1:]))

        # update collision.impulse_range and collision.frame_range
        collision.update_impulse_range(rise_sample)
        collision.update_frame_range(total_impact_sample)
        
        # Hertzian force profile: F(t) = F_max * (1 - (t/T)^(3/2))
        # Using qualitative correct form 1 − cos(2πt/τ ) for 0 ≤ t ≤ τ,
        # with τ the total duration as in van den Doel FoleyAutomatic
        F_max = normal_force_mag
        force_envelope = F_max * (1 - np.cos(2 * np.pi * t))/2
        
        # Normalize to match total impulse
        actual_impulse = np.trapz(force_envelope, t)
        if actual_impulse > 0:
            force_envelope = force_envelope * (impulse / actual_impulse)
        
        # Create impact signal
        impact_samples = np.zeros(total_samples)
        for force_idx in range(force_envelope.shape[0]):
            idx = int(sample_idx - rise_sample) + force_idx
            if not impact_samples.shape[0] <= idx:
                impact_samples[idx] = force_envelope[force_idx]
        
        # Normalize signal
        impact_samples = impact_samples / np.max(np.abs(impact_samples))

        # Create tracks
        result = {
            'impact': impact_samples,
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples)
        }
        
        return result

    def _synthesize_contact(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize contact audio-force (scraping, sliding, rolling)."""
        # Determine contact type based on velocities and forces
        tangential_velocity_mag = np.linalg.norm(force.get_tangential_velocity(sample_idx))
        normal_force_mag = force.get_normal_force_magnitude(sample_idx)
        tangential_force_mag = force.get_tangential_force_magnitude(sample_idx)
        
        # Contact type classification
        if tangential_velocity_mag < 0.001:  # Static contact
            return self._create_empty_tracks(total_samples)
        elif round(tangential_force_mag / max(normal_force_mag, 1e-10), 2) <= 0.3:  # Rolling
            return self._synthesize_rolling(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)
        else:  # Scraping/Sliding
            return self._synthesize_sliding(trajectory=trajectory, force=force, collision=collision, config_obj=config_obj, other_config_obj=other_config_obj, sample_idx=sample_idx, total_samples=total_samples, sample_rate=sample_rate)

    def _synthesize_non_collision(self, force: Any, config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize non-collision audio-forces (air resistance, etc.)."""
        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples)
        }

        return result

    def _synthesize_sliding(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize scraping/sliding sound using fractal noise with resonant filter."""
        classification, scores = self.contact_classifier.compute(sample_idx, trajectory, force)
        # Extract parameters

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
        contact_velocities, normal_forces = ([] for _ in range(2))
        for i in range(n_samples):
            contact_velocities.append(np.linalg.norm(force.get_relative_velocity(sample_idx + i)))
            normal_forces.append(np.linalg.norm(force.get_stochastic_normal_force(sample_idx + i)))
        contact_velocities = np.array(contact_velocities)
        normal_forces = np.array(normal_forces)
    
        # Frequency array for FFT
        freqs = fftfreq(n_samples, 1/sample_rate)
    
        # Initialize total noise
        total_noise = np.zeros(n_samples)
    
        # Pre-calculate fractal frequency response
        mask = freqs != 0
        fractal_response = np.ones_like(freqs, dtype=complex)
        fractal_response[mask] = (2 * np.pi * np.abs(freqs[mask])) ** (B/2)

        # Generate white noise
        white_noise = np.random.randn(n_samples)
        
        # Apply fractal spectrum
        noise_fft = fft(white_noise) * fractal_response
        
        # Design resonant filter for this contact
        f0 = 1000 + 5000 * np.abs(contact_velocities)
        f0 = np.clip(f0, 20, sample_rate/2 - 1)
        
        Q = 10.0 / (roughness_rms + 1e-6)
        Q = np.clip(Q, 0.1, 100.0)

        resonant_noises = np.zeros(total_samples)
        for s_idx in range(n_samples):
            # Design and apply resonant filter
            b, a = signal.iirpeak(f0[s_idx], Q, fs=sample_rate)
            w, h = signal.freqz(b, a, worN=n_samples, fs=sample_rate)
        
            resonant_noise_fft = noise_fft * h
            resonant_noise = np.real(ifft(resonant_noise_fft))
            idx = int(sample_idx + s_idx)
            if not resonant_noises.shape[0] <= idx:
                resonant_noises[idx] = resonant_noise[s_idx]

        # Apply amplitude scaling
        amplitude = np.sqrt(np.abs(contact_velocities) * normal_forces)
        total_noise += resonant_noises * amplitude

        # Normalize signal
        total_noise = total_noise / np.max(np.abs(total_noise))

        if classification == 'sliding':
            sliding_signal = total_noise
            scraping_signal = np.zeros(al_samples)
        elif classification == 'scraping':
            sliding_signal = np.zeros(al_samples)
            scraping_signal = total_noise
        
        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': sliding_signal,
            'scraping': scraping_signal,
            'rolling': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples)
        }

        return result
    
    def _synthesize_rolling(self, trajectory: Any, force: Any, collision: Any, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int):
        """Synthesize rolling sound using second-order resonant filter driven by noise."""
        # Extract parameters
        n_samples = int(collision.frame_range)
        normal_forces, tangential_velocities, angular_velocities = ([] for _ in range(3))
        for i in range(n_samples):
            normal_forces.append(np.linalg.norm(force.get_stochastic_normal_force(sample_idx + i)))
            tangential_velocities.append(np.linalg.norm(force.get_tangential_velocity(sample_idx + i)))
            angular_velocities.append(np.linalg.norm(trajectory.get_angular_velocity(sample_idx + i)))
        normal_forces = np.array(normal_forces)
        tangential_velocities = np.array(tangential_velocities)
        angular_velocities = np.array(angular_velocities)

        # Generate white noise
        noise = np.random.randn(n_samples)
        
        # Design spectral envelope S(w) = 1 / sqrt((w - p)² + d²)
        # where p is center frequency, d is damping
        
        # Center frequency depends on normal force and velocity
        p = 50.0 + 100.0 * normal_forces * 0.001  # Hz
        d = 10.0 + 5.0 * tangential_velocities  # Damping

        rolling_samples = np.zeros(total_samples)
        for i in range(n_samples):
            # Create IIR filter that approximates the spectral envelope
            # Using a bandpass filter with specific Q
            Q = p[i] / (2 * d[i]) if d[i] > 0 else 10.0
            nyquist = sample_rate / 2
            center_normalized = p[i] / nyquist
        
            # Design resonant filter
            b, a = signal.iirpeak(center_normalized, Q)
        
            # Apply filter to noise
            rolling_signal = signal.lfilter(b, a, noise)
        
            # Amplitude modulation
#            amplitude = normal_forces[i] * tangential_velocities[i] * 0.0001
            amplitude = 1 + angular_velocities[i] / np.max(np.abs(angular_velocities))
            rolling_signal *= amplitude
            idx = int(sample_idx + i)
            if not rolling_samples.shape[0] <= idx:
                rolling_samples[idx] = rolling_signal[i]

        # Normalize signal
        rolling_samples = rolling_samples / np.max(np.abs(rolling_samples))
        
        # Create tracks
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': rolling_samples,
            'non_collision': np.zeros(total_samples)
        }
        
        return result

    def _create_empty_tracks(self, total_samples: int) -> Dict:
        """Create empty tracks for silent sections."""
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples)
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
#            npz_file = f"{self.output_dir}/{config_obj.name}_{track_name}.npz"
#            np.savez_compressed(npz_file, track_data)
            track_file = f"{config_obj.name}_{track_name}.raw"
            wave_file = f"{self.output_dir}/{track_file}"
            sf.write(wave_file, track_data, sample_rate, subtype='DOUBLE')
            project_data['tracks'].append({
                'name': track_name,
                'file': track_file,
                'channels': 1,
                'position': 0.0,
                'volume': 1.0,
                'pan': 0.0
            })
            print(f"Saved {track_name} tracks to {self.output_dir}")

        # Save project file
        json_file = f"{self.output_dir}/{config_obj.name}.json"
        with open(json_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        print(f"Created multitrack project: {json_file}")
