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

        self.sample_rate = config.system.sample_rate
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
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
                    ir = self.ir_table.ir_table[size_idx, type_idx, band_idx]
                    # Pad to FFT size
                    padded = np.zeros(self.fft_size)
                    ir_len = min(len(ir), self.fft_size)
                    padded[:ir_len] = ir[:ir_len]
                    self._ir_ffts[size_idx, type_idx, band_idx] = np.fft.rfft(padded)
    
    def compute(self, obj_idx: int) -> None:
        """
        Compute proxy synth for an object.
        
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
        
        # Get force data for this object
        forces = self.entity_manager.get('forces')
        force_data = None
        for f_idx in forces.keys():
            if forces[f_idx].obj_idx == obj_idx:
                force_data = forces[f_idx]
                break
        
        if force_data is None:
            return
        
        # Get trajectories for velocity data
        trajectories = self.entity_manager.get('trajectories')
        trajectory = None
        for t_idx in trajectories.keys():
            if trajectories[t_idx].obj_idx == obj_idx:
                trajectory = trajectories[t_idx]
                break
        
        if trajectory is None:
            return
        
        # Process audio
        audio = self._process_object_audio(config_obj, force_data, trajectory, size_scale)
        
        # Save output
        self._save_audio(config_obj, audio)
    
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
    
    def _process_object_audio(self, config_obj: Any, force_data: Any, trajectory: Any, size_scale: float) -> np.ndarray:
        """
        Process all audio for an object.
        
        Uses SIMD-optimized operations throughout.
        """
        # Get total duration
        frames = force_data.frames
        total_samples = int(force_data.frames[-1])
        
        # Initialize output tracks
        impact_track = np.zeros(total_samples, dtype=np.float32)
        sliding_track = np.zeros(total_samples, dtype=np.float32)
        scraping_track = np.zeros(total_samples, dtype=np.float32)
        rolling_track = np.zeros(total_samples, dtype=np.float32)
        
        # Process each frame
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]
            sample_idx = int(frame)
            
            if sample_idx >= total_samples:
                break
            
            # Get contact type
            contact_type = force_data.get_contact_type(frame)
            
            # Get force magnitude
            force_mag = force_data.get_normal_force_magnitude(frame)
            
            # Get velocity
            velocity = trajectory.get_velocity(frame)
            velocity_mag = np.linalg.norm(velocity)
            
            # Process based on contact type
            if contact_type == 1:  # Impact
                # Get impact duration
                impact_duration = force_data.get_impact_duration(frame)
                
                # Process impact
                if np.isnan(impact_duration):
                    debug_print('ERROR: impact_duration is nan, at frame:', frame, 'for', config_obj.idx, config_obj.name)
                impact_audio = self._process_impact(size_scale, force_mag, impact_duration)
                
                # Add to track
                start = sample_idx
                end = min(start + len(impact_audio), total_samples)
                if end > start:
                    impact_track[start:end] += impact_audio[:end - start]
            
            elif contact_type in [2, 3, 4]:  # Scraping, Sliding, Rolling
                # Get contact duration
                contact_duration = 1.0 / self.sample_rate  # 1 sample
                
                # Process continuous contact
                contact_audio = self._process_continuous(size_scale, contact_type, force_mag, velocity_mag, contact_duration)
                
                # Add to appropriate track
                start = sample_idx
                end = min(start + len(contact_audio), total_samples)
                if end > start:
                    if contact_type == 2:
                        scraping_track[start:end] += contact_audio[:end - start]
                    elif contact_type == 3:
                        sliding_track[start:end] += contact_audio[:end - start]
                    else:
                        rolling_track[start:end] += contact_audio[:end - start]
        
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
        
        return mixed
    
    def _process_impact(self, size_scale: float, force: float, duration: float) -> np.ndarray:
        """
        Process impact event using SIMD-optimized convolution.
        """
        # Get IR for impact
        ir = self.ir_table.get_ir(size_scale, 0)
        
        # Generate impact excitation
        n_samples = int(duration * self.sample_rate)
        if n_samples_samples <= 0:
            n_samples = int(0.01 * self.sample_rate)  # Minimum 10ms
        
        t = np.arange(n_samples) / self.sample_rate
        
        # Asymmetric impact envelope (Hertzian-like)
        rise_time = duration * 0.3
        decay_time = duration * 0.7
        
        envelope = np.zeros(n_samples)
        rise_samples = int(rise_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        
        # Rise phase (sinusoidal)
        if rise_samples > 0:
            rise_env = np.sin(np.linspace(0, np.pi/2, rise_samples))**2
            envelope[:rise_samples] = rise_env
        
        # Decay phase (exponential)
        if decay_samples > 0:
            decay_env = np.exp(-np.linspace(0, 5, decay_samples))
            envelope[rise_samples:rise_samples + decay_samples] = decay_env
        
        # Scale by force
        excitation = force * envelope
        
        # Convolve with IR using FFT
        output = self._fft_convolve(excitation, ir)
        
        return output
    
    def _process_continuous(self, size_scale: float, contact_type: int, force: float, velocity: float, duration: float) -> np.ndarray:
        """
        Process continuous contact using SIMD-optimized operations.
        """
        # Get IR for contact type
        ir = self.ir_table.get_ir(size_scale, contact_type)
        
        # Generate excitation
        n_samples = int(duration * self.sample_rate)
        if n_samples <= 0:
            n_samples = 1
        
        if contact_type == 2:  # Scraping
            excitation = self._generate_scraping_excitation(n_samples, force, velocity)
        elif contact_type == 3:  # Sliding
            excitation = self._generate_sliding_excitation(n_samples, force, velocity)
        else:  # Rolling
            excitation = self._generate_rolling_excitation(n_samples, force, velocity, size_scale)
        
        # Convolve with IR
        output = self._fft_convolve(excitation, ir)
        
        return output
    
    def _generate_sliding_excitation(self, n_samples: int, force: float, velocity: float) -> np.ndarray:
        """Generate sliding excitation using SIMD operations."""
        # White noise
        noise = np.random.randn(n_samples)
        
        # Amplitude modulation
        amplitude = np.sqrt(np.abs(force) * np.abs(velocity))
        
        # Frequency modulation based on velocity
        base_freq = 500 + 2000 * np.abs(velocity) / 10.0
        base_freq = np.clip(base_freq, 100, 5000)
        
        # Generate modulated noise
        t = np.arange(n_samples) / self.sample_rate
        phase = 2 * np.pi * base_freq * t
        
        excitation = noise * amplitude * np.sin(phase)
        
        return excitation
    
    def _generate_scraping_excitation(self, n_samples: int, force: float, velocity: float) -> np.ndarray:
        """Generate scraping excitation using SIMD operations."""
        # Bandpass noise with higher frequency content
        noise = np.random.randn(n_samples)
        
        # Apply simple highpass filter (first difference)
        noise = np.diff(noise, prepend=0)
        
        # Amplitude modulation
        amplitude = np.abs(force) * np.abs(velocity)
        
        # Add transient spikes
        n_spikes = max(1, int(n_samples / (self.sample_rate * 0.05)))
        spike_positions = np.random.choice(n_samples, min(n_spikes, n_samples), replace=False)
        
        excitation = noise * amplitude
        for pos in spike_positions:
            spike_len = min(50, n_samples - pos)
            if spike_len > 0:
                spike = np.exp(-np.arange(spike_len) / 10) * np.random.randn()
                excitation[pos:pos + spike_len] += spike * amplitude
        
        return excitation
    
    def _generate_rolling_excitation(self, n_samples: int, force: float, velocity: float, size_scale: float) -> np.ndarray:
        """Generate rolling excitation using SIMD operations."""
        # Pulse rate based on size and velocity
        base_rate = 5.0 + 20.0 * (1 - size_scale)
        pulse_rate = base_rate * np.abs(velocity) / 10.0
        
        # Generate pulse train
        t = np.arange(n_samples) / self.sample_rate

        pulse_phase = np.cumsum(np.full(n_samples, pulse_rate)) / self.sample_rate
        
        # Gaussian pulses
        pulse_width = 0.005
        excitation = np.exp(-((np.mod(pulse_phase, 11.0) - 0.5) / pulse_width)**2)
        
        # Modulate by force
        excitation *= np.abs(force)
        
        # Add some noise
        excitation += 0.1 * np.random.randn(n_samples) * np.abs(force)
        
        return excitation
    
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
            ir_fft = np.fft.rfft(padded_ir)
            
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

