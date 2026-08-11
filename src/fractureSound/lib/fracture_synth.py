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
Fracture sound synthesis implementation.

Based on the paper:
"Fracture Sound: A physically based approach to the synthesis of fracture sounds"
by K. van den Doel, P.G. Kry, and D.K. Pai
"""

import os
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import soundfile as sf
from scipy import signal
from scipy.signal import windows

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _parse_lib
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from .fracture_data import FractureEvent, FractureType, FragmentData

@dataclass
class FractureSynth:
    """
    Synthesize fracture sounds based on the FractureSound paper.
    
    Implements:
    - SHATTER: Poisson-like burst of micro-impulses
    - CRACK: Sparse, irregular sequence of discrete events
    - SNAP: Single event with two components (nucleation + ringdown)
    """
    
    entity_manager: EntityManager
    
    # Synthesis parameters
    sample_rate: int = 48000
    fragment_modal_path: str = None
    fracture_audio_dir: str = None
    
    # Grain parameters for shatter
    grain_density: float = 1000.0  # Grains per second for shatter
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        self.sample_rate = config.system.sample_rate
        self.fragment_modal_path = f"{config.system.cache_path}/fracture_modal"
        self.fracture_audio_dir = f"{config.system.cache_path}/fracture_audio"
        os.makedirs(self.fracture_audio_dir, exist_ok=True)
        
        # Frequency bands for crack sound
        self.freq_bands = self._create_frequency_bands()
    
    def _create_frequency_bands(self) -> List[Tuple[float, float]]:
        """Create logarithmic frequency bands for crack synthesis."""
        bands = []
        freqs = np.logspace(np.log10(50), np.log10(20000), 12)
        for i in range(len(freqs) - 1):
            bands.append((freqs[i], freqs[i+1]))
        return bands
    
    def compute(self, event: FractureEvent) -> Dict[str, np.ndarray]:
        """
        Compute fracture sound for a fracture event.
        
        Parameters:
        -----------
        event : FractureEvent
            The fracture event to synthesize
            
        Returns:
        --------
        Dict with audio tracks for each component
        """
        # Get fragment objects
        fragment_objs = []
        config = self.entity_manager.get('config')
        for idx in event.fragment_indices:
            for obj in config.objects:
                if obj.idx == idx:
                    fragment_objs.append(obj)
                    break
        
        if len(fragment_objs) == 0:
            debug_print("No fragments found for fracture synthesis")
            return {}
        
        # Synthesize based on fracture type
        if event.fracture_type == FractureType.SHATTER:
            audio = self._synthesize_shatter(event, fragment_objs)
        elif event.fracture_type == FractureType.CRACK:
            audio = self._synthesize_crack(event, fragment_objs)
        else:  # SNAP
            audio = self._synthesize_snap(event, fragment_objs)
        
        # Save audio
        if audio is not None:
            self._save_fracture_audio(event, audio)
        
        return audio
    
    def _synthesize_shatter(self, event: FractureEvent, fragment_objs: List[Any]) -> Dict[str, np.ndarray]:
        """
        Synthesize shatter fracture sound.
        
        Characteristics:
        - Poisson-like burst of micro-impulses
        - Extremely fast attack (<1ms)
        - Broadband with high-frequency emphasis
        - Hundreds of grains within first 10ms
        """
        debug_print(f"Synthesizing shatter fracture for {event.original_obj_name}")
        
        n_samples = int(0.5 * self.sample_rate)  # 500ms duration
        output = np.zeros(n_samples)
        
        # Get material properties
        damping = event.damping
        young_modulus = event.young_modulus
        density = event.density
        
        # 1. Generate dense impulse train (grains)
        grain_times = self._generate_poisson_grains(
            density=self.grain_density,
            duration=0.01,  # 10ms burst
            sample_rate=self.sample_rate,
            power_law=1.5  # Power-law distribution for inter-onset times
        )
        
        # 2. Process each grain
        for grain_time in grain_times:
            if grain_time >= n_samples:
                break
            
            # Each grain: short excitation pulse → modal resonator
            # Scale fragment size for pitch variation
            fragment_idx = np.random.randint(0, len(fragment_objs))
            fragment_obj = fragment_objs[fragment_idx]
            
            # Get modal model for this fragment
            modal_data = self._get_fragment_modal(event, fragment_obj)
            if modal_data is None:
                continue
            
            # Grain amplitude (decaying envelope)
            amplitude = np.exp(-grain_time / (0.005 * self.sample_rate))
            
            # Generate grain signal
            grain_len = int(0.005 * self.sample_rate)  # 5ms grain
            grain_signal = self._generate_grain(
                modal_data=modal_data,
                duration=grain_len,
                amplitude=amplitude,
                damping=damping,
                fragment_size=self._get_fragment_size(fragment_obj, event.frame)
            )
            
            # Add to output with random phase
            start = int(grain_time)
            end = min(start + grain_len, n_samples)
            if end > start:
                output[start:end] += grain_signal[:end-start] * np.random.uniform(0.5, 1.5)
        
        # 3. Add stochastic branch crackling
        crackle = self._generate_crackle_noise(
            n_samples=n_samples,
            density=0.5,  # Crackle density
            amplitude=0.3,
            damping=damping
        )
        output += crackle
        
        # 4. Apply envelope
        envelope = self._shatter_envelope(n_samples)
        output *= envelope
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9
        
        return {'shatter': output}
    
    def _synthesize_crack(self, event: FractureEvent, 
                          fragment_objs: List[Any]) -> Dict[str, np.ndarray]:
        """
        Synthesize crack fracture sound.
        
        Characteristics:
        - Sparse, irregular sequence of discrete events
        - Each "pop" is a crack increment
        - Inter-event times follow power-law or log-normal distributions
        - Individual events have sharp attacks (1-3ms)
        """
        debug_print(f"Synthesizing crack fracture for {event.original_obj_name}")
        
        n_samples = int(0.3 * self.sample_rate)  # 300ms duration
        output = np.zeros(n_samples)
        
        # Get material properties
        damping = event.damping
        young_modulus = event.young_modulus
        
        # 1. Generate sparse impulse sequence
        # Crack velocity and heterogeneity control inter-event statistics
        crack_velocity = event.crack_velocity
        crack_length = event.crack_length
        
        # Number of crack events
        n_events = int(10 + 20 * np.random.random())
        
        # Generate event times with log-normal distribution
        event_times = self._generate_log_normal_events(
            n_events=n_events,
            duration=0.3,
            sample_rate=self.sample_rate,
            shape=0.8,  # Log-normal shape parameter
            scale=0.01  # Time scale
        )
        
        # 2. Process each crack event
        for i, event_time in enumerate(event_times):
            if event_time >= n_samples:
                break
            
            # Instantaneous crack length increases with time
            crack_progress = event_time / (0.3 * self.sample_rate)
            current_length = crack_length * (0.1 + 0.9 * crack_progress)
            
            # Frequency tracks instantaneous crack length: f ∝ 1/L
            # For a vibrating flap: f ∝ sqrt(EI/ρL⁴)
            base_freq = 5000.0 / (current_length + 0.01)
            base_freq = np.clip(base_freq, 100, 15000)
            
            # Amplitude depends on crack velocity and energy
            velocity_scale = crack_velocity / 500.0  # Normalize to typical crack velocity
            amplitude = 0.5 * velocity_scale * (0.5 + 0.5 * np.random.random())
            
            # Generate crack event signal
            event_signal = self._generate_crack_event(
                base_freq=base_freq,
                duration=0.02,  # 20ms per event
                amplitude=amplitude,
                damping=damping,
                young_modulus=young_modulus
            )
            
            # Add to output
            start = int(event_time)
            end = min(start + len(event_signal), n_samples)
            if end > start:
                output[start:end] += event_signal[:end-start]
        
        # 3. Add subtle background noise
        background = self._generate_crackle_noise(
            n_samples=n_samples,
            density=0.1,
            amplitude=0.05,
            damping=damping
        )
        output += background
        
        # Apply envelope
        envelope = self._crack_envelope(n_samples)
        output *= envelope
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9
        
        return {'crack': output}
    
    def _synthesize_snap(self, event: FractureEvent, fragment_objs: List[Any]) -> Dict[str, np.ndarray]:
        """
        Synthesize snap fracture sound.
        
        Characteristics:
        - Single event with two components:
          1. Initial crack nucleation "tick"
          2. Ringing of separated pieces
        - Extremely sharp transient (<1ms)
        - Strongly tonal ringdown from fragment modes
        - Excitation is an unloading step function (1/f spectral slope)
        """
        debug_print(f"Synthesizing snap fracture for {event.original_obj_name}")
        
        n_samples = int(1.0 * self.sample_rate)  # 1 second duration
        output = np.zeros(n_samples)
        
        # Get material properties
        damping = event.damping
        young_modulus = event.young_modulus
        density = event.density
        fracture_energy = event.fracture_energy
        
        # 1. Crack nucleation "tick" (high-frequency burst)
        tick = self._generate_nucleation_tick(
            amplitude=0.3,
            duration=0.001,  # 1ms
            damping=damping
        )
        output[:len(tick)] += tick
        
        # 2. Process each fragment for ringdown
        for fragment_obj in fragment_objs:
            # Get fragment modal model
            modal_data = self._get_fragment_modal(event, fragment_obj)
            if modal_data is None:
                continue
            
            # Fragment size affects pitch
            fragment_size = self._get_fragment_size(fragment_obj, event.frame)
            
            # Generate fragment ringdown
            # Excitation is a step function (unloading)
            ringdown = self._generate_fragment_ringdown(
                modal_data=modal_data,
                duration=1.0,
                fracture_energy=fracture_energy,
                fragment_size=fragment_size,
                damping=damping
            )
            
            # Add to output with slight delay for each fragment
            delay = 0.001 + 0.002 * np.random.random()  # 1-3ms delay
            delay_samples = int(delay * self.sample_rate)
            end = min(delay_samples + len(ringdown), n_samples)
            output[delay_samples:end] += ringdown[:end-delay_samples]
        
        # 3. Apply envelope
        envelope = self._snap_envelope(n_samples)
        output *= envelope
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9
        
        return {'snap': output}
    
    def _generate_poisson_grains(self, density: float, duration: float, sample_rate: int, power_law: float = 1.0) -> np.ndarray:
        """
        Generate grain times with Poisson-like distribution.
        
        Inter-onset times follow a power-law distribution for shatter.
        """
        n_grains = int(density * duration)
        if n_grains == 0:
            return np.array([])
        
        # Generate inter-onset times with power-law distribution
        u = np.random.random(n_grains)
        # Power-law: t ∝ u^(1/(1-α)) where α is the power-law exponent
        inter_times = np.power(u, 1/(1 - power_law)) * duration / n_grains
        
        # Cumulative sum to get event times
        grain_times = np.cumsum(inter_times)
        
        # Convert to samples
        grain_samples = (grain_times * sample_rate).astype(int)
        
        # Filter to within duration
        max_samples = int(duration * sample_rate)
        grain_samples = grain_samples[grain_samples < max_samples]
        
        return grain_samples
    
    def _generate_log_normal_events(self, n_events: int, duration: float, sample_rate: int, shape: float, scale: float) -> np.ndarray:
        """
        Generate event times with log-normal distribution for crack.
        """
        if n_events == 0:
            return np.array([])
        
        # Generate inter-event times with log-normal distribution
        inter_times = np.random.lognormal(mean=0, sigma=shape, size=n_events) * scale
        
        # Ensure minimum spacing
        min_spacing = 0.001  # 1ms minimum
        inter_times = np.maximum(inter_times, min_spacing)
        
        # Cumulative sum to get event times
        event_times = np.cumsum(inter_times)
        
        # Convert to samples
        event_samples = (event_times * sample_rate).astype(int)
        
        # Filter to within duration
        max_samples = int(duration * sample_rate)
        event_samples = event_samples[event_samples < max_samples]
        
        return event_samples
    
    def _generate_grain(self, modal_data: Dict, duration: int, amplitude: float, damping: float, fragment_size: float) -> np.ndarray:
        """
        Generate a single grain for shatter synthesis.
        """
        # Short excitation pulse
        pulse = np.zeros(duration)
        pulse_len = min(int(0.0005 * self.sample_rate), duration)  # 0.5ms
        pulse[:pulse_len] = windows.tukey(pulse_len, alpha=0.5)
        
        # Apply modal filter
        frequencies = modal_data['frequencies']
        gains = modal_data['gains'][0]  # Use first vertex
        t60s = modal_data['t60s']
        
        # Scale frequencies by fragment size
        if fragment_size > 0:
            scale = 0.1 / fragment_size
            frequencies = frequencies * scale
        
        # Generate modal response
        grain = self._apply_modal_filter(
            pulse, frequencies, gains, t60s, self.sample_rate, damping
        )
        
        # Apply amplitude
        grain *= amplitude
        
        return grain
    
    def _generate_crack_event(self, base_freq: float, duration: float, amplitude: float, damping: float, young_modulus: float) -> np.ndarray:
        """
        Generate a single crack event.
        """
        n_samples = int(duration * self.sample_rate)
        
        # Impulse excitation with sharp attack
        impulse = np.zeros(n_samples)
        attack_len = int(0.001 * self.sample_rate)  # 1ms attack
        if attack_len < n_samples:
            impulse[:attack_len] = windows.tukey(attack_len, alpha=0.3)
        
        # Resonant filter centered at base_freq
        # Q factor depends on damping
        Q = 10.0 / (damping + 0.01)
        Q = np.clip(Q, 1, 100)
        
        # Design resonant filter
        b, a = signal.iirpeak(base_freq / (self.sample_rate / 2), Q, fs=self.sample_rate)
        
        # Apply filter
        filtered = signal.lfilter(b, a, impulse)
        
        # Apply envelope
        envelope = np.exp(-np.arange(n_samples) / (0.01 * self.sample_rate))
        filtered *= envelope
        
        # Scale
        filtered *= amplitude
        
        return filtered
    
    def _generate_fragment_ringdown(self, modal_data: Dict, duration: float, fracture_energy: float, fragment_size: float, damping: float) -> np.ndarray:
        """
        Generate fragment ringdown for snap fracture.
        
        Excitation is a step function (unloading), not a click.
        """
        n_samples = int(duration * self.sample_rate)
        output = np.zeros(n_samples)
        
        # Step excitation (unloading)
        step_excitation = np.ones(n_samples)
        # Smooth rise time (0.1-0.5ms)
        rise_time = int(0.0003 * self.sample_rate)
        step_excitation[:rise_time] = np.linspace(0, 1, rise_time)
        
        # Get modal parameters
        frequencies = modal_data['frequencies']
        n_freqs = len(frequencies)
        
        # Use multiple vertices for richer sound
        n_vertices = len(modal_data['gains'])
        vertex_indices = np.random.choice(n_vertices, min(3, n_vertices), replace=False)
        
        # Apply modal filters
        for vi in vertex_indices:
            gains = modal_data['gains'][vi]
            t60s = modal_data['t60s']
            
            # Scale frequencies by fragment size
            if fragment_size > 0:
                scale = 0.1 / fragment_size
                scaled_freqs = frequencies * scale
            else:
                scaled_freqs = frequencies
            
            # Generate modal response
            modal_response = self._apply_modal_filter(
                step_excitation, scaled_freqs, gains, t60s, self.sample_rate, damping
            )
            output += modal_response
        
        # Scale by fracture energy
        energy_scale = np.sqrt(fracture_energy / 0.01)  # Normalize to 0.01J
        output *= energy_scale
        
        return output
    
    def _generate_nucleation_tick(self, amplitude: float, duration: float, damping: float) -> np.ndarray:
        """
        Generate high-frequency nucleation tick for snap.
        """
        n_samples = int(duration * self.sample_rate)
        
        # High-frequency burst
        tick = np.random.randn(n_samples) * 0.1
        
        # Add tonal component
        t = np.arange(n_samples) / self.sample_rate
        freq = 5000 + 10000 * np.random.random()
        tone = np.sin(2 * np.pi * freq * t)
        tick += tone
        
        # Apply envelope
        envelope = windows.tukey(n_samples, alpha=0.3)
        tick *= envelope
        
        # Scale
        tick *= amplitude
        
        return tick
    
    def _generate_crackle_noise(self, n_samples: int, density: float, amplitude: float, damping: float) -> np.ndarray:
        """
        Generate crackling noise for shatter and crack.
        """
        # Number of crackle events
        n_events = int(density * n_samples / 1000)
        if n_events == 0:
            return np.zeros(n_samples)
        
        output = np.zeros(n_samples)
        
        # Generate events
        event_positions = np.random.choice(n_samples, n_events, replace=False)
        
        for pos in event_positions:
            # Short burst
            burst_len = int(0.0005 * self.sample_rate)  # 0.5ms
            burst = np.random.randn(burst_len)
            
            # Apply high-pass filter (crackle is high-frequency)
            b, a = signal.butter(4, 2000 / (self.sample_rate / 2), btype='high')
            burst = signal.lfilter(b, a, burst)
            
            # Add to output
            end = min(pos + burst_len, n_samples)
            output[pos:end] += burst[:end-pos] * 0.1
        
        # Scale
        output *= amplitude
        
        return output
    
    def _apply_modal_filter(self, excitation: np.ndarray, frequencies: np.ndarray, gains: np.ndarray, t60s: np.ndarray, sample_rate: int, damping: float) -> np.ndarray:
        """
        Apply modal filter to excitation signal.
        """
        n_modes = len(frequencies)
        if n_modes == 0:
            return np.zeros_like(excitation)
        
        output = np.zeros_like(excitation, dtype=np.float32)
        
        # Process each mode
        for i in range(n_modes):
            if i >= len(gains) or i >= len(t60s):
                continue
            
            freq = frequencies[i]
            gain = gains[i]
            t60 = t60s[i]
            
            if freq <= 0 or gain == 0:
                continue
            
            # Design resonant filter
            Q = 10.0 / (damping + 0.01)
            Q = np.clip(Q, 1, 50)
            
            # Apply filter
            try:
                b, a = signal.iirpeak(freq / (sample_rate / 2), Q, fs=sample_rate)
                filtered = signal.lfilter(b, a, excitation)
                
                # Scale by gain and T60
                decay_scale = np.exp(-np.arange(len(excitation)) / (t60 * sample_rate / 3))
                filtered *= gain * decay_scale
                
                output += filtered
            except:
                continue
        
        return output
    
    def _get_fragment_modal(self, event: FractureEvent, fragment_obj: Any) -> Optional[Dict]:
        """
        Get modal model for a fragment.
        """
        # Try to load fracture modal
        lib_file = f"{self.fragment_modal_path}/{fragment_obj.name}_fracture.lib"
        
        if os.path.exists(lib_file):
            return _parse_lib(lib_file)
        
        # Fallback to original modal
        original_lib = f"{self.fragment_modal_path}/{fragment_obj.name}.lib"
        if os.path.exists(original_lib):
            return _parse_lib(original_lib)
        
        return None
    
    def _get_fragment_size(self, fragment_obj: Any, frame: float) -> float:
        """Get characteristic size of a fragment."""
        try:
            from pbrAudioCommon import _load_mesh
            vertices, _, _ = _load_mesh(fragment_obj, int(frame))
            if len(vertices) > 0:
                center = np.mean(vertices, axis=0)
                distances = np.linalg.norm(vertices - center, axis=1)
                return np.max(distances)
        except:
            pass
        return 0.05  # Default
    
    def _shatter_envelope(self, n_samples: int) -> np.ndarray:
        """Envelope for shatter fracture."""
        t = np.arange(n_samples) / self.sample_rate
        # Fast attack, chaotic decay
        attack = 1 - np.exp(-t / 0.001)
        decay = np.exp(-t / 0.05)
        return attack * decay
    
    def _crack_envelope(self, n_samples: int) -> np.ndarray:
        """Envelope for crack fracture."""
        t = np.arange(n_samples) / self.sample_rate
        # Sharp attack, moderate decay
        attack = 1 - np.exp(-t / 0.002)
        decay = np.exp(-t / 0.08)
        return attack * decay
    
    def _snap_envelope(self, n_samples: int) -> np.ndarray:
        """Envelope for snap fracture."""
        t = np.arange(n_samples) / self.sample_rate
        # Very sharp attack, long ringdown
        attack = 1 - np.exp(-t / 0.0005)
        decay = np.exp(-t / 0.3)
        return attack * decay
    
    def _save_fracture_audio(self, event: FractureEvent, audio: Dict[str, np.ndarray]):
        """
        Save fracture audio to files.
        """
        base_name = f"fracture_{event.original_obj_name}_{event.frame:.3f}"
        
        for name, data in audio.items():
            if len(data) == 0:
                continue
            
            # Normalize
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val * 0.9
            
            # Save
            filename = f"{self.fracture_audio_dir}/{base_name}_{name}.wav"
            sf.write(filename, data, self.sample_rate, subtype='FLOAT')
            debug_print(f"Saved {name} fracture audio to {filename}")
        
        # Save metadata
        import json
        metadata = {
            'event': {
                'type': event.fracture_type.value,
                'frame': float(event.frame),
                'original_obj': event.original_obj_name,
                'fragments': event.fragment_indices,
                'energy': float(event.fracture_energy)
            },
            'audio': list(audio.keys()),
            'sample_rate': self.sample_rate
        }
        
        metadata_file = f"{self.fracture_audio_dir}/{base_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        debug_print(f"Saved fracture metadata to {metadata_file}")
