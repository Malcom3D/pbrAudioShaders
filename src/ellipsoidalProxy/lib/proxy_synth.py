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

# pbrAudioShaders/src/ellipsoidalProxy/lib/proxy_synth.py

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from .proxy_ir_table import ProxyIRTable
from .proxy_eq import ProxyEqualizer


@dataclass
class ProxySynth:
    """
    Lightweight physically-based synthesizer for proxy meshes.
    
    Uses Toeplitz IR matrix from ProxyIRTable where each row corresponds to
    a modal frequency band. The excitation signal is split into frequency bands
    and convolved with the corresponding IR row.
    """
    
    entity_manager: EntityManager
    
    # Components
    ir_table: ProxyIRTable = None
    equalizer: ProxyEqualizer = None
    
    # Processing parameters
    sample_rate: int = 48000
    
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
        self.sfps = (fps / fps_base) * subframes  # subframes per second
        
        # Initialize components
        if self.ir_table is None:
            self.ir_table = ProxyIRTable(self.entity_manager)
        
        if self.equalizer is None:
            self.equalizer = ProxyEqualizer(self.entity_manager)
        
        # Set output directory
        if self.output_dir is None:
            self.output_dir = f"{config.system.cache_path}/proxy_audio"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Path to audio-force tracks
        self.audio_force_dir = f"{config.system.cache_path}/audio_force"
        
        # Cache for material_key lookups
        self._material_key_cache = {}
        
        # Cache for band filters
        self._band_filters = {}
    
    def _get_material_key(self, config_obj: Any) -> int:
        """
        Get material key for an object.
        
        The material key is used to look up the correct IR table.
        """
        if config_obj.idx in self._material_key_cache:
            return self._material_key_cache[config_obj.idx]
        
        # Create material key from acoustic properties
        if config_obj.acoustic_shader:
            material_key = (
                config_obj.acoustic_shader.young_modulus,
                config_obj.acoustic_shader.poisson_ratio,
                config_obj.acoustic_shader.density,
                config_obj.acoustic_shader.damping
            )
        else:
            material_key = (None, None, None, None)
        
        # Use a hashable key for the material
        material_key = hash(material_key)
        self._material_key_cache[config_obj.idx] = material_key
        
        return material_key
    
    def _get_num_modes(self, config_obj: Any) -> int:
        """Get the number of modes for this object."""
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        return self.ir_table.get_num_modes(material_key, proxy_type)
    
    def _get_ir_matrix(self, config_obj: Any, size_scale: float) -> np.ndarray:
        """
        Get interpolated IR matrix for an object.
        
        Returns a Toeplitz matrix of shape (num_modes, max_ir_length).
        """
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        
        return self.ir_table.get_ir(material_key, proxy_type, size_scale)
    
    def _get_band_filters(self, num_modes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create bandpass filters for splitting the excitation signal into
        frequency bands corresponding to each mode.
        
        Returns:
            List of (b, a) filter coefficients for each band
        """
        cache_key = num_modes
        if cache_key in self._band_filters:
            return self._band_filters[cache_key]
        
        # Define frequency bands based on modal distribution
        # Use logarithmic spacing from 20Hz to Nyquist
        nyquist = self.sample_rate / 2
        min_freq = 20.0
        max_freq = nyquist * 0.95
        
        # Create logarithmically spaced band edges
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_modes + 1)
        
        filters = []
        for i in range(num_modes):
            low = band_edges[i]
            high = band_edges[i + 1]
            
            # Normalize to Nyquist
            low_norm = low / nyquist
            high_norm = high / nyquist
            
            # Design bandpass filter (4th order Butterworth)
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filters.append((b, a))
        
        self._band_filters[cache_key] = filters
        return filters
    
    def _split_into_bands(self, signal: np.ndarray, num_modes: int) -> List[np.ndarray]:
        """
        Split an excitation signal into frequency bands corresponding to each mode.
        
        Returns:
            List of band signals, each of shape (n_samples,)
        """
        if num_modes <= 0:
            return [signal]
        
        filters = self._get_band_filters(num_modes)
        band_signals = []
        
        for b, a in filters:
            # Apply bandpass filter
            band_signal = filtfilt(b, a, signal)
            band_signals.append(band_signal)
        
        return band_signals
    
    def _convolve_with_ir_matrix(self, band_signals: List[np.ndarray], ir_matrix: np.ndarray) -> np.ndarray:
        """
        Convolve each band signal with the corresponding IR row from the Toeplitz matrix.
        
        Parameters:
        -----------
        band_signals : List[np.ndarray]
            List of band signals (one per mode)
        ir_matrix : np.ndarray
            Toeplitz matrix of shape (num_modes, max_ir_length)
        
        Returns:
        --------
        np.ndarray : Sum of convolved signals
        """
        num_modes = min(len(band_signals), ir_matrix.shape[0])
        output = np.zeros_like(band_signals[0], dtype=np.float64)
        
        for mode_idx in range(num_modes):
            # Get IR for this mode (row of the Toeplitz matrix)
            ir = ir_matrix[mode_idx]
            
            # Get band signal
            band_signal = band_signals[mode_idx]
            
            # Skip if signal is zero or IR is all zeros
            if np.all(band_signal == 0) or np.all(ir == 0):
                continue
            
            # FFT-based convolution with the IR
            # Use overlap-add for efficiency
            convolved = self._fft_convolve(band_signal, ir)
            
            # Add to output
            output += convolved
        
        return output.astype(np.float32)

    def _fft_convolve(self, signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        FFT-based convolution with a single IR.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        ir : np.ndarray
            Impulse response
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        n_signal = len(signal)
        n_ir = len(ir)
        
        # Determine FFT size (next power of 2)
        fft_size = 1 << (n_signal + n_ir - 1).bit_length()
        
        # FFT of signal and IR
        signal_fft = np.fft.rfft(signal, n=fft_size)
        ir_fft = np.fft.rfft(ir, n=fft_size)
        
        # Multiply and inverse FFT
        result = np.fft.irfft(signal_fft * ir_fft, n=fft_size)
        
        # Trim to expected length
        result = result[:n_signal]
        
        return result

    def compute(self, obj_idx: int, total_samples: int) -> None:
        """
        Compute proxy synth for an object using audio-force tracks.
        
        The excitation signal is split into frequency bands and convolved
        with the corresponding IR rows from the Toeplitz matrix.
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
        debug_print(f'Size scale for {config_obj.name}: {size_scale:.3f}')
        
        # Get IR matrix (Toeplitz matrix)
        ir_matrix = self._get_ir_matrix(config_obj, size_scale)
        num_modes = ir_matrix.shape[0]
        debug_print(f'IR matrix shape: {ir_matrix.shape} for {config_obj.name}')
        
        # Load audio-force tracks
        audio_tracks = self._load_audio_force_tracks(config_obj.name)
        
        if audio_tracks is None:
            debug_print(f"No audio-force tracks found for {config_obj.name}")
            return
        
        # Process each track
        processed_tracks = {}
        
        for track_name, excitation in audio_tracks.items():
            if excitation is None:
                continue
            
            debug_print(f'Processing {track_name} for {config_obj.name}: {excitation.shape}, non-zero: {np.count_nonzero(excitation)}')
            
            # Trim or pad to total_samples
            if excitation.shape[0] < total_samples:
                excitation = np.pad(excitation, (0, total_samples - len(excitation)))
            elif excitation.shape[0] > total_samples:
                excitation = excitation[:total_samples]
            
            # Split excitation into frequency bands
            band_signals = self._split_into_bands(excitation, num_modes)
            
            # Convolve each band with the corresponding IR
            if track_name.endswith('_sound'):
                # Sound tracks are already processed (no IR convolution)
                processed = excitation
            else:
                processed = self._convolve_with_ir_matrix(band_signals, ir_matrix)
                debug_print(f'After IR convolution for {config_obj.name} - {track_name}: {processed.shape}, non-zero: {np.count_nonzero(processed)}')
                
                # Apply equalization
                contact_type_map = {
                    'impact': 0,
                    'sliding': 1,
                    'scraping': 2,
                    'rolling': 3,
                }
                contact_type = contact_type_map.get(track_name, 0)
                processed = self.equalizer.apply_equalization(processed, contact_type, excitation)
                debug_print(f'After equalization for {config_obj.name} - {track_name}: {processed.shape}, non-zero: {np.count_nonzero(processed)}')
            
            processed_tracks[track_name] = processed

        # Mix all tracks
        mixed = np.zeros(total_samples, dtype=np.float32)

        for track_name in processed_tracks.keys():
            track = processed_tracks[track_name]
            
            # Apply track-specific volume adjustments
            if track_name == 'impact':
                max_val = np.max(np.abs(track))
                if max_val > 0:
                    track = track / max_val * 0.9
            elif track_name == 'rolling':
                track *= 0.01
            elif track_name in ['sliding', 'scraping']:
                track *= 0.0075
            elif track_name == 'rolling_sound':
                track *= 2.5
            elif track_name == 'sliding_sound':
                track *= 1.0
            elif track_name == 'scraping_sound':
                track *= 5

            mixed += track
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed /= max_val * 0.9
        
        # Save output
        self._save_audio(config_obj, mixed, processed_tracks)
    
    def _load_audio_force_tracks(self, obj_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load audio-force tracks for an object.
        
        Returns:
            Dictionary with keys: 'impact', 'sliding', 'scraping', 'rolling', 
            'rolling_sound', 'sliding_sound', 'scraping_sound'
            or None if no tracks found.
        """
        tracks = {}
        track_names = ['impact', 'sliding', 'scraping', 'rolling', 'rolling_sound', 'sliding_sound', 'scraping_sound']
        
        for track_name in track_names:
            track_file = f"{self.audio_force_dir}/{obj_name}_{track_name}.raw"
            
            if os.path.exists(track_file):
                try:
                    track_data = np.fromfile(track_file, dtype=np.float32)
                    if len(track_data) > 0:
                        tracks[track_name] = track_data
                        debug_print(f"Loaded {track_name} track: {len(track_data)} samples")
                except Exception as e:
                    debug_print(f"Error loading {track_name} track: {e}")
        
        if not tracks:
            return None
        
        return tracks
    
    def _compute_size_scale(self, config_obj: Any) -> float:
        """Compute normalized size scale (0-1) for an object."""
        from pbrAudioCommon import _load_mesh
        try:
            vertices, _, _ = _load_mesh(config_obj, 0, use_proxy_path=True)
            if len(vertices) > 0:
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                size = np.linalg.norm(max_coords - min_coords)
                
                # Get min/max size for this material/shape
                material_key = self._get_material_key(config_obj)
                proxy_type = config_obj.proxy_type
                key = (material_key, proxy_type)
                
                min_size = self.ir_table.min_size.get(key, 0.0)
                max_size = self.ir_table.max_size.get(key, 1.0)
                
                # Normalize to 0-1 range
                size_range = max_size - min_size
                if size_range > 0:
                    size_scale = (size - min_size) / size_range
                else:
                    size_scale = 0.5
                
                return np.clip(size_scale, 0, 1)
        except Exception as e:
            debug_print(f"Error computing size scale: {e}")
        
        return 0.5  # Default
    
    def _save_audio(self, config_obj: Any, mixed: np.ndarray, tracks: Dict[str, np.ndarray]) -> None:
        """Save synthesized audio to files."""
        # Save mixed audio
        mixed_file = f"{self.output_dir}/{config_obj.name}_proxy_mixed.wav"
        sf.write(mixed_file, mixed, self.sample_rate, subtype='FLOAT')
        debug_print(f"Saved mixed proxy audio to {mixed_file}")
        
        # Save individual tracks
        for track_name, track_data in tracks.items():
            if len(track_data) > 0:
                track_file = f"{self.output_dir}/{config_obj.name}_proxy_{track_name}.wav"
                sf.write(track_file, track_data, self.sample_rate, subtype='FLOAT')
                debug_print(f"Saved {track_name} track to {track_file}")
        
        # Save metadata
        import json
        metadata = {
            'object_name': config_obj.name,
            'object_idx': config_obj.idx,
            'proxy_type': config_obj.proxy_type,
            'sample_rate': self.sample_rate,
            'total_samples': len(mixed),
            'duration': len(mixed) / self.sample_rate,
            'tracks': list(tracks.keys())
        }
        
        metadata_file = f"{self.output_dir}/{config_obj.name}_proxy_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        debug_print(f"Saved metadata to {metadata_file}")
