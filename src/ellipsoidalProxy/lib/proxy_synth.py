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
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from .proxy_ir_table import ProxyIRTable
from .proxy_eq import ProxyEqualizer

@dataclass
class ProxySynth:
    """
    Lightweight physically-based synthesizer for proxy meshes.
    
    Uses audio-force tracks from ForceSynth as excitation signals,
    applies IR convolution and frequency equalization.
    
    Features:
    - Loads audio-force tracks (impact, sliding, scraping, rolling)
    - Applies IR convolution with size interpolation
    - Dynamic frequency equalization
    - Supports all contact types
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
        
        # Cache for IR FFTs - will be populated per object
        self._ir_fft_cache = {}
        
        # Path to audio-force tracks
        self.audio_force_dir = f"{config.system.cache_path}/audio_force"
        
        # Cache for material_key lookups
        self._material_key_cache = {}
    
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
    
    def _get_ir(self, config_obj: Any, size_scale: float, contact_type: int) -> np.ndarray:
        """
        Get interpolated IR for an object.
        
        Parameters:
        -----------
        config_obj : ObjectConfig
            Object configuration
        size_scale : float
            Normalized size (0-1)
        contact_type : int
            Contact type (0=impact, 1=sliding, 2=scraping, 3=rolling)
            
        Returns:
        --------
        np.ndarray : Interpolated IR
        """
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        
        # Get IR from table
        return self.ir_table.get_ir(material_key, proxy_type, size_scale)
    
    def _precompute_ir_ffts_for_object(self, config_obj: Any) -> None:
        """
        Pre-compute FFT of IRs for a specific object.
        
        This is called once per object to cache the IR FFTs.
        """
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        cache_key = (material_key, proxy_type)
        
        if cache_key in self._ir_fft_cache:
            return
        
        # Get size steps for this material/shape
        size_steps = self.ir_table.size_steps.get(cache_key)
        if size_steps is None:
            debug_print(f"No size steps found for cache_key {cache_key}")
            return
        
        n_steps = len(size_steps)
        n_types = 4  # impact, sliding, scraping, rolling
        
        # Pre-compute FFTs for this object
        # Note: We assume IRs for all bands are available. 
        # In ProxyIRTable, ir_table is indexed by (material_key, proxy_type)
        # and each entry has shape (n_steps, max_ir_length)
        ir_matrix = self.ir_table.ir_table.get(cache_key)
        
        if ir_matrix is None:
            debug_print(f"No IR table found for cache_key {cache_key}")
            return
        
        # ir_matrix shape: (n_steps, max_ir_length)
        # For each size step and contact type, we need to compute FFT
        # Since the IR table currently stores a single IR per size step (not per type),
        # we use the same IR for all contact types and apply type-specific scaling later.
        fft_shape = (n_steps, n_types, self.fft_size // 2 + 1)
        ir_ffts = np.zeros(fft_shape, dtype=np.complex64)
        
        for size_idx in range(n_steps):
            ir = ir_matrix[size_idx]
            # Pad to FFT size
            padded = np.zeros(self.fft_size)
            ir_len = min(len(ir), self.fft_size)
            padded[:ir_len] = ir[:ir_len]
            ir_fft = np.fft.rfft(padded)
            
            # Use the same IR for all contact types
            # (type-specific shaping is handled by the equalizer)
            for type_idx in range(n_types):
                ir_ffts[size_idx, type_idx, :] = ir_fft
        
        self._ir_fft_cache[cache_key] = ir_ffts
    
    def _get_ir_fft(self, config_obj: Any, size_idx: int, contact_type: int) -> np.ndarray:
        """
        Get precomputed IR FFT for a specific size and contact type.
        """
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        cache_key = (material_key, proxy_type)
        
        if cache_key not in self._ir_fft_cache:
            self._precompute_ir_ffts_for_object(config_obj)
        
        ir_ffts = self._ir_fft_cache.get(cache_key)
        if ir_ffts is None:
            # Return a default impulse response
            return np.ones(self.fft_size // 2 + 1, dtype=np.complex64)
        
        return ir_ffts[size_idx, contact_type, :]
    
    def _get_size_idx(self, config_obj: Any, size_scale: float) -> int:
        """
        Get the size index for interpolation.
        """
        material_key = self._get_material_key(config_obj)
        proxy_type = config_obj.proxy_type
        cache_key = (material_key, proxy_type)
        
        size_steps = self.ir_table.size_steps.get(cache_key)
        if size_steps is None:
            return 0
        
        # Clamp size_scale
        size_scale = np.clip(size_scale, 0, 1)
        
        # Find the nearest size step
        n_steps = len(size_steps)
        idx = int(size_scale * (n_steps - 1))
        return min(idx, n_steps - 1)

    def compute(self, obj_idx: int, total_samples: int) -> None:
        """
        Compute proxy synth for an object using audio-force tracks from ForceSynth.
        
        Parameters:
        -----------
        obj_idx : int
            Object index
        total_samples : int
            Total number of samples for the audio output
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
        
        # Precompute IR FFTs for this object
        self._precompute_ir_ffts_for_object(config_obj)
        
        # Get size scale for this object
        size_scale = self._compute_size_scale(config_obj)
        size_idx = self._get_size_idx(config_obj, size_scale)
        debug_print(f'Size scale for {config_obj.name}: {size_scale:.3f}, size_idx: {size_idx}')
        
        # Load audio-force tracks
        audio_tracks = self._load_audio_force_tracks(config_obj.name)
        
        if audio_tracks is None:
            debug_print(f"No audio-force tracks found for {config_obj.name}")
            return
        
        # Get total samples from the longest track
        tracks_samples = max(len(track) for track in audio_tracks.values() if track is not None)
        
        if tracks_samples == 0:
            debug_print(f"Audio-force tracks are empty for {config_obj.name}")
            return
        
        # Process each track through IR convolution and EQ
        processed_tracks = {}
        
        # Contact type mapping: track_name -> contact_type_index
        contact_type_map = {
            'impact': 0,
            'sliding': 1,
            'scraping': 2,
            'rolling': 3,
            'rolling_sound': 4
            'sliding_sound': 4
            'scraping_sound': 4
        }
        
        for track_name, contact_type in contact_type_map.items():
            if audio_tracks.get(track_name) is not None:
                # Get the excitation signal
                excitation = audio_tracks[track_name]
                debug_print(f'Loading excitation signal for {config_obj.name} - {track_name}: {excitation.shape}, non-zero: {np.count_nonzero(excitation)}')
                
                # Trim or pad to total_samples
                if excitation.shape[0] < total_samples:
                    excitation = np.pad(excitation, (0, total_samples - len(excitation)))
                elif excitation.shape[0] > total_samples:
                    excitation = excitation[:total_samples]
                debug_print(f'Trimmed/padded excitation for {config_obj.name} - {track_name}: {excitation.shape}, non-zero: {np.count_nonzero(excitation)}')
                
                # Apply IR convolution
                if contact_type == 4:
                    processed = excitation
                    excitation = audio_tracks.get(track_name.replace('_sound',''), np.zeros_like(excitation))
                else:
                    processed = self._convolve_with_ir(config_obj, excitation, size_idx, contact_type)
                    debug_print(f'After IR convolution for {config_obj.name} - {track_name}: {processed.shape}, non-zero: {np.count_nonzero(processed)}')
                
                # Apply equalization
                processed = self.equalizer.apply_equalization(processed, contact_type, excitation)
                debug_print(f'After equalization for {config_obj.name} - {track_name}: {processed.shape}, non-zero: {np.count_nonzero(processed)}')
                processed_tracks[track_name] = processed

        # Mix all tracks
        mixed = np.zeros(total_samples, dtype=np.float32)

        for track_name in processed_tracks.keys():
            if track_name in ['impact', 'rolling']:
                # Normalize impact
                max_val = np.max(np.abs(processed_tracks[track_name]))
                if max_val > 0:
                    processed_tracks[track_name] /= max_val * 0.9
            if track_name == 'rolling':
                # Reduce Volume
                processed_tracks[track_name] *= 0.01
            if track_name in ['sliding', 'scraping']:
                # Reduce Volume
                processed_tracks[track_name] *= 0.0075
            if track_name == 'rolling_sound':
                processed_tracks[track_name] *= 2.5
            if track_name == 'sliding_sound':
                processed_tracks[track_name] *= 2.5
            if track_name == 'scraping_sound':
                processed_tracks[track_name] *= 2.5

            mixed += processed_tracks[track_name]
        
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
            Dictionary with keys: 'impact', 'sliding', 'scraping', 'rolling', 'rolling_sound', 'sliding_sound', 'scraping_sound'
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
            else:
                debug_print(f"Track file not found: {track_file}")
        
        # Return None if no tracks were loaded
        if not tracks:
            return None
        
        return tracks
    
    def _convolve_with_ir(self, config_obj: Any, signal: np.ndarray, size_idx: int, contact_type: int) -> np.ndarray:
        """
        Apply IR convolution to the excitation signal.
        
        Parameters:
        -----------
        config_obj : ObjectConfig
            Object configuration
        signal : np.ndarray
            Excitation signal (audio-force track)
        size_idx : int
            Size index for IR lookup
        contact_type : int
            Contact type (0=impact, 1=sliding, 2=scraping, 3=rolling)
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        # Get IR FFT for this size and contact type
        ir_fft = self._get_ir_fft(config_obj, size_idx, contact_type)
        
        n_samples = len(signal)
        
        # Use overlap-add for long signals
        if n_samples > self.fft_size:
            output = self._overlap_add_convolve_fft(signal, ir_fft)
        else:
            output = self._fft_convolve(signal, ir_fft)
        
        return output
    
    def _fft_convolve(self, signal: np.ndarray, ir_fft: np.ndarray) -> np.ndarray:
        """
        FFT-based convolution with precomputed IR FFT.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        ir_fft : np.ndarray
            Precomputed FFT of the impulse response (complex)
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        n_samples = len(signal)
        
        # Pad signal to FFT size
        padded_signal = np.zeros(self.fft_size)
        padded_signal[:min(n_samples, self.fft_size)] = signal[:min(n_samples, self.fft_size)]
        
        # FFT of signal
        signal_fft = np.fft.rfft(padded_signal)
        
        # Multiply in frequency domain
        result_fft = signal_fft * ir_fft
        
        # Inverse FFT
        result = np.fft.irfft(result_fft, n=self.fft_size)
        
        # Trim to signal length
        output = result[:n_samples]
        
        return output
    
    def _overlap_add_convolve_fft(self, signal: np.ndarray, ir_fft: np.ndarray) -> np.ndarray:
        """
        Overlap-add convolution for long signals using precomputed IR FFT.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        ir_fft : np.ndarray
            Precomputed FFT of the impulse response (complex)
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        n_samples = len(signal)
        
        # Initialize output
        output = np.zeros(n_samples + self.fft_size, dtype=np.float32)
        
        # Process in blocks
        n_blocks = int(np.ceil(n_samples / self.hop_size))
        
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
            
            # Multiply in frequency domain
            result_fft = block_fft * ir_fft
            
            # Inverse FFT
            result = np.fft.irfft(result_fft, n=self.fft_size)
            
            # Overlap-add
            output[start:start + self.fft_size] += result
        
        # Trim to signal length
        output = output[:n_samples]
        
        return output

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
        """
        Save synthesized audio to files.
        
        Parameters:
        -----------
        config_obj : ObjectConfig
            Object configuration
        mixed : np.ndarray
            Mixed audio
        tracks : Dict[str, np.ndarray]
            Individual processed tracks
        """
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
        metadata = {
            'object_name': config_obj.name,
            'object_idx': config_obj.idx,
            'proxy_type': config_obj.proxy_type,
            'sample_rate': self.sample_rate,
            'total_samples': len(mixed),
            'duration': len(mixed) / self.sample_rate,
            'tracks': list(tracks.keys())
        }
        
        import json
        metadata_file = f"{self.output_dir}/{config_obj.name}_proxy_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        debug_print(f"Saved metadata to {metadata_file}")
