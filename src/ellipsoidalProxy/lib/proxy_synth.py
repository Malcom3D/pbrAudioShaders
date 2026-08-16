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
        
        # Pre-compute FFT of IRs for fast convolution
        self._precompute_ir_ffts()
        
        # Path to audio-force tracks
        self.audio_force_dir = f"{config.system.cache_path}/audio_force"
    
    def _precompute_ir_ffts(self):
        """Pre-compute FFT of all IRs for faster convolution."""
        n_sizes = self.ir_table.n_size_steps
#        n_types = 6  # no-contact, impact, sliding, scraping, rolling, static
        n_types = 4  # impact, sliding, scraping, rolling
        n_bands = self.ir_table.n_frequency_bands
        
        # Pre-compute FFTs: (n_sizes, n_types, n_bands, fft_size/2+1)
        self._ir_ffts = np.zeros((n_sizes, n_types, n_bands, self.fft_size // 2 + 1), dtype=np.complex64)
        
        for size_idx in range(n_sizes):
            for type_idx in range(n_types):
                for band_idx in range(n_bands):
                    debug_print('size_idx', size_idx, 'type_idx', type_idx, 'band_idx', band_idx)
                    ir = self.ir_table.ir_table[size_idx, type_idx, band_idx]
                    # Pad to FFT size
                    padded = np.zeros(self.fft_size)
                    ir_len = min(len(ir), self.fft_size)
                    padded[:ir_len] = ir[:ir_len]
                    self._ir_ffts[size_idx, type_idx, band_idx] = np.fft.rfft(padded)
    
    def compute(self, obj_idx: int, total_samples: int) -> None:
        """
        Compute proxy synth for an object using audio-force tracks from ForceSynth.
        
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
        debug_print('Get size scale for this object', size_scale.shape, np.count_nonzero(size_scale))
        
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
        }
        
        for track_name, contact_type in contact_type_map.items():
            if audio_tracks.get(track_name) is not None:
                # Get the excitation signal
                excitation = audio_tracks[track_name]
                debug_print('Get the excitation signal', config_obj.name, excitation.shape, np.count_nonzero(excitation))
                
                # Trim or pad to total_samples
                if excitation.shape[0] < total_samples:
                    excitation = np.pad(excitation, (0, total_samples - len(excitation)))
                elif excitation.shape[0] > total_samples:
                    excitation = excitation[:total_samples]
                debug_print('Trim or pad to total_samples', config_obj.name, excitation.shape, np.count_nonzero(excitation))
                
                # Apply IR convolution
                if not track_name == 'rolling_sound':
                    processed = self._convolve_with_ir(excitation, size_scale, contact_type)
                    debug_print('Apply IR convolution', config_obj.name, processed.shape, np.count_nonzero(processed))
                
                # Apply equalization
                if track_name == 'rolling_sound':
                    processed = excitation
                    excitation = audio_tracks['rolling']
                processed = self.equalizer.apply_equalization(processed, contact_type, excitation)
                debug_print('Apply equalization', config_obj.name, processed.shape, np.count_nonzero(processed))
                processed_tracks[track_name] = processed

        # Mix all tracks
        mixed = np.zeros(total_samples, dtype=np.float32)
#        for track in processed_tracks.values():
#            debug_print('Mix all tracks', mixed.shape, track.shape)
#            mixed += track

        for track_name in processed_tracks.keys():
            if track_name in ['impact', 'rolling']:
                # Normalize impact
                max_val = np.max(np.abs(processed_tracks[track_name]))
                if max_val > 0:
                    processed_tracks[track_name] /= max_val * 0.9
            if track_name == 'rolling':
                # Reduce Volume
                processed_tracks[track_name] *= 0.5
            if track_name in ['sliding', 'scraping']:
                # Reduce Volume
                processed_tracks[track_name] *= 0.0075
            if track_name == 'rolling_sound':
                processed_tracks[track_name] *= 5

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
            Dictionary with keys: 'impact', 'sliding', 'scraping', 'rolling', 'rolling_sound'
            or None if no tracks found.
        """
        tracks = {}
        track_names = ['impact', 'sliding', 'scraping', 'rolling', 'rolling_sound']
        
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
    
    def _convolve_with_ir(self, signal: np.ndarray, size_scale: float, contact_type: int) -> np.ndarray:
        """
        Apply IR convolution to the excitation signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Excitation signal (audio-force track)
        size_scale : float
            Normalized size (0-1)
        contact_type : int
            Contact type (0=impact, 1=sliding, 2=scraping, 3=rolling)
        
        Returns:
        --------
        np.ndarray : Convolved signal
        """
        # Get IR for this size and contact type
        ir = self.ir_table.get_ir(size_scale, contact_type)
        
        n_samples = len(signal)
        n_bands = ir.shape[0]
        
        # Use overlap-add for long signals
        if n_samples > self.fft_size:
            output = self._overlap_add_convolve(signal, ir)
        else:
            output = self._fft_convolve(signal, ir)
        
        return output
    
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
    
    def _overlap_add_convolve(self, signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Overlap-add convolution for long signals.
        
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
                
                # Normalize to 0-1 range
                size_range = self.ir_table.max_size - self.ir_table.min_size
                if size_range > 0:
                    size_scale = (size - self.ir_table.min_size) / size_range
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
                track_file = f"{self.output_dir}/{config_obj.name}_proxy_{track_name}.raw"
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
