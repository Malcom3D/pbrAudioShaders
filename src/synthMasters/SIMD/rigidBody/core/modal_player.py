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
import time
import numpy as np
import soundfile as sf
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.signal import fftconvolve
from numba import jit, prange

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _parse_lib

@dataclass
class ModalPlayer:
    """Optimized ModalPlayer with batch processing"""
    entity_manager: EntityManager
    obj_idx: int
    player_id: int = None
    
    def __post_init__(self):
        print(f'ModalPlayer init: {self.obj_idx}')
        config = self.entity_manager.get('config')
        self.sample_counter = self.entity_manager.get('sample_counter')
        self.score_path = f"{config.system.cache_path}/score"
        self.output_dir = f"{config.system.cache_path}/modal_player"
        os.makedirs(self.output_dir, exist_ok=True)
        
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = (fps / fps_base) * subframes
        spsf = sample_rate / sfps
        
        if self.player_id is None:
            self.player_id = id(self)
        
        self.timeout = 60
        self.poll_interval = 0.001
        
        # Initialize audio tracks
        self.rigidbody_synth_track = np.zeros(self.sample_counter.total_samples)
        self.resonance_synth_track = np.zeros(self.sample_counter.total_samples)
        self.sliding_synth_track = np.zeros.zeros(self.sample_counter.total_samples)
        self.scraping_synth_track = np.zeros(self.sample_counter.total_samples)
        self.rolling_synth_track = np.zeros(self.sample_counter.total_samples)
        
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj
                t60_obj = self._get_modal_t60(config_obj)
                self.t60_samples = int(t60_obj * spsf)
                
                # Get synths
                rigidbody_synth = self.entity_manager.get('rigidbody_synth')
                resonance_synth = self.entity_manager.get('resonance_synth')
                
                for rb_key in rigidbody_synth.keys():
                    if rigidbody_synth[rb_key].obj_idx == self.obj_idx:
                        self.rigidbody_synth = rigidbody_synth[rb_key]
                        break
                
                for re_key in resonance_synth.keys():
                    if resonance_synth[re_key].obj_idx == self.obj_idx:
                        self.resonance_synth = resonance_synth[re_key]
                        break
                
                score_tracks = self.entity_manager.get('score_tracks')
                for idx in score_tracks.keys():
                    if score_tracks[idx].obj_idx == self.obj_idx and score_tracks[idx].is_final:
                        self.score = score_tracks[idx]
        
        self.sample_counter.register_player(self.player_id)
        print(f'ModalPlayer init end: {self.obj_idx}, t60_samples: {self.t60_samples}')
        
        sound_path = f"{config.system.cache_path}/audio_force"
        self.sliding_sound, self.scraping_sound, self.rolling_sound = self._load_sound_tracks(sound_path, config_obj.name)
    
    def compute(self) -> None:
        """Batch processing version using convolution matrix approach"""
        config = self.entity_manager.get('config')
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj
        
        sample_rate = int(config.system.sample_rate)
        sfps = (config.system.fps / config.system.fps_base) * config.system.subframes
        
        # Prepare batch data
        batch_data = self._prepare_batch_data(config_obj, sample_rate, sfps)
        
        # Process in batches using convolution
        self._process_batch_convolution(batch_data, config_obj, sample_rate)
        
        # Post-processing
        self.sample_counter.unregister_player(self.player_id)
        print(f"Player {self.player_id} finished processing")
    
    def _prepare_batch_data(self, config_obj, sample_rate: int, sfps: float) -> Dict:
        """Prepare batch data from score events"""
        fracture_frame = None
        is_shard_frame = None
        
        if not isinstance(config_obj.fractured, bool):
            fracture_frame = config_obj.fractured * sample_rate / sfps
        
        if not isinstance(config_obj.is_shard, bool):
            is_shard_frame = config_obj.is_shard * sample_rate / sfps
        
        batch_data = {
            'events': [],
            'fracture_frame': fracture_frame,
            'is_shard_frame': is_shard_frame,
            'total_samples': self.sample_counter.total_samples,
            't60_samples': self.t60_samples
        }
        
        # Group events by type for batch processing
        event_groups = {
            1: [],  # Impact
            2: [],  # Scraping
            3: [],  # Sliding
            4: [],  # Rolling
            5: [],  # Mixed
            6: []   # Static
        }
        
        for event in self.score.events:
            event_groups.setdefault(1, []).append(event)
        
        # Actually, let's process each event type separately
        for event in self.score.events:
            batch_data['events'].append({
                'start_sample': event.start_sample,
                'stop_sample': event.stop_sample,
                'coll_obj': event.coll_obj,
                'type': event.type,
                'vertex_ids': event.vertex_ids,
                'contact_area': event.contact_area,
                'force': event.force,
                'coupling_data': event.coupling_data
            })
        
        return batch_data
    
    def _process_batch_convolution(self, batch_data: Dict, config_obj, sample_rate: int):
        """Process audio using convolution matrix approach"""
        total_samples = batch_data['total_samples']
        t60_samples = batch_data['t60_samples']
        
        # Initialize output buffers for each event type
        rigidbody_output = np.zeros(total_samples, dtype=np.float32)
        resonance_output = np.zeros(total_samples, dtype=np.float32)
        noise_output = np.zeros(total_samples, dtype=np.float32)
        
        # Process each event
        for event_data in batch_data['events']:
            start = event_data['start_sample']
            stop = min(event_data['stop_sample'] + t60_samples, total_samples)
            
            if start >= total_samples:
                continue
            
            # Get event data for the duration
            event_length = stop - start
            
            # Create excitation signal for this event
            excitation = np.zeros(event_length, dtype=np.float32)
            
            # Fill excitation from force data
            force_data = event_data['force']
            if isinstance(force_data, np.ndarray) and len(force_data) > 0:
                n_force = min(len(force_data), event_length)
                excitation[:n_force] = force_data[:n_force]
            
            # Process with convolution
            if np.any(excitation != 0):
                # Get vertex IDs for this event
                vertex_ids = self._get_active_vertices(event_data)
                
                if len(vertex_ids) > 0:
                    # Process rigidbody synthesis
                    rb_result = self._convolve_with_banks(
                        excitation, vertex_ids, 
                        self.rigidbody_synth.banks
                    )
                    rigidbody_output[start:stop] += rb_result[:event_length]
                    
                    # Process resonance synthesis if available
                    if hasattr(self, 'resonance_synth') and self.resonance_synth is not None:
                        res_result = self._convolve_with_banks(
                            excitation, list(range(len(self.resonance_synth.banks))),
                            self.resonance_synth.banks
                        )
                        resonance_output[start:stop] += res_result[:event_length]
            
            # Process noise components
            noise_result = self._process_noise_components(
                event_data, start, stop, total_samples
            )
            noise_output[start:stop] += noise_result[:event_length]
        
        # Store results
        self.rigidbody_synth_track = rigidbody_output
        self.resonance_synth_track = resonance_output
        
        # Combine noise tracks
        self.sliding_synth_track = noise_output * 0.3  # Adjust scaling as needed
        self.scraping_synth_track = noise_output * 0.3
        self.rolling_synth_track = noise_output * 0.2
    
    def _convolve_with_banks(self, excitation: np.ndarray, 
                              vertex_ids: List[int], 
                              banks: np.ndarray) -> np.ndarray:
        """Convolve excitation with modal banks using FFT"""
        if len(vertex_ids) == 0 or len(banks) == 0:
            return np.zeros_like(excitation)
        
        output = np.zeros(len(excitation), dtype=np.float32)
        
        for vertex_id in vertex_ids:
            if vertex_id < len(banks) and isinstance(banks[vertex_id], ModalBank):
                # Get pre-computed impulse response
                ir = banks[vertex_id].compute_impulse_response(len(excitation))
                
                # FFT convolution
                conv_result = fftconvolve(excitation, ir, mode='full')[:len(excitation)]
                output += conv_result
        
        return output
    
    def _get_active_vertices(self, event_data: Dict) -> List[int]:
        """Get list of active vertex IDs for an event"""
        vertex_ids = event_data.get('vertex_ids')
        if vertex_ids is None:
            return []
        
        if isinstance(vertex_ids, np.ndarray):
            # Handle compressed boolean array
            if vertex_ids.dtype == np.bool_:
                return np.where(vertex_ids)[0].tolist()
            else:
                return vertex_ids.tolist()
        
        return vertex_ids
    
    def _process_noise_components(self, event_data: Dict, start: int, 
                                   stop: int, total_samples: int) -> np.ndarray:
        """Process noise-based components (sliding, scraping, rolling)"""
        event_length = stop - start
        noise_output = np.zeros(event_length, dtype=np.float32)
        
        # Get contact area for scaling
        contact_area = event_data.get('contact_area', 0)
        if contact_area == 0:
            return noise_output
        
        # Get force data
        force_data = event_data['force']
        if not isinstance(force_data, np.ndarray):
            return noise_output
        
        # Get sound files
        sound_files = {
            2: self.scraping_sound,  # Scraping
            3: self.sliding_sound,   # Sliding
            4: self.rolling_sound    # Rolling
        }
        
        synth_type = event_data['type']
        if isinstance(synth_type, np.ndarray):
            # Use the most common type
            synth_type = np.bincount(synth_type.astype(int)).argmax()
        
        if synth_type in sound_files:
            sound = sound_files[synth_type]
            if start < len(sound):
                n_sound = min(len(sound) - start, event_length)
                noise_output[:n_sound] = sound[start:start + n_sound] * contact_area
        
        return noise_output
    
    def _get_modal_t60(self, config_obj) -> float:
        """Get maximum T60 from modal model"""
        cache_path = self.entity_manager.get('config').system.cache_path
        obj_name = config_obj.name
        if config_obj.proxy_type is not False:
            obj_name = f"{config_obj.name}_proxy_{config_obj.proxy_type}"
        lib_file = f"{cache_path}/dsp/{obj_name}.lib"
        
        modal_data = _parse_lib(lib_file)
        t60s = modal_data['t60s']
        return float(np.max(t60s))
    
    def save_synth_tracks(self):
        """Save all synth tracks"""
        self.save_synth_track(self.rigidbody_synth_track, 'rigidbody', True)
        self.save_synth_track(self.resonance_synth_track, 'resonance')
        self.save_synth_track(self.sliding_synth_track, 'sliding')
        self.save_synth_track(self.scraping_synth_track, 'scraping')
        self.save_synth_track(self.rolling_synth_track, 'rolling')
    
    def save_synth_track(self, track: np.ndarray, suffix: str, normalize: bool = False):
        """Save individual track as WAV file"""
        config = self.entity_manager.get('config')
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj
        
        if not np.any(track):
            print(f"Track {suffix} for {config_obj.name} is empty, skipping")
            return
        
        if normalize:
            track = track / (np.max(np.abs(track)) + 1e-10)
        
        sample_rate = int(config.system.sample_rate)
        track_name = config_obj.name
        track_file = f"{track_name}_{suffix}.raw"
        wave_file = f"{self.output_dir}/{track_file}"
        
        sf.write(wave_file, track, sample_rate, subtype='FLOAT')
        print(f"Saved {track_name} tracks to {self.output_dir}")
    
    def _load_sound_tracks(self, sound_path: str, obj_name: str):
        """Load pre-computed sound tracks"""
        sliding_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)
        scraping_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)
        rolling_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)
        
        if os.path.exists(f"{sound_path}/{obj_name}_sliding_sound.raw"):
            sliding_sound += np.fromfile(f"{sound_path}/{obj_name}_sliding_sound.raw", dtype=np.float32)
        if os.path.exists(f"{sound_path}/{obj_name}_scraping_sound.raw"):
            scraping_sound += += np.fromfile(f"{sound_path}/{obj_name}_scraping_sound.raw", dtype=np.float32)
        if os.path.exists(f"{sound_path}/{obj_name}_rolling_sound.raw"):
            rolling_sound += np.fromfile(f"{"{sound_path}/{obj_name}_rolling_sound.raw", dtype=np.float32)
        
        return sliding_sound, scraping_sound, rolling_sound
