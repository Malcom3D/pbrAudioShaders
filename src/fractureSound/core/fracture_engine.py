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
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, field
from dask import delayed, compute

from pbrAudioCommon import EntityManager, CollisionData, ForceDataSequence, TrajectoryData, FractureEvent, FractureType
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.fracture_detector import FractureDetector
from ..lib.fracture_modal import FractureModalModel
from ..lib.fracture_synth import FractureSynth

@dataclass
class fractureEngine:
    """
    Main engine for fracture sound synthesis.

    Orchestrates:
    1. Fracture detection from trajectory data
    2. Modal model adaptation for fragments
    3. Fracture sound synthesis
    4. Non-destructive post-processing chain attachment
    """
    
    entity_manager: EntityManager
    fracture_events: List[FractureEvent] = field(default_factory=list)
    storage_engine: StorageEngine = None
    total_samples: int = 1

    def __post_init__(self):
        config = self.entity_manager.get('config')
        
        self.sample_rate = config.system.sample_rate
        self.fps = config.system.fps
        self.fps_base = config.system.fps_base
        self.subframes = config.system.subframes
        self.sfps = (self.fps / self.fps_base) * self.subframes
        
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        self.progress = 0

        self.fracture_modal_dir = f"{config.system.cache_path}/fracture_modal"
        
        os.makedirs(self.status_dir, exist_ok=True)
        os.makedirs(self.fracture_modal_dir, exist_ok=True)
        
        # Initialize components
        self.detector = FractureDetector(self.entity_manager)
        self.modal_model = FractureModalModel(self.entity_manager)
        self.synth = FractureSynth(self.entity_manager)

        # Get fracture_events
        self.fracture_events = self.entity_manager.get('fracture_events') or []

        # Compute total samples
        self.total_samples = int(config.system.total_frames * self.sample_rate / self.sfps)

        # StorageEngine + schema
        self.storage_engine = StorageEngine(self.entity_manager)
        self._register_storage_schema()

    def _register_storage_schema(self):
        """
        Register the AudioSchema for fracture-engine per-fragment tracks.

        Track layout (per fragment object that is a shard):
          - shatter   : burst-of-grains synthesis (one-shot; short burst)
          - crack     : sparse crack events
          - snap      : nucleation tick + fragment ringdown
          - ringdown  : pure modal ringdown (post-fracture resonance)
          - tick      : high-frequency nucleation tick

        Only the shatter/crack/snap tracks are populated depending on the
        fracture type. All tracks are registered so that the DAG walker
        always finds a consistent layout per object.
        """
        config = self.entity_manager.get('config')
        debug_print("Registering storage schema for fracture tracks...")

        track_names = ['shatter', 'crack', 'snap', 'ringdown', 'tick']

        default_behaviors = {
            'collection': 'none',
            'engine': 'none',
            'object': 'multi',
            'track': 'single',
            'signal': 'merge',
        }
        default_kinds = {
            'collection': 'multi-track',
            'engine': 'multi-track',
            'object': 'multi-track',
            'track': 'mono',
            'signal': 'mono',
        }

        self.storage_engine.register(
            collection=self.collection,
            objs_type="objects",
            engine=self.__class__.__name__,
            track_group="modal-displacement",
            track_names=track_names,
            signal_type="float32",
            total_samples=int(self.total_samples),
            sample_rate=self.sample_rate,
            bit_depth=config.system.bit_depth,
            subtype='FLOAT',
            metadata={'description': f'Physically-based fracture synthesis tracks from {self.__class__.__name__}'},
            default_behaviors=default_behaviors,
            default_kinds=default_kinds,
        )
        _update_status(f"{self.status_dir}", "/init", 99)
        debug_print("Fracture storage schema registered and materialized.")

    def bake(self) -> None:
        """
        Main bake function - detect fractures and synthesize sounds.
        """
        config = self.entity_manager.get('config')
        
        # Process all objects with fracture configuration
        for conf_obj in config.objects:
            self.progress = _update_status(f"{self.status_dir}", "/bake", self.progress + 50/len(config.objects))
            if conf_obj.fractured is not False and conf_obj.shard is not False:
                # This is a fractured object with shards
                original_idx = conf_obj.idx
                fragment_indices = conf_obj.shard
                
                # Detect fracture events
                events = self.detect_fractures()
#                events = self.detector.detect_fracture_events(original_idx, fragment_indices)
                
                for event in events:
                    self.fracture_events.append(event)
        
        # Process all fracture events
        if self.fracture_events:
            self._process_fracture_events()
    
        _update_status(f"{self.status_dir}", "/bake", 99)
        debug_print("Fracture detection and synthesis ended.")

    def _process_fracture_events(self):
        """
        Process all fracture events in parallel.
        """
        debug_print(f"Processing {len(self.fracture_events)} fracture events")
        
        # Tasks for modal model computation
        modal_tasks = []
        for event in self.fracture_events:
            for frag_idx in event.fragment_indices:
                modal_tasks.append(self._delayed_compute_modal(event, frag_idx))
        
        if modal_tasks:
            compute(*modal_tasks)
            debug_print(f"Computed modal models for {len(modal_tasks)} fragments")
        
        # Tasks for sound synthesis
        sound_tasks = []
        for event in self.fracture_events:
            sound_tasks.append(self._delayed_synthesize(event))
        
        if sound_tasks:
            compute(*sound_tasks)
            debug_print(f"Synthesized sounds for {len(sound_tasks)} fracture events")
    
    @delayed
    def _delayed_compute_modal(self, event: FractureEvent, fragment_idx: int):
        """Delayed computation of modal model."""
        self.modal_model.compute(event, fragment_idx)
    
    @delayed
    def _delayed_synthesize(self, event: FractureEvent):
        """Delayed synthesis of fracture sound."""
        self.synth.compute(event, self.total_samples)
    
    def detect_fractures(self) -> List[FractureEvent]:
        """
        Detect fracture events from configuration.
        
        Returns:
        --------
        List of detected fracture events
        """
        config = self.entity_manager.get('config')
        events = []
        
        for conf_obj in config.objects:
            if conf_obj.fractured is not False and conf_obj.shard is not False:
                original_idx = conf_obj.idx
                fragment_indices = conf_obj.shard
                
                detected = self.detector.detect_fracture_events(original_idx, fragment_indices)
                events.extend(detected)
        
        return events
    
    def synthesize_fracture(self, event: FractureEvent) -> Dict[str, np.ndarray]:
        """
        Synthesize sound for a single fracture event.
        
        Parameters:
        -----------
        event : FractureEvent
            The fracture event to synthesize
            
        Returns:
        --------
        Dict with audio tracks
        """
        # Compute modal models for fragments if not already done
        for frag_idx in event.fragment_indices:
            self.modal_model.compute(event, frag_idx)
        
        # Synthesize fracture sound
        return self.synth.compute(event, self.total_samples)
    
    def get_fracture_events(self, original_obj_idx: int = None) -> List[FractureEvent]:
        """
        Get fracture events, optionally filtered by object index.
        
        Parameters:
        -----------
        original_obj_idx : int, optional
            Filter by original object index
            
        Returns:
        --------
        List of fracture events
        """
        if original_obj_idx is not None:
            return [e for e in self.fracture_events 
                   if e.original_obj_idx == original_obj_idx]
        return self.fracture_events
    
    def clear_events(self):
        """Clear all fracture events."""
        self.fracture_events = []
        
        # Clear saved files
        for filename in os.listdir(self.fracture_dir):
            if filename.endswith('.pkl'):
                os.remove(f"{self.fracture_dir}/{filename}")
        
        debug_print("Cleared all fracture events")
