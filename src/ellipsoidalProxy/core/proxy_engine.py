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

import numpy as np
from dask import delayed
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioStorage import StorageEngine

from ..lib.proxy_synth import ProxySynth
from ..lib.proxy_ir_table import ProxyIRTable

@dataclass
class ProxyEngine:
    """
    Lightweight physically-based synthesizer for proxy meshes (proxy_type 0,1 and 2).
    
    Uses precomputed IRs and FFT convolution for efficient sound synthesis.
    Post-process rendered signals with dynamic frequency equalizer to match event type.
    Supports impact, sliding, scraping, rolling and mixed force type.
    """
    entity_manager: EntityManager
    
    # Components
    proxy_synth: ProxySynth = None
    ir_table: ProxyIRTable = None
    
    def __post_init__(self):
        # Initialize StorageEngine
        self.storage_engine = StorageEngine(self.entity_manager)
        debug_print("StorageEngine initialized for ProxyEngine.")
        config = self.entity_manager.get('config')
        
        # Initialize proxy synth
        self.proxy_synth = ProxySynth(entity_manager=self.entity_manager)
    
        # Map IR table
        self.ir_table = self.proxy_synth.ir_table

        config = self.entity_manager.get('config')
        proxy_meshes = []

        for config_obj in config.objects:
            if config.system.enable_proxy_synth and config_obj.proxy_type in [0,1,2]:
                proxy_meshes.append(config_obj)

        # Compute IR table from proxy meshes
        self.ir_table.compute_ir_table(proxy_meshes)

    def _register_storage_schema(self):
        """
        Registers the data schema for raw proxy synth tracks with the StorageEngine.
        This defines how the data is organized on the graph.
        """
        debug_print("Registering storage schema for proxy synth tracks...")
        config = self.entity_manager.get('config')
        track_names = ['impact', 'resonance', 'sliding', 'scraping', 'rolling']
        if config.system.enable_noise_enhancement:
            track_names += ['sliding_sound', 'scraping_sound', 'rolling_sound']

        # default_behaviors/kinds = [ collection, engine, object, track, signal ]
        default_behaviors = {
            'collection': 'none',
            'engine': 'none',
            'object': 'multi',
            'track': 'single',
            'signal': 'merge'
            }
        default_kinds = {
            'collection': 'multi-track',
            'engine': 'multi-track',
            'object': 'multi-track',
            'track': 'mono',
            'signal': 'mono',
            }
        self.storage_engine.register(
            collection=config.system.collection,
            objs_type="objects",
            engine=self.__class__.__name__,
            track_group="modal_displacement",
            track_names=track_names,
            signal_type="float32",
            total_samples=int(self.total_samples),
            sample_rate=int(config.system.sample_rate),
            bit_depth=config.system.bit_depth,
            subtype='FLOAT',
            metadata={'description': f'Final synthesized proxy tracks from {self.__class__.__name__}'},
            default_behaviors=default_behaviors,
            default_kinds=default_kinds
        )

        self.progress = _update_status(f"{self.status_dir}", "/prebake", 5) # Update progress
        debug_print("Proxy synth storage schema registered and materialized.")

    @delayed
    def compute(self, obj_idx: int, total_samples: int) -> None:
        """
        Process audio for a proxy object.
        
        Parameters:
        -----------
        obj_idx : int
            Object index
        total_samples : int
            Total number of samples for the audio output
        """
        self.proxy_synth.compute(obj_idx, total_samples)
