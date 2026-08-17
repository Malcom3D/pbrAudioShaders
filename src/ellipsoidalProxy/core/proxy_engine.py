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
