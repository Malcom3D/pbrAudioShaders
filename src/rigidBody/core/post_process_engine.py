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
Integration wrapper for post-processing in the rigid body engine.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dask import delayed, compute

from physicsSolver import EntityManager
from ..lib.post_process import PostProcess
from ..lib.functions import _update_status


@dataclass
class PostProcessEngine:
    """
    Engine that orchestrates post-processing of all rendered audio tracks.
    
    Integrates with the existing rigidBodyEngine pipeline.
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        system_config = config.system
        self.status_dir = f"{system_config.cache_path}/status/PostProcessEngine"
        os.makedirs(self.status_dir, exist_ok=True)
        
        self.post_processor = PostProcess(entity_manager=self.entity_manager)
    
    def process(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all objects in the scene after modal player has finished.
        
        Returns:
            Dictionary of processed tracks per object
        """
        _update_status(f"{self.status_dir}/process", 0)
        
        config = self.entity_manager.get('config')
        tasks = []
        
        # Create delayed tasks for each dynamic object
        for obj in config.objects:
            task = self._process_single_object(obj.name, obj.idx)
            tasks.append(task)
        
        # Compute all tasks in parallel
        results_list = compute(*tasks)
        
        # Combine results
        all_results = {}
        for result in results_list:
            if result:
                all_results.update(result)
        
        _update_status(f"{self.status_dir}/process", 100)
        
        return all_results
    
    @delayed
    def _process_single_object(self, obj_name: str, obj_idx: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Process a single object's tracks."""
        tracks = self.post_processor.process_object(obj_name, obj_idx)
        return {obj_name: tracks} if tracks else {}
    
    def process_with_modal_player(self, modal_player_results: List[Any] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process tracks after ModalPlayer has finished rendering.
        
        This is designed to be called in the bake pipeline after ModalPlayer.
        """
        _update_status(f"{self.status_dir}/post_bake", 0)
        
        # If modal player results are provided, use them
        if modal_player_results:
            # Process based on saved files (modal player saves to disk)
            pass
        
        # Process all objects
        results = self.process()
        
        _update_status(f"{self.status_dir}/post_bake", 100)
        
        return results
