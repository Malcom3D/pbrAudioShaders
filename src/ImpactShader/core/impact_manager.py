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

import threading
from typing import List, Tuple, Any

from ..utils.config import Config
#from utils.config import Config

class ImpactManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: str):
        print('ImpactManager.init')
        with self._lock:
            if not self._initialized:
                self._impacts = {}
                self._initialized = True
                self._config = Config(config)

    def register(self, impact_data: Any) -> None:
        if 'ImpactData' in str(type(impact_data)):
            self._impacts[len(self._impacts)] = impact_data

    def get(self, type: str = None, idx: int = None) -> Any:
        if type == None:
            return self._config, self._impacts
        if 'config' in type:
            return self._config
        elif 'impact' in type:
            return self._impacts[idx] if not idx == None else self._impacts

    def get_expos(self, obj_idx: int) -> List[int]:
        """
        Get all vertex IDs (excitation positions) for a specific object
        from all registered impacts.
        """
        expos_list = []
        
        # Iterate through all registered impacts
        for idx, impact in self._impacts.items():
            # Get vertex IDs for this specific object from the current impact
            vertex_ids = impact.get_all_vertices(obj_idx)
            
            # Add all vertex IDs to the list removing duplicates
            for vertex_id in vertex_ids:
                if vertex_id not in expos_list:
                    expos_list.append(vertex_id)
        
        expos_list.sort()
        return expos_list
    
    def get_force_mag(self) -> float:
        """Get maximum force magnitude across all impacts"""
        max_force = 0.0
        for idx, impact in self._impacts.items():
            force_mag = impact.get_force_magnitude()
            max_force = max(max_force, force_mag)
        
        # Return order of magnitude
        if max_force > 0:
            return 10**math.ceil(math.log10(max_force))
        return 1.0

    def unregister(self) -> None:
        pass
