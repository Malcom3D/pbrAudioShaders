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
from ..lib.functions import _soxel_grid_shape

class EntityManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: str):
        with self._lock:
            if not self._initialized:
                self._sources = {}
                self._objects = {}
                self._outputs = {}
                self._wave_propagators = {}
                self._layer_managers = {}
                self._trajectories = {}
                self._collisions = {}
                self._forces = {}
                self._singleton = {}
                self._initialized = True

                self.sigleton_map = {
                    'config': 'Config',
                    'frames': 'FrameCounter',
                    'frequency_bands': 'FrequencyBands',
                    'soxel_grid': 'SoxelGrid'
                }
                self.entities_map = {
                    'sources': ['SphericalSource', 'PlanarSource'],
                    'objects': ['AcousticObject'],
                    'outputs': ['AmbisonicOutput', 'OmnidirectionalOutput', 'Figure8Output', 'CardioidOutput', 'HypercardioidOutput'],
                    'wave_propagators': 'WavePropagator',
                    'trajectories': ['TrajectoryData', 'tmpTrajectoryData'],
                    'collisions': [ 'CollisionData', 'tmpCollisionData'],
                    'forces': [ 'ForceData', 'ForceDataSequence']
                }

                config = Config(config)
#                ad = config.acoustic_domain
#                voxel_size = ad.voxel_size
#                grid_geometry = ad.geometry
#                ad.shape = _soxel_grid_shape(grid_geometry, voxel_size)
                self.register('config', config)

    # Dispatcher:
    def register(self, entity: str, obj: Any, idx: int = None) -> None:
        if entity in self.sigleton_map and not entity in self._singleton:
            self._singleton[entity] = obj
        elif entity in self.entities_map.keys() and not idx == None:
            for key in self.entities_map.keys():
                if entity in key:
                    for sub in self.entities_map[key]:
                        if sub in str(type(obj)):
                            entities = eval(f"self._{key}")
                            entities[idx] = obj
               
    def get(self, entity: str = None, idx: int = None) -> dict[str, Any]:
        """Get all objects"""
        if entity == None:
            return self._singleton, self._sources, self._objects, self._outputs, self._wave_propagators, self._trajectories, self._collisions, self._forces
        for key in self.sigleton_map.keys():
            if entity in key:
                return self._singleton[entity]
            else:
                for key in self.entities_map.keys():
                    if entity in key:
                        entities = eval(f"self._{entity}")
                        return entities.get(idx) if not idx == None else entities

    def unregister(self, entity: str, idx: int = None) -> None:
        """Unregister an object"""
        for key in self.sigleton_map.keys():
            if entity in key:
                del self._singleton[entity]
            elif not idx == None:
                for key in self.entities_map.keys():
                    if entity in key:
                        entities = eval(f"self._{entity}")
                        if idx in entities.keys():
                            del entities[idx]
