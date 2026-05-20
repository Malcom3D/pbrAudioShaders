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

# ./physicsSolver/core/entity_manager.py
import copy
import numpy as np
from multiprocessing import Manager, Lock
from typing import List, Tuple, Any, Dict
from ..utils.config import Config

class EntityManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: str = None):
        if not self._initialized:
            # Use Manager for shared state across processes
            self.manager = Manager()
            
            # Initialize shared dictionaries
            self._singleton = self.manager.dict()
            self._sources = self.manager.dict()
            self._objects = self.manager.dict()
            self._outputs = self.manager.dict()
            self._output_datas = self.manager.dict()
            self._wave_propagators = self.manager.dict()
            self._layer_managers = self.manager.dict()
            self._trajectories = self.manager.dict()
            self._collisions = self.manager.dict()
            self._forces = self.manager.dict()
            self._modal_vertices = self.manager.dict()
            self._score_tracks = self.manager.dict()
            self._rigidbody_synth = self.manager.dict()
            self._resonance_synth = self.manager.dict()
            
            self._initialized = True
            
            # Local lock for thread safety within a process
            self._local_lock = Lock()

            self.singleton_map = {
                'config': 'Config',
                'frames': 'FrameCounter',
                'sample_counter': 'SampleCounter',
                'connected_buffer': 'ConnectedBuffer',
                'frequency_bands': 'FrequencyBands',
                'soxel_grid': 'SoxelGrid',
                'geometry_data': 'GeometryData',
                'material_properties': 'MaterialProperties',
                'medium_properties': 'MediumProperties'
            }
            self.entities_map = {
                'sources': ['SphericalSource', 'PlanarSource'],
                'objects': ['AcousticObject'],
                'outputs': ['AmbisonicOutput', 'OmnidirectionalOutput', 'Figure8Output', 'CardioidOutput', 'HypercardioidOutput'],
                'wave_propagators': 'WavePropagator',
                'output_datas': 'OutputData',
                'trajectories': ['TrajectoryData', 'tmpTrajectoryData'],
                'collisions': ['CollisionData'],
                'forces': ['ForceData', 'ForceDataSequence'],
                'modal_vertices': 'ModalVertices',
                'score_tracks': 'ScoreTrack',
                'rigidbody_synth': 'RigidBodySynth',
                'resonance_synth': 'ResonanceSynth'
            }

            if config:
                config_obj = Config(config)
                self.register('config', config_obj)

    def register(self, entity: str, obj: Any) -> int:
        with self._local_lock:
            if entity in self.singleton_map and entity not in self._singleton:
                self._singleton[entity] = obj
            elif entity in self.entities_map.keys():
                for key in self.entities_map.keys():
                    if entity in key:
                        for sub in self.entities_map[key]:
                            if sub in str(type(obj)):
                                entities = eval(f"self._{key}")
                                idx = 0
                                if not len(entities.keys()) == 0:
                                    idx = max(list(entities.keys())) + 1
                                entities[idx] = obj
                                return idx
            return -1
               
    def get(self, entity: str = None, idx: int = None) -> dict[str, Any]:
        """Get all objects"""
        if entity is None:
            return (dict(self._singleton), dict(self._sources), dict(self._objects), 
                    dict(self._outputs), dict(self._wave_propagators), dict(self._output_datas),
                    dict(self._trajectories), dict(self._collisions), dict(self._forces),
                    dict(self._modal_vertices), dict(self._score_tracks),
                    dict(self._rigidbody_synth), dict(self._resonance_synth))
        
        if entity in self.singleton_map:
            if entity in ['geometry_data', 'material_properties', 'medium_properties']:
                return copy.deepcopy(self._singleton[entity])
            return self._singleton.get(entity)
        else:
            for key in self.entities_map.keys():
                if entity in key:
                    entities = eval(f"self._{entity}")
                    if idx is not None:
                        return entities.get(idx)
                    return dict(entitiesentities)
        return None

    def unregister(self, entity: str, idx: int = None) -> None:
        with self._local_lock:
            if entity in self.singleton_map:
                if entity in self._singleton:
                    del self._singleton[entity]
            elif idx is not None:
                for key in self.entities_map.keys():
                    if entity in key:
                        entities = eval(f"self._{entity}")
                        if idx in entities:
                            del entities[idx]
                            return
