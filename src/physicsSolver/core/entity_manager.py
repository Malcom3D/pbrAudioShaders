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
import fcntl
import pickle
import tempfile
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from ..utils.config import Config

class EntityManager:
    _instance = None
    _initialized = False
    
    def __init__(self, config: str, lock_dir: Optional[str] = None):
        if not self._initialized:
            # Initialize in-memory storage
            self._sources = {}
            self._objects = {}
            self._outputs = {}
            self._output_datas = {}
            self._wave_propagators = {}
            self._layer_managers = {}
            self._trajectories = {}
            self._collisions = {}
            self._forces = {}
            self._modal_vertices = {}
            self._score_tracks = {}
            self._rigidbody_synth = {}
            self._resonance_synth = {}
            self._singleton = {}
            self._initialized = True

            # Create lock directory
            if lock_dir is None:
                lock_dir = tempfile.mkdtemp(prefix='entity_manager_locks_')
            
            self.lock_dir = lock_dir
            os.makedirs(self.lock_dir, exist_ok=True)
            
            # Lock timeout and retry settings
            self.lock_timeout = 30  # seconds
            self.lock_retry_interval = 0.01  # 10ms between retries
            
            # Storage directory for persisted data
            self.data_dir_dir = os.path.join(self.lock_dir, 'data')
            os.makedirs(self.data_dir, exist_ok=True)

            self.sigleton_map = {
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

            config = Config(config)
            self.register('config', config)

    def _get_lock_path(self, entity: str, idx: Optional[int] = None) -> str:
        """Get the path for a lock file."""
        if idx is not None:
            return os.path.join(self.lock_dir, f"{entity}_{idx}.lock")
        return os.path.join(self.lock_dir, f"{entity}.lock")

    def _get_data_path(self, entity: str, idx: Optional[int] = None) -> str:
        """Get the path for a data file."""
        if idx is not None:
            return os.path.join(self.data_dir, f"{entity}_{idx}.pkl")
        return os.path.join(self.data_dir, f"{entity}.pkl")

    def _acquire_lock(self, lock lock_path: str, shared: bool = False) -> Optional[int]:
        """
        Acquire a file lock with retry logic.
        
        Args:
            lock_path: Path to the lock file
            shared: If True, acquire a shared lock (read). If False, exclusive lock (write).
            
        Returns:
            File descriptor if lock acquired, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.lock_timeout:
            try:
                # Create lock file if it doesn't exist
                if not os.path.exists(lock_path):
                    # Use a temporary file to avoid race conditions
                    temp_path = lock_path + '.tmp'
                    fd = os.open(temp_path, os.O_CREAT | os.O_WRONLY, 0o644)
                    os.close(fd)
                    os.rename(temp_path, lock_path)
                
                fd = os.open(lock_path, os.O_RDWR)
                
                if shared:
                    lock_type = fcntl.LOCK_SH
                else:
                    lock_type = fcntl.LOCK_EX
                
                # Try to acquire lock with LOCK_NB (non-blocking)
                try:
                    fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
                    return fd
                except (IOError, OSError) as e:
                    if e.errno == 11:  # Resource temporarily unavailable
                        os.close(fd)
                        time.sleep(self.lock_retry_interval)
                        continue
                    else:
                        os.close(fd)
                        raise
                        
            except (IOError, OSError) as e:
                if e.errno == 11:  # Resource temporarily unavailable
                    time.sleep(self.lock_retry_interval)
                    continue
                else:
                    raise
        
        raise TimeoutError(f"Could not acquire lock for {lock_path} after {self.lock_timeout} seconds")

    def _release_lock(self, fd: int):
        """Release a file lock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except (IOError, OSError):
            pass

    def _save_to_file(self, entity: str, data: Any, idx: Optional[int] = None):
        """Save data to a file with proper locking."""
        lock_path = self._get_lock_path(entity, idx)
        data_path = self._get_data_path(entity, idx)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        fd = None
        try:
            fd = self._acquire_lock(lock_path, shared=False)
            
            # Save data using pickle
            with open(data_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        finally:
            if fd is not None:
                self._release_lock(fd)

    def _load_from_file(self, entity: str, idx: Optional[int] = None) -> Any:
        """Load data from a file with proper locking."""
        lock_path = self._get_lock_path(entity, idx)
        data_path = self._get_data_path(entity, idx)
        
        if not os.path.exists(data_path):
            return None
        
        fd = None
        try:
            fd = self._acquire_lock(lock_path, shared=True)
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            return data
            
        finally:
            if fd is not None:
                self._release_lock(fd)

    def _get_next_index(self, entity: str) -> int:
        """Get the next available index for an entity type."""
        lock_path = self._get_lock_path(f"{entity}_counter")
        counter_path = self._get_data_path(f"{entity}_counter")
        
        fd = None
        try:
            fd = self._acquire_lock(lock_path, shared=False)
            
            # Load current counter
            counter = 0
            if os.path.exists(counter_path):
                with open(counter_path, 'rb') as f:
                    counter = pickle.load(f)
            
            # Increment and save

            counter += 1
            with open(counter_path, 'wb') as f:
                pickle.dump(counter, f)
                
            return counter - 1  # Return the index before increment
            
        finally:
            if fd is not None:
                self._release_lock(fd)

    def register(self, entity: str, obj: Any) -> int:
        """
        Register an object with the entity manager.
        
        Args:
            entity: Entity type name
            obj: Object to register
            
        Returns:
            Index of the registered object (if applicable)
        """
        if entity in self.sigleton_map:
            # Singleton - save to file
            self._save_to_file(entity, obj)
            self._singleton[entity] = obj
            return 0
            
        elif entity in self.entities_map.keys():
            # Get next available index
            idx = self._get_next_index(entity)
            
            # Save to file
            self._save_to_file(entity, obj, idx)
            
            # Update in-memory cache
            entities = eval(f"self._{entity}")
            entities[idx] = obj
            
            return idx

    def get(self, entity: str = None, idx: int = None) -> dict[str, Any]:
        """
        Get registered objects.
        
        Args:
            entity: Entity type to get (None returns all)
            idx: Specific index to get (None returns all of that type)
            
        Returns:
            Requested object(s)
        """
        if entity is None:
            return (self._singleton, self._sources, self._objects, self._outputs,
                    self._wave_propagators, self._output_datas, self._trajectories,
                    self._collisions, self._forces, self._modal_vertices,
                    self._score_tracks, self._rigidbody_synth, self._resonance_synth)
        
        # Check singletons
        for key in self.sigleton_map.keys():
            if entity in key:
                if entity in self._singleton:
                    return self._singleton[entity]
                
                # Try to load from file
                data = self._load_from_file(entity)
                if data is not None:
                    self._singleton[entity] = data
                    return data
                return None
        
        # Check entities
        for key in self.entities_map.keys():
            if entity in key:
                entities = eval(f"self._{entity}")
                
                if idx is not None:
                    # Get specific index
                    if idx in entities:
                        return entities[idx]
                    
                    # Try to load from file
                    data = self._load_from_file(entity, idx)
                    if data is not None:
                        entities[idx] = data
                        return data
                    return None
                else:
                    # Get all entities of this type
                    # Try to load any missing from files
                    data_dir = self.data_dir
                    if os.path.exists(data_dir):
                        for filename in os.listdir(data_dir):
                            if filename.startswith(f"{entity}_") and filename.endswith(".pkl"):
                                try:
                                    file_idx = int(filename.split('_')[1].split('.')[0])
                                    if file_idx not in entities:
                                        data = self._load_from_file(entity, file_idx)
                                        if data is not None:
                                            entities[file_idx] = data
                                except (ValueError, IndexError):
                                    continue
                    
                    return entities

    def unregister(self, entity: str, idx: int = None) -> None:
        """
        Unregister an object.
        
        Args:
            entity: Entity type to unregister
            idx: Specific index to unregister (None for all of that type)
        """
        # Check singletons
        for key in self.sigleton_map.keys():
            if entity in key:
                if entity in self._singleton:
                    del self._singleton[entity]
                    
                # Remove data file
                               data_path = self._get_data_path(entity)
                if os.path.exists(data_path):
                    os.remove(data_path)
                    
                # Remove lock file
                lock_path = self._get_lock_path(entity)
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                return
        
        # Check entities
        for key in self.entities_map.keys():
            if entity in key:
                entities = eval(f"self._{entity}")
                
                if idx is not None:
                    # Remove specific index
                    if idx in entities:
                        del entities[idx]
                    
                    # Remove data file
                    data_path = self._get_data_path(entity, idx)
                    if os.path.exists(data_path):
                        os.remove(data_path)
                    
                    # Remove lock file
                    lock_path = self._get_lock_path(entity, idx)
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                else:
                    # Remove all entities of this type
                    for existing_idx in list(entities.keys()):
                        del entities[existing_idx]
                    
                    # Remove all data files
                    data_dir = self.data_dir
                    if os.path.exists(data_dir):
                        for filename in os.listdir(data_dir):
                            if filename.startswith(f"{entity}_"):
                                file_path = os.path.join(data_dir, filename)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                    
                    # Remove all lock files
                    lock_dir = self.lock_dir
                    if os.path.exists(lock_dir):
                        for filename in os.listdir(lock_dir):
                            if filename.startswith(f"{entity}_"):
                                file_path = os.path.join(lock_dir, filename)
                                if os.path.exists(file_path):
                                    os.remove(file_path)

    def clear_all(self):
        """Clear all registered objects and data files."""
        # Clear in-memory storage
        self._sources.clear()
        self._objects.clear()
        self._outputs.clear()
        self._output_datas.clear()
        self._wave_propagators.clear()
        self._layer_managers.clear()
        self._trajectories.clear()
        self._collisions.clear()
        self._forces.clear()
        self._modal_vertices.clear()
        self._score_tracks.clear()
        self._rigidbody_synth.clear()
        self._resonance_synth.clear()
        self._singleton.clear()
        
        # Remove all data and lock files
        import shutil
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        if os.path.exists(self.lock_dir):
            shutil.rmtree(self.lock_dir)
        
        # Recreate directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)

