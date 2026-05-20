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
from typing import List, Tuple, Any, Dict, Optional, Callable
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
            self.data_dir = os.path.join(self.lock_dir, 'data')
            os.makedirs(self.data_dir, exist_ok=True)

            # Track which processes have modified which entities
            self._dirty_entities = {}  # {entity_type: {idx: set(process_ids)}}
            
            # Callbacks for entity modifications
            self._modification_callbacks = {}  # {entity_type: {idx: [callbacks]}}

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
            return os.path.join(self.l.lock_dir, f"{entity}_{idx}.lock")
        return os.path.join(self.lock_dir, f"{entity}.lock")

    def _get_data_path(self, entity: str, idx: Optional[int] = None) -> str:
        """Get the path for a data file."""
        if idx is not None:
            return os.path.join(self.data_dir, f"{entity}_{idx}.pkl")
        return os.path.join(self.data_dir, f"{entity}.pkl")

    def _acquire_lock(self, lock_path: str, shared: bool = False) -> Optional[int]:
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

    def update_entity(self, entity: str, idx: int, update_func: Callable[[Any], Any]) -> Any:
        """
        Atomically update an entity using a callback function.
        This is the recommended way to modify entities in a multiprocess environment.
        
        Args:
            entity: Entity type name
            idx: Index of the entity to update
            update_func: Function that takes the current entity and returns the modified entity
            
        Returns:
            The modified entity
            
        Example:
            def add_rotation(traj):
                traj.add_data('rotation', rotation_data)
                return traj
            
            entity_manager.update_entity('trajectories', 0, add_rotation)
        """
        # Get lock path for this entity
        lock_path = self._get_lock_path(entity, idx)
        data_path = self._get_data_path(entity, idx)
        
        fd = None
        try:
            # Acquire exclusive lock for writing
            fd = self._acquire_lock(lock_path, shared=False)
            
            # Load current state from file
            current_data = None
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    current_data = pickle.load(f)
            
            # Apply the update function
            modified_data = update_func(current_data)
            
            # Save modified data
            with open(data_path, 'wb') as f:
                pickle.dump(modified_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update in-memory cache
            entities = self._get_entities_dict(entity)
            if entities is not None:
                entities[idx] = modified_data
            
            # Track dirty entities
            if entity not in self._dirty_entities:
                self._dirty_entities[entity] = {}
            if idx not in self._dirty_entities[entity]:
                self._dirty_entities[entity][idx] = set()
            self._dirty_entities[entity][idx].add(os.getpid())
            
            # Execute modification callbacks
            self._execute_modification_callbacks(entity, idx, modified_data)
            
            return modified_data
            
        finally:
            if fd is not None:
                self._release_lock(fd)

    def _get_entities_dict(self, entity: str) -> Optional[Dict]:
        """Get the in-memory dictionary for an entity type."""
        entity_map = {
            'sources': self._sources,
            'objects': self._objects,
            'outputs': self._outputs,
            'output_datas': self._output_datas,
            'wave_propagators': self._wave_propagators,
            'layer_managers': self._layer_managers,
            'trajectories': self._trajectories,
            'collisions': self._collisions,
            'forces': self._forces,
            'modal_vertices': self._modal_vertices,
            'score_tracks': self._score_tracks,
            'rigidbody_synth': self._rigidbody_synth,
            'resonance_synth': self._resonance_synth,
        }
        
        for key in entity_map:
            if entity in key:
                return entity_map[key]
        return None

    def register_modification_callback(self, entity: str, idx: int, callback: Callable[[Any], None]):
        """
        Register a callback that will be called when an entity is modified.
        
        Args:
            entity: Entity type name
            idx: Index of the entity
            callback: Function that takes the modified entity as argument
        """
        if entity not in self._modification_callbacks:
            self._modification_callbacks[entity] = {}
        if idx not in self._modification_callbacks[entity]:
            self._modification_callbacks[entity][idx] = []
        self._modification_callbacks[entity][idx].append(callback)

    def _execute_modification_callbacks(self, entity: str, idx: int, data: Any):
        """Execute all registered callbacks for an entity modification."""
        if entity in self._modification_callbacks:
            if idx in self._modification_callbacks[entity]:
                for callback in self._modification_callbacks[entity][idx]:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"ErrorError in modification callback: {e}")

    def batch_update_entities(self, updates: List[Tuple[str, int, Callable]]) -> List[Any]:
        """
        Perform multiple entity updates atomically.
        Useful when multiple entities need to be updated consistently.
        
        Args:
            updates: List of (entity, idx, update_func) tuples
            
        Returns:
            List of modified entities
        """
        # Sort by entity type and idx to avoid deadlocks
        updates.sort(key=lambda x: (x[0], x[1]))
        
        results = []
        try:
            for entity, idx, update_func in updates:
                result = self.update_entity(entity, idx, update_func)
                results.append(result)
        except Exception as e:
            print(f"Batch update failed: {e}")
            raise
        
        return results

    def get_dirty_entities(self, entity: Optional[str] = None) -> Dict:
        """
        Get entities that have been modified by multiple processes.
        
        Args:
            entity: Optional entity type to filter by
            
        Returns:
            Dictionary of dirty entities
        """
        if entity:
            return self._dirty_entities.get(entity, {})
        return self._dirty_entities

    def clear_dirty_flags(self, entity: Optional[str] = None, idx: Optional[int] = None):
        """
        Clear dirty flags for entities.
        
        Args:
            entity: Optional entity type to clear
            idx: Optional specific index to clear
        """
        if entity and idx is not None:
            if entity in self._dirty_entities and idx in self._dirty_entities[entity]:
                del self._dirty_entities[entity][idx]
        elif entity:
            if entity in self._dirty_entities:
                del self._dirty_entities[entity]
        else:
            self._dirty_entities.clear()

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

    def unregister(self, entity: str, idx: int = None) -> None None:
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
                    
                    # Remove dirty flags
                    self.clear_dirty_flags(entity, idx)
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
                    
                    # Clear dirty flags
                    self.clear_dirty_flags(entity)

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
        self._dirty_entities.clear()
        self._modification_callbacks.clear()
        
        # Remove all data and lock files
        import shutil
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        if os.path.exists(self.lock_dir):
            shutil.rmtree(self.lock_dir)
        
        # Recreate directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)
