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
import sys
import json
import copy
import pickle
import numpy as np
import tempfile
import shutil
from typing import List, Tuple, Any, Dict, Optional
from ..utils.config import Config

class EntityManager:
    _instance = None
    _initialized = False
    
    # Lock file management
    _lock_dir = None
    _lock_files = {}
    
    # Catalog directory
    _catalog_dir = None
    
    def __init__(self, config: str, catalog_dir: str = None, lock_dir: str = None):
        if not self._initialized:
            # Setup catalog directory
            if catalog_dir is None:
                # Use a default location relative to the config cache path
                config_obj = Config(config)
                self._catalog_dir = os.path.join(
                    config_obj.system.cache_path, 
                    "entity_catalog"
                )
            else:
                self._catalog_dir = catalog_dir
            
            # Setup lock directory
            if lock_dir is None:
                self._lock_dir = os.path.join(
                    self._catalog_dir, 
                    "locks"
                )
            else:
                self._lock_dir = lock_dir
            
            # Create directories
            os.makedirs(self._catalog_dir, exist_ok=True)
            os.makedirs(self._lock_dir, exist_ok=True)
            
            # Initialize catalog structure
            self._init_catalog()
            
            # Initialize singleton storage
            self._singleton = {}
            
            # Define entity maps
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
            
            # Register config
            config_obj = Config(config)
            self.register('config', config_obj)
            
            self._initialized = True
    
    def _init_catalog(self):
        """Initialize catalog directory structure."""
        # Create subdirectories for each entity type
        for entity_type in list(self.singleton_map.keys()) + list(self.entities_map.keys()):
            entity_dir = os.path.join(self._catalog_dir, entity_type)
            os.makedirs(entity_dir, exist_ok=True)
    
    def _get_lock_file(self, entity_type: str, entity_id: Any = None) -> str:
        """
        Get the lock file path for a specific entity.
        
        Args:
            entity_type: Type of entity (e.g., 'config', 'trajectories')
            entity_id: Optional entity ID for indexed entities
            
        Returns:
            Path to the lock file
        """
        if entity_id is not None:
            lock_name = f"{entity_type}_{entity_id}.lock"
        else:
            lock_name = f"{entity_type}.lock"
        
        return os.path.join(self._lock_dir, lock_name)
    
    def _acquire_lock(self, lock_file: str, exclusive: bool = True):
        """
        Acquire a file lock.
        
        Args:
            lock_file: Path to the lock file
            exclusive: If True, acquire exclusive lock; otherwise shared lock
            
        Returns:
            Lock file descriptor or raises exception
        """
        # Ensure lock file exists
        if not os.path.exists(lock_file):
            try:
                with open(lock_file, 'w') as f:
                    f.write('')
            except:
                pass
        
        try:
            fd = os.open(lock_file, os.O_RDWR | os.O_CREAT)
            
            if sys.platform.startswith('win'):
                # Windows: use msvcrt
                import msvcrt
                if exclusive:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # Windows doesn't distinguish well
            else:
                # Unix: use fcntl
                import fcntl
                if exclusive:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            
            return fd
            
        except (IOError, BlockingIOError, PermissionError) as e:
            if fd:
                os.close(fd)
            raise RuntimeError(f"Could not acquire lock on {lock_file}: {e}")
    
    def _release_lock(self, fd):
        """Release a file lock."""
        if fd is not None:
            try:
                if sys.platform.startswith('win'):
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                
                os.close(fd)
            except:
                pass
    
    def _get_entity_path(self, entity_type: str, entity_id: Any = None) -> str:
        """
        Get the file path for a specific entity in the catalog.
        
        Args:
            entity_type: Type of entity
            entity_id: Optional entity ID
            
        Returns:
            Path to the entity file
        """
        entity_dir = os.path.join(self._catalog_dir, entity_type)
        
        if entity_id is not None:
            return os.path.join(entity_dir, f"{entity_id}.pkl")
        else:
            return os.path.join(entity_dir, "data.pkl")
    
    def _serialize_object(self, obj: Any) -> bytes:
        """
        Serialize an object to bytes for storage.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized bytes
        """
        # Handle special types
        if isinstance(obj, Config):
            # Config objects need special handling
            return pickle.dumps({
                '_type': 'Config',
                'data': obj.data
            })
        elif hasattr(obj, '__dict__'):
            return pickle.dumps(obj)
        else:
            return pickle.dumps(obj)
    
    def _deserialize_object(self, data: bytes) -> Any:
        """
        Deserialize an object from bytes.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized object
        """
        try:
            obj = pickle.loads(data)
            
            # Handle special types
            if isinstance(obj, dict) and obj.get('_type') == 'Config':
                config = Config.__new__(Config)
                config.data = obj['data']
                return config
            
            return obj
        except:
            return data
    
    def _get_next_id(self, entity_type: str) -> int:
        """
        Get the next available ID for an entity type.
        
        Args:
            entity_type: Type of entity
            
        Returns:
            Next available ID
        """
        lock_file = self._get_lock_file(entity_type, 'counter')
        fd = None
        
        try:
            fd = self._acquire_lock(lock_file, exclusive=True)
            
            counter_file = os.path.join(self._catalog_dir, entity_type, '_counter.txt')
            
            if os.path.exists(counter_file):
                with open(counter_file, 'r') as f:
                    counter = int(f.read().strip())
            else:
                counter = 0
            
            counter += 1
            
            with open(counter_file, 'w') as f:
                f.write(str(counter))
            
            return counter - 1  # Return the previous counter value
            
        finally:
            if fd is not None:
                self._release_lock(fd)
    
    def register(self, entity: str, obj: Any) -> int:
        """
        Register an object in the catalog.
        
        Args:
            entity: Entity type
            obj: Object to register
            
        Returns:
            Entity ID (for indexed entities) or None
        """
        # Check if it's a singleton
        if entity in self.singleton_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=True)
                
                # Store in local cache
                self._singleton[entity] = obj
                
                # Store in catalog
                entity_path = self._get_entity_path(entity)
                serialized = self._serialize_object(obj)
                
                with open(entity_path, 'wb') as f:
                    f.write(serialized)
                
                return None
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        # Check if it's an indexed entity
        elif entity in self.entities_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=True)
                
                # Get next ID
                entity_id = self._get_next_id(entity)
                
                # Store in catalog
                entity_path = self._get_entity_path(entity, entity_id)
                serialized = self._serialize_object(obj)
                
                with open(entity_path, 'wb') as f:
                    f.write(serialized)
                
                return entity_id
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        else:
            raise ValueError(f"Unknown entity type: {entity}")
    
    def get(self, entity: str = None, idx: int = None) -> dict[str, Any]:
        """
        Get objects from the catalog.
        
        Args:
            entity: Entity type to retrieve
            idx: Optional entity ID for indexed entities
            
        Returns:
            Requested object(s)
        """
        if entity is None:
            # Return all entities
            result = {}
            
            # Get singletons
            for singleton_type in self.singleton_map.keys():
                result[singleton_type] = self.get(singleton_type)
            
            # Get indexed entities
            for entity_type in self.entities_map.keys():
                result[entity_type] = self.get(entity_type)
            
            return result
        
        # Check if it's a singleton
        if entity in self.singleton_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=False)
                
                # Check local cache first
                if entity in self._singleton:
                    obj = self._singleton[entity]
                    
                    # For certain types, return deep copy
                    if entity in ['geometry_data', 'material_properties', 'medium_properties']:
                        return copy.deepcopy(obj)
                    return obj
                
                # Try to load from catalog
                entity_path = self._get_entity_path(entity)
                if os.path.exists(entity_path):
                    with open(entity_path, 'rb') as f:
                        data = f.read()
                    
                    obj = self._deserialize_object(data)
                    self._singleton[entity] = obj
                    
                    if entity in ['geometry_data', 'material_properties', 'medium_properties']:
                        return copy.deepcopy(obj)
                    return obj
                
                return None
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        # Check if it's an indexed entity
        elif entity in self.entities_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=False)
                
                entity_dir = os.path.join(self._catalog_dir, entity)
                
                if idx is not None:
                    # Return specific entity
                    entity_path = self._get_entity_path(entity, idx)
                    if os.path.exists(entity_path):
                        with open(entity_path, 'rb') as f:
                            data = f.read()
                        return self._deserialize_object(data)
                    return None
                else:
                    # Return all entities of this type
                    result = {}
                    
                    if os.path.exists(entity_dir):
                        for filename in os.listdir(entity_dir):
                            if filename.endswith('.pkl') and filename != '_counter.txt':
                                try:
                                    entity_id = int(filename.replace('.pkl', ''))
                                    entity_path = os.path.join(entity_dir, filename)
                                    
                                    with open(entity_path, 'rb') as f:
                                        data = f.read()
                                    
                                    result[entity_id] = self._deserialize_object(data)
                                except (ValueError, IOError):
                                    continue
                    
                    return result
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        else:
            raise ValueError(f"Unknown entity type: {entity}")
    
    def unregister(self, entity: str, idx: int = None) -> None:
        """
        Unregister an object from the catalog.
        
        Args:
            entity: Entity type
            idx: Optional entity ID for indexed entities
        """
        # Check if it's a singleton
        if entity in self.singleton_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=True)
                
                # Remove from local cache
                if entity in self._singleton:
                    del self._singleton[entity]
                
                # Remove from catalog
                entity_path = self._get_entity_path(entity)
                if os.path.exists(entity_path):
                    os.remove(entity_path)
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        # Check if it's an indexed entity
        elif entity in self.entities_map:
            lock_file = self._get_lock_file(entity)
            fd = None
            
            try:
                fd = self._acquire_lock(lock_file, exclusive=True)
                
                if idx is not None:
                    entity_path = self._get_entity_path(entity, idx)
                    if os.path.exists(entity_path):
                        os.remove(entity_path)
                else:
                    # Remove all entities of this type
                    entity_dir = os.path.join(self._catalog_dir, entity)
                    if os.path.exists(entity_dir):
                        shutil.rmtree(entity_dir)
                        os.makedirs(entity_dir, exist_ok=True)
                
            finally:
                if fd is not None:
                    self._release_lock(fd)
        
        else:
            raise ValueError(f"Unknown entity type: {entity}")
    
    def clear(self):
        """Clear all registered entities."""
        # Acquire locks for all entity types
        fds = []
        try:
            for entity_type in list(self.singleton_map.keys()) + list(self.entities_map.keys()):
                lock_file = self._get_lock_file(entity_type)
                try:
                    fd = self._acquire_lock(lock_file, exclusive=True)
                    fds.append((entity_type, fd))
                except:
                    pass
            
            # Clear local cache
            self._singleton = {}
            
            # Clear catalog
            if os.path.exists(self._catalog_dir):
                shutil.rmtree(self._catalog_dir)
                os.makedirs(self._catalog_dir, exist_ok=True)
                self._init_catalog()
            
        finally:
            for _, fd in fds:
                self._release_lock(fd)
    
    def __del__(self):
        """Cleanup on deletion."""
        # Release any held locks
        for fd in self._lock_files.values():
            self._release_lock(fd)
        self._lock_files.clear()

