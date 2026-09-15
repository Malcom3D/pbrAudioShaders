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
import trimesh
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

from pbrAudioCommon import EntityManager, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import TrajectoryData

@dataclass
class SurfaceVoxelObject:
    """
    Manages a surface voxel grid for an object over its trajectory.
    """
    entity_manager: EntityManager
    obj_idx: int
    voxel_size: float
    trajectory: TrajectoryData

    # Store the voxelized surface for each frame
    voxelized_frames: Dict[float, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.config_obj = next((obj for obj in config.objects if obj.idx == self.obj_idx), None)
        if not self.config_obj:
            raise ValueError(f"Object config for idx {self.obj_idx} not found.")

    def compute(self):
        """Voxelize the object's surface for each frame in its trajectory."""
        frames = self.trajectory.get_x()
        if self.trajectory.static:
            frames = np.array([0])

        for frame in frames:
            try:
                vertices = self.trajectory.get_vertices(frame)
                faces = self.trajectory.get_faces()
                if vertices.size == 0 or faces.size == 0:
                    continue

                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                if not mesh.is_volume:
                    continue

                # Voxelize the surface
                voxel_grid = mesh.voxelized(pitch=self.voxel_size).fill()
                
                # Get voxel centers and indices
                voxel_indices = np.argwhere(voxel_grid.matrix)
                voxel_centers = voxel_grid.indices_to_points(voxel_indices)

                self.voxelized_frames[frame] = {
                    'indices': voxel_indices,
                    'centers': voxel_centers,
                    'tree': cKDTree(voxel_centers) if len(voxel_centers) > 0 else None
                }
            except Exception as e:
                debug_print(f"Failed to voxelize {self.config_obj.name} at frame {frame}: {e}")
                self.voxelized_frames[frame] = {'indices': np.array([]), 'centers': np.array([]), 'tree': None}

    def query_collision(self, frame: float, particle_position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Query if a particle position collides with the object's surface at a given frame.
        Returns the index of the closest voxel and the distance.
        """
        # Find the closest available frame data
        if self.trajectory.static:
            frame_key = 0
        else:
            available_frames = np.array(list(self.voxelized_frames.keys()))
            if available_frames.size == 0:
                return np.array([]), float('inf')
            frame_key = available_frames[np.argmin(np.abs(available_frames - frame))]

        frame_data = self.voxelized_frames.get(frame_key)
        if not frame_data or frame_data['tree'] is None:
            return np.array([]), float('inf')

        # Query the KDTree for the nearest voxel
        distance, index = frame_data['tree'].query(particle_position, k=1)
        
        if distance < self.voxel_size * 1.5: # Collision threshold
            return frame_data['indices'][index], distance
        
        return np.array([]), distance

    def save(self, filepath: str):
        """Saves the voxelized data to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # We need to remove KDTree objects before pickling
        save_data = {}
        for frame, data in self.voxelized_frames.items():
            save_data[frame] = {
                'indices': data['indices'],
                'centers': data['centers']
            }
        np.savez_compressed(filepath, **save_data)
        debug_print(f"Saved surface voxel data for {self.config_obj.name} to {filepath}")

    def load(self, filepath: str):
        """Loads voxelized data and rebuilds KDTrees."""
        if not os.path.exists(filepath):
            debug_print(f"No surface voxel data found for {self.config_obj.name} at {filepath}")
            return
        loaded_data = np.load(filepath, allow_pickle=True)
        self.voxelized_frames = {}
        for frame, data in loaded_data.items():
            indices = data.item()['indices']
            centers = data.item()['centers']
            self.voxelized_frames[frame] = {
                'indices': indices,
                'centers': centers,
                'tree': cKDTree(centers) if len(centers) > 0 else None
            }
        debug_print(f"Loaded surface voxel data for {self.config_obj.name} from {filepath}")

