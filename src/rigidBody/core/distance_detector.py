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
from scipy.spatial import cKDTree
from typing import List,  Tuple
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..utils.config import Config, ObjectConfig

@dataclass
class DistanceDetector:
    entity_manager: EntityManager
    frames: int = None

    def compute(self, objs_idx: List[Tuple[int, int]]) -> List[Tuple[int, float]]:
        config = self.entity_manager.get('config')
 
        config_objs = []
        config_objs.append(config.objects[objs_idx[0]])
        config_objs.append(config.objects[objs_idx[1]])
        if not config_objs[0].static or not config_objs[1].static:
            self.frames = max(len(os.listdir(config_objs[0].obj_path)), len(os.listdir(config_objs[1].obj_path)))
        elif config_objs[0].static and config_objs[1].static:
            # exit: objs_idx[0] and objs_idx[1] are static
            return

        distances_data = []
        for frame_idx in range(1, self.frames):
            mesh1 = self._load_obj(objs_idx[0], frame_idx)
            mesh2 = self._load_obj(objs_idx[1], frame_idx)

            # add dask delayed tasks
            min_dist, _ = self._min_distance(mesh1, mesh2, refine=True)
            distances_data.append([frame_idx, min_dist])
        return distances_data
 
    def _load_obj(self, obj_idx: int, frame_idx: int):
        config = self.entity_manager.get('config')
        # Load mesh
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                if config_obj.static == True:
                    for filename in os.listdir(config_obj.obj_path):
                        if filename.endswith('.obj'):
                            obj_file = f"{config_obj.obj_path}/{filename}"
                            return trimesh.load_mesh(obj_file)
                elif config_obj.static == False:
                    items = os.listdir(config_obj.obj_path)
                    obj_filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
                    obj_file = os.path.join(config_obj.obj_path, obj_filenames[frame_idx])
                    if not os.path.exists(obj_file):
                        raise FileNotFoundError(f"OBJ file not found for {obj_name}: {obj_file}")
                    return trimesh.load_mesh(obj_file)

    def _approx_distance(self, mesh1, mesh2, sample_density=1000000):
        """
        Compute minimal distance between two meshes using KD-tree sampling.
    
        Args:
            mesh1, mesh2: trimesh objects
            sample_density: number of points to sample per unit surface area
    
        Returns:
            min_distance: minimal distance between meshes
            closest_points: tuple of closest points (point1, point2)
        """
        # Sample points from both meshes
        area1 = mesh1.area
        area2 = mesh2.area
    
        n_samples1 = max(100, int(area1 * sample_density))
        n_samples2 = max(100, int(area2 * sample_density))
    
        points1 = mesh1.sample(n_samples1)
        points2 = mesh2.sample(n_samples2)
    
        # Build KD-tree for mesh2 points
        tree2 = cKDTree(points2)
    
        # Query all points from mesh1 against mesh2
        distances, indices = tree2.query(points1)
    
        # Find minimal distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        closest_point1 = points1[min_idx]
        closest_point2 = points2[indices[min_idx]]
    
        return min_distance, (closest_point1, closest_point2)

    def _min_distance(self, mesh1, mesh2, initial_sample_density=1000000, refine=True):
        """
        Hybrid approach: fast KD-tree sampling followed by exact computation
        in the region of interest.
        """
        # Step 1: Fast approximate search
        approx_dist, (approx_p1, approx_p2) = self._approx_distance(
            mesh1, mesh2, sample_density=initial_sample_density
        )
    
        if not refine:
            return approx_dist, (approx_p1, approx_p2)
    
        # Step 2: Refine search in the neighborhood
        # Create bounding boxes around approximate closest points
        search_radius = approx_dist * 2.0  # Search in twice the approximate distance
    
        # Find vertices within search radius
        vertices1 = mesh1.vertices
        vertices2 = mesh2.vertices
    
        # Find vertices near the approximate closest points
        mask1 = np.linalg.norm(vertices1 - approx_p1, axis=1) < search_radius
        mask2 = np.linalg.norm(vertices2 - approx_p2, axis=1) < search_radius
    
        if np.any(mask1) and np.any(mask2):
            # Build KD-tree for nearby vertices
            nearby_vertices2 = vertices2[mask2]
            tree2 = cKDTree(nearby_vertices2)
        
            # Query nearby vertices from mesh1
            distances, indices = tree2.query(vertices1[mask1])
        
            # Find minimal distance
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            closest_point1 = vertices1[mask1][min_idx]
            closest_point2 = nearby_vertices2[indices[min_idx]]
        
            return min_distance, (closest_point1, closest_point2)
    
        return approx_dist, (approx_p1, approx_p2)
