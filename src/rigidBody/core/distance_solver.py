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
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass, field
from itertools import groupby

from ..core.entity_manager import EntityManager
from ..utils.config import Config, ObjectConfig
from ..lib.collision_data import tmpCollisionData

from ..lib.functions import _load_pose, _load_mesh

@dataclass
class DistanceSolver:
    entity_manager: EntityManager

    def compute(self, objs_idx: Tuple[int, int]) -> List[Tuple[int, float]]:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
 
        trajectory, frames  = ([] for _ in range(2))
        config_objs = [config.objects[objs_idx[0]], config.objects[objs_idx[1]]]
        if config_objs[0].static and config_objs[1].static:
            # exit: objs_idx[0] and objs_idx[1] are static
            return
        elif not config_objs[0].static or not config_objs[1].static:
            trajectories = self.entity_manager.get('trajectories')
            for idx in trajectories.keys():
                if 'TrajectoryData' in str(type(trajectories[idx])):
                    if trajectories[idx].obj_idx == config_objs[0].idx or trajectories[idx].obj_idx == config_objs[1].idx:
                        trajectory.append(trajectories[idx])
                        frames.append(trajectories[idx].get_x())
        frames = np.unique(np.sort(np.concatenate((frames[0], frames[1]))))

        # assign trajectory
        trajectory1 = trajectory[0] if trajectory[0].obj_idx == config_objs[0].idx else trajectory[1]
        trajectory2 = trajectory[1] if trajectory[1].obj_idx == config_objs[1].idx else trajectory[0]

        distances_data = []
    
        for idx in range(len(frames)):
            min_dist = self._distance(config_objs=config_objs, trajectory1=trajectory1, trajectory2=trajectory2, frame=frames[idx], sfps=sfps, sample_rate=sample_rate)
            distances_data.append(min_dist)
            if min_dist < 1.2e-3:
                print(f"distances_data between {config_objs[0].name} and {config_objs[1].name} at frame {sfps * frames[idx] / sample_rate}: ", min_dist)

#        tmp_collision_data = tmpCollisionData(obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], distances=distances_data, consec_idx=consec_idx)
#        tmp_collision_idx = len(self.entity_manager.get('collisions')) + 1
#        self.entity_manager.register('collisions', tmp_collision_data, tmp_collision_idx)
 
#    def _distance(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, trajectory1: Any, trajectory2: Any, frame: float) -> float:
#        """Get transforms from trajectory and apply to meshes before sampling and cKDTree query""" 
#        #transform1 = trajectory1.get_relative_transformation(first_frame, frame)
#        #transform2 = trajectory2.get_relative_transformation(first_frame, frame)
#        pass
#        # apply transform to meshes

    def _distance(self, config_objs: Tuple[Any, Any], trajectory1: Any, trajectory2: Any, frame: float, sfps: int, sample_rate: int) -> Tuple[float, Dict[str, Any]]:
        """
        Apply transformations to meshes based on trajectory data, verify with landmarks,
        and calculate minimum distance between transformed meshes.
    
        Parameters:
        -----------
        trajectory1 : Any
            TrajectoryData object for first object
        trajectory2 : Any
            TrajectoryData object for second object
        frame : float
            Frame number (can be fractional for subframes)
    
        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            Minimum distance between transformed meshes and verification results
        """
        frame_idx = (sfps * frame / sample_rate) - 1
        if frame_idx.is_integer():
            frame_idx = int(frame_idx)
            vertices1, normals1, faces1 = _load_mesh(config_objs[0], frame_idx)
            vertices2, normals2, faces2 = _load_mesh(config_objs[1], frame_idx)
        else:
            vertices1 = trajectory1.get_vertices(frame)
            vertices2 = trajectory2.get_vertices(frame)
            normals1 = trajectory1.get_normals(frame)
            normals2 = trajectory2.get_normals(frame)
            faces1 = trajectory1.get_faces(frame)
            faces2 = trajectory2.get_faces(frame)

        mesh1 = trimesh.Trimesh(vertices=vertices1, vertex_normals=normals1, faces=faces1)
        mesh2 = trimesh.Trimesh(vertices=vertices2, vertex_normals=normals2, faces=faces2)

        # Calculate minimum distance between transformed meshes
#        min_distance, closest_points = self._calculate_min_distance(mesh1, mesh2)
        min_distance = self._calculate_min_distance(mesh1, mesh2)
    
#        return min_distance, closest_points
        return min_distance

    def _calculate_min_distance(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, sample_density: int = 1000000) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Calculate minimum distance between two meshes using KDTree.
    
        Parameters:
        -----------
        mesh1 : trimesh.Trimesh
            First transformed mesh
        mesh2 : trimesh.Trimesh
            Second transformed mesh
        sample_density: int
            Number of points to sample per unit surface area
    
        Returns:
        --------
        Tuple[float, Dict[str, np.ndarray]]
            Minimum distance and closest points information
        """
        # Sample points from both meshes
        area1 = mesh1.area
        area2 = mesh2.area
   
        n_samples1 = max(100, int(area1 * sample_density))
        n_samples2 = max(100, int(area2 * sample_density))
   
        points1 = mesh1.sample(n_samples1)
        points2 = mesh2.sample(n_samples2)

        # Use KDTree for efficient distance calculation
        from scipy.spatial import cKDTree
    
        # Build KD-tree for mesh2 points
        tree2 = cKDTree(points2)

        # Query all points from mesh1 against mesh2
        distances, indices = tree2.query(points1)

        # Create KDTree for mesh2 vertices
#        tree2 = cKDTree(mesh2.vertices)
    
        # Query distances from mesh1 vertices to mesh2
#        distances, indices = tree2.query(mesh1.vertices)
    
        # Find minimum distance
        min_dist_idx = np.argmin(distances)
        min_distance = distances[min_dist_idx]
    
        # Get closest points
#        closest_point_mesh1 = mesh1.vertices[min_dist_idx]
#        closest_point_mesh2 = mesh2.vertices[indices[min_dist_idx]]
    
#        closest_points = {
#            'mesh1_point': closest_point_mesh1,
#            'mesh2_point': closest_point_mesh2,
#            'mesh1_vertex_idx': min_dist_idx,
#            'mesh2_vertex_idx': indices[min_dist_idx]
#        }
    
#        return min_distance, closest_points
        return min_distance
