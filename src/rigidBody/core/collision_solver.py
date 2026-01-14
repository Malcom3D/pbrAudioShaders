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
from scipy.integrate import solve_ivp
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from itertools import groupby

from ..core.entity_manager import EntityManager
from ..lib.collision_data import CollisionData, CollisionArea

@dataclass
class CollisionSolver:
    entity_manager: EntityManager

    def compute(self, objs_idx: Tuple[int, int]) -> None:
        """ 
        Calculate normal, tangential, and total impact forces and velocities
        based on the stochastic physically-based model.
        """
        config = self.entity_manager.get('config')
        collision_margin = config.system.collision_margin
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        trajectory, collisions, frames  = ([] for _ in range(3))
        config_objs = [config.objects[objs_idx[0]], config.objects[objs_idx[1]]]
        if config_objs[0].static and config_objs[1].static:
            # exit: objs_idx[0] and objs_idx[1] are static
            return
        elif not config_objs[0].static or not config_objs[1].static:
            collisions_data = self.entity_manager.get('collisions')
            trajectories = self.entity_manager.get('trajectories')
            for t_idx in trajectories.keys():
                if 'TrajectoryData' in str(type(trajectories[t_idx])):
                    if trajectories[t_idx].obj_idx [config_objs[0].idx, config_objs[1].idx]:
                        trajectory.append(trajectories[t_idx])
                        frames.append(trajectories[t_idx].get_x()) 
            for c_idx in collisions_data.keys():
                if collisions_data[c_idx].obj1_idx in [config_objs[0].idx, config_objs[1].idx] and collisions_data[c_idx].obj2_idx in [config_objs[0].idx, config_objs[1].idx]:
                    collisions.append(collisions_data[c_idx])

        frames = np.unique(np.sort(np.concatenate((frames[0], frames[1]))))

        # assign trajectory
        trajectory1 = trajectory[0] if trajectory[0].obj_idx == config_objs[0].idx else trajectory[1]
        trajectory2 = trajectory[1] if trajectory[1].obj_idx == config_objs[1].idx else trajectory[0]

        for idx in range(len(frames) -1):
            for collision_idx in range(len(collisions)):
                if collisions[collision_idx].frame == frames[idx]:
                    frame = int(frames[idx] - ((frames[idx] - frames[idx -1]) / 2))
                    samples = int(frames[idx] + ((frames[idx + 1] - frames[idx]) / 2)) - frame
                    collision_area = self._facing_face(config_objs=config_objs, trajectory1=trajectory1, trajectory2=trajectory2, frame=frame, samples=samples, collision_margin=collision_margin)
                    collisions[collision_idx].add_area('collision_area', collision_area)

    def _facing_face(self, config_objs: Any, trajectory1: Any, trajectory2: Any, frame: int, samples: int, collision_margin: float) -> List[Tuple[int, Tuple[CollisionArea, CollisionArea]]]:
        collision_area = []
        for sample_idx in range(frame, frame + samples):
            mesh1_vertices = trajectory1.get_vertices(sample_idx)
            mesh1_faces = trajectory1.get_faces(sample_idx)
            mesh1_normals = trajectory1.get_normals(sample_idx)
            face1_normals = []
            for face_idx in range(len(mesh1_faces)):
                face = mesh1_faces[face_idx]
                vertex_normals = mesh1_normals[face] 
                face_normal = self._face_normal_from_vertex_normals(vertex_normals)
                face1_normals.append(face_normal)
            face1_normals = np.array(face1_normals)
            mesh2_vertices = trajectory2.get_vertices(sample_idx)
            mesh2_faces = trajectory2.get_faces(sample_idx)
            mesh2_normals = trajectory2.get_normals(sample_idx)
            face2_normals = []
            for face_idx in range(len(mesh2_faces)):
                face = mesh2_faces[face_idx] 
                vertex_normals = mesh2_normals[face]
                face_normal = self._face_normal_from_vertex_normals(vertex_normals)
                face2_normals.append(face_normal)
            face2_normals = np.array(face2_normals)

            mesh1_faces_idx, mesh2_faces_idx = self._find_mutual_facing_faces(mesh1_vertices=mesh1_vertices, mesh1_faces=mesh1_faces, mesh1_normals=face1_normals, mesh2_vertices=mesh2_vertices, mesh2_faces=mesh2_faces, mesh2_normals=face2_normals, threshold_angle=90, distance_threshold=collision_margin)
            if not len(mesh1_faces_idx) == 0 or not len(mesh2_faces_idx) == 0:
                print(f"facing faces between {config_objs[0].name} and {config_objs[1].name} at frame {sample_idx}: {len(mesh1_faces_idx)} {len(mesh2_faces_idx)}")
                collision_area1 = CollisionArea(obj_idx=trajectory1.obj_idx, faces_idx=mesh1_faces_idx)
                collision_area2 = CollisionArea(obj_idx=trajectory2.obj_idx, faces_idx=mesh2_faces_idx)
                collision_area.append([sample_idx, [collision_area1, collision_area2]])

        return collision_area

    def _find_facing_faces(self, mesh1_vertices, mesh1_faces, mesh1_normals, mesh2_vertices, mesh2_faces, mesh2_normals, threshold_angle=90.0, distance_threshold=None) -> np.ndarray:
        """
        Find faces in mesh1 that are facing mesh2.
    
        Parameters:
        -----------
        mesh1_vertices : numpy array of shape (n1, 3)
            Vertex positions of mesh1 in world coordinates
        mesh1_faces : numpy array of shape (m1, 3)
            Face indices for mesh1 (triangles)
        mesh1_normals : numpy array of shape (m1, 3)
            Face normals for mesh1 (normalized)
    
        mesh2_vertices : numpy array of shape (n2, 3)
            Vertex positions of mesh2 in world coordinates
        mesh2_faces : numpy array of shape (m2, 3)
            Face indices for mesh2 (triangles)
        mesh2_normals : numpy array of shape (m2, 3)
            Face normals for mesh2 (normalized)
    
        threshold_angle : float, optional
            Maximum angle (in degrees) between face normal and direction vector
            to consider as "facing" (default: 90°)
    
        distance_threshold : float, optional
            Maximum distance between face centers to consider (default: None = all distances)
    
        Returns:
        --------
        facing_indices : numpy array
            Indices of faces in mesh1 that are facing mesh2
        """
    
        # Convert threshold angle to radians and compute cosine threshold
        cos_threshold = np.cos(np.radians(threshold_angle))
    
        # Calculate face centers for both meshes
        mesh1_centers = np.mean(mesh1_vertices[mesh1_faces], axis=1)
        mesh2_centers = np.mean(mesh2_vertices[mesh2_faces], axis=1)
    
        # Build KD-tree for mesh2 face centers for fast nearest neighbor search
        tree = cKDTree(mesh2_centers)
    
        # For each face in mesh1, find the nearest face center in mesh2
        distances, nearest_indices = tree.query(mesh1_centers)
    
        # Filter by distance threshold if specified
        if distance_threshold is not None:
            valid_mask = distances <= distance_threshold
            mesh1_centers = mesh1_centers[valid_mask]
            mesh1_normals = mesh1_normals[valid_mask]
            distances = distances[valid_mask]
            nearest_indices = nearest_indices[valid_mask]
            original_indices = np.where(valid_mask)[0]
        else:
            original_indices = np.arange(len(mesh1_centers))
    
        # Calculate direction vectors from mesh1 face centers to nearest mesh2 face centers
        direction_vectors = mesh2_centers[nearest_indices] - mesh1_centers
    
        # Normalize direction vectors
        direction_norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
        # Avoid division by zero for coincident points
        direction_norms[direction_norms == 0] = 1.0
        direction_vectors_normalized = direction_vectors / direction_norms
    
        # Normalize mesh1 face normals (ensure they're unit vectors)
        mesh1_normals_normalized = mesh1_normals / np.linalg.norm(mesh1_normals, axis=1, keepdims=True)
    
        # Calculate dot products between face normals and direction vectors
        dot_products = np.sum(mesh1_normals_normalized * direction_vectors_normalized, axis=1)
    
        # Faces are facing if dot product > cos(threshold_angle)
        # (positive dot product means angle < 90°)
        facing_mask = dot_products > cos_threshold
    
        # Return original indices of facing faces
        facing_indices = original_indices[facing_mask]
    
        return facing_indices

    def _find_mutual_facing_faces(self, mesh1_vertices, mesh1_faces, mesh1_normals, mesh2_vertices, mesh2_faces, mesh2_normals, threshold_angle=90.0, distance_threshold=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find pairs of faces that are mutually facing each other.
    
        Returns:
        --------
        mesh1_facing_indices : numpy array
            Indices of faces in mesh1 that are facing mesh2
        mesh2_facing_indices : numpy array
            Corresponding indices of faces in mesh2 that are facing mesh1
        """
    
        # Find mesh1 faces facing mesh2
        mesh1_facing = self._find_facing_faces(mesh1_vertices, mesh1_faces, mesh1_normals, mesh2_vertices, mesh2_faces, mesh2_normals, threshold_angle, distance_threshold)
    
        # Find mesh2 faces facing mesh1
        mesh2_facing = self._find_facing_faces(mesh2_vertices, mesh2_faces, mesh2_normals, mesh1_vertices, mesh1_faces, mesh1_normals, threshold_angle, distance_threshold)
    
        # Calculate centers for all faces
        mesh1_centers = np.mean(mesh1_vertices[mesh1_faces], axis=1)
        mesh2_centers = np.mean(mesh2_vertices[mesh2_faces], axis=1)
    
        # Build KD-trees for both meshes
        tree1 = cKDTree(mesh1_centers)
        tree2 = cKDTree(mesh2_centers)
    
        # For each facing face in mesh1, check if the nearest mesh2 face is also facing mesh1
        if len(mesh1_facing) > 0:
            distances, nearest_mesh2 = tree2.query(mesh1_centers[mesh1_facing])
            # Check if these nearest mesh2 faces are in mesh2_facing
            mutual_mask = np.isin(nearest_mesh2, mesh2_facing)
            mutual_mesh1 = mesh1_facing[mutual_mask]
            mutual_mesh2 = nearest_mesh2[mutual_mask]
        else:
            mutual_mesh1 = np.array([], dtype=int)
            mutual_mesh2 = np.array([], dtype=int)
    
        return mutual_mesh1, mutual_mesh2

    def _face_normal_from_vertex_normals(self, vertex_normals):
        """
        Calculate face normal by averaging vertex normals.
    
        Args:
            vertex_normals: List of 3 vertex normals as numpy arrays or lists
    
        Returns:
            face_normal: Normalized average of vertex normals
        """
        # Convert to numpy arrays
        n0 = np.array(vertex_normals[0])
        n1 = np.array(vertex_normals[1])
        n2 = np.array(vertex_normals[2])
    
        # Average the vertex normals
        face_normal = (n0 + n1 + n2) / 3.0
    
        # Normalize
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm
    
        return face_normal
