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
from numba import jit, prange
from scipy.spatial import cKDTree
from scipy.integrate import solve_ivp
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from itertools import groupby

from ..core.entity_manager import EntityManager
from ..lib.collision_data import CollisionData
from ..lib.functions import _parse_lib

@dataclass
class CollisionSolver:
    entity_manager: EntityManager

    def compute(self, collision: CollisionData) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        trajectories = self.entity_manager.get('trajectories')
        forces = self.entity_manager.get('forces')

        obj1_idx = collision.obj1_idx
        obj2_idx = collision.obj2_idx
        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
            if conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
        if not config_obj1.static or not config_obj2.static:
            for t_idx in trajectories.keys():
                if trajectories[t_idx].obj_idx == obj1_idx:
                    self.trajectory1 = trajectories[t_idx]
                    mesh1_faces = self.trajectory1.get_faces()
                if trajectories[t_idx].obj_idx == obj2_idx:
                    self.trajectory2 = trajectories[t_idx]
                    mesh2_faces = self.trajectory2.get_faces()
                for f_idx in forces.keys():
                    if forces[f_idx].obj_idx == obj1_idx and obj2_idx in forces[f_idx].other_obj_idx:
                        force = forces[f_idx]

        total_samples = int(self.trajectory1.get_x()[-1])

        collision_margin = collision.avg_distance * (1 + collision.threshold/2)

        start_samples = int(collision.frame - collision.impulse_range / 2)
        stop_samples = int(collision.frame + collision.impulse_range)

        if collision.type.value == 'contact':
            stop_samples = int(collision.frame + collision.frame_range + collision.impulse_range)
        if not stop_samples <= total_samples:
            stop_samples = total_samples

        samples_idx, collision_v_idx1, collision_v_idx2, num_v_idx1, num_v_idx2, vertex1_id_list, vertex2_id_list = ([] for _ in range(7))
        for sample_idx in range(start_samples, stop_samples):
            samples_idx.append(sample_idx)
            mesh1_faces_idx, mesh2_faces_idx = self._get_facing_face(sample_idx, collision_margin)
            print(f"facing faces between {config_obj1.name} and {config_obj2.name} at frame {sample_idx}: {mesh1_faces_idx.shape[0]} {mesh2_faces_idx.shape[0]}")
            cvidx1 = np.unique(mesh1_faces[mesh1_faces_idx])
            cvidx2 = np.unique(mesh2_faces[mesh2_faces_idx])
            vertex1_id_list += cvidx1.tolist()
            vertex2_id_list += cvidx2.tolist()
            collision_v_idx1.append(cvidx1)
            collision_v_idx2.append(cvidx2)
            num_v_idx1.append(cvidx1.shape[0])
            num_v_idx2.append(cvidx2.shape[0])

        samples_idx = np.array(samples_idx, dtype=np.int32)
        vertex1_id = np.array(vertex1_id_list, dtype=np.int32)
        vertex2_id = np.array(vertex2_id_list, dtype=np.int32)
        num_v_idx1 = np.array(num_v_idx1, dtype=np.int32)
        num_v_idx2 = np.array(num_v_idx2, dtype=np.int32)

        max_vertex1_num = max(num_v_idx1)
        collision_v_arr1 = []
        for i in range(len(collision_v_idx1)):
            delta_vertex = int(max_vertex1_num - collision_v_idx1[i].shape[0])
            collision_v_arr1.append(np.append(collision_v_idx1[i], np.zeros(delta_vertex, dtype=np.int32)))
        collision_v_idx1 = np.array(collision_v_arr1, dtype=np.int32)

        max_vertex2_num = max(num_v_idx2)
        collision_v_arr2 = []
        for i in range(len(collision_v_idx2)):
            delta_vertex = int(max_vertex2_num - collision_v_idx2[i].shape[0])
            collision_v_arr2.append(np.append(collision_v_idx2[i], np.zeros(delta_vertex, dtype=np.int32)))
        collision_v_idx2 = np.array(collision_v_arr2, dtype=np.int32)

        collision_area = [samples_idx, collision_v_idx1, num_v_idx1, vertex1_id, collision_v_idx2, num_v_idx2, vertex2_id]
        collision.add_area('collision_area', collision_area)

    def _get_facing_face(self, sample_idx: int, collision_margin: float) -> List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]:
        collision_area = []
        mesh1_vertices = self.trajectory1.get_vertices(sample_idx)
        mesh1_faces = self.trajectory1.get_faces(sample_idx)
        mesh1_normals = self.trajectory1.get_normals(sample_idx)
        face1_normals = self._calculate_face_normals(mesh1_faces, mesh1_normals)

        mesh2_vertices = self.trajectory2.get_vertices(sample_idx)
        mesh2_faces = self.trajectory2.get_faces(sample_idx)
        mesh2_normals = self.trajectory2.get_normals(sample_idx)
        face2_normals = self._calculate_face_normals(mesh2_faces, mesh2_normals)

        mesh1_faces_idx, mesh2_faces_idx = self._find_mutual_facing_faces(mesh1_vertices=mesh1_vertices, mesh1_faces=mesh1_faces, mesh1_normals=face1_normals, mesh2_vertices=mesh2_vertices, mesh2_faces=mesh2_faces, mesh2_normals=face2_normals, threshold_angle=90, distance_threshold=collision_margin)

        return mesh1_faces_idx, mesh2_faces_idx

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
        distances, nearest_indices = tree.query(mesh1_centers, workers=-1)

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
            distances, nearest_mesh2 = tree2.query(mesh1_centers[mesh1_facing], workers=-1)
            # Check if these nearest mesh2 faces are in mesh2_facing
            mutual_mask = np.isin(nearest_mesh2, mesh2_facing)
            mutual_mesh1 = mesh1_facing[mutual_mask]
            mutual_mesh2 = nearest_mesh2[mutual_mask]
        else:
            mutual_mesh1 = np.array([], dtype=int)
            mutual_mesh2 = np.array([], dtype=int)
    
        return mutual_mesh1, mutual_mesh2

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_face_normals(faces: np.ndarray, vertex_normals: np.ndarray) -> np.ndarray:
        """Calculate face normals from from vertex normals in batch (Numba optimized)."""
        n_faces = len(faces)
        face_normals = np.empty((n_faces, 3), dtype=np.float64)
        
        for i in prange(n_faces):
            face = faces[i]
            # Get vertex normals for this face
            n0 = vertex_normals[face[0]]
            n1 = vertex_normals[face[1]]
            n2 = vertex_normals[face[2]]
            
            # Average and normalize
            avg = (n0 + n1 + n2) / 3.0
            norm = np.sqrt(avg[0]*avg[0] + avg[1]*avg[1] + avg[2]*avg[2])
            
            if norm > 0:
                face_normals[i] = avg / norm
            else:
                face_normals[i] = avg
        
        return face_normals
