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
import numba as nb
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _compute_face_normals

@nb.njit(parallel=True, fastmath=True, cache=True)
def _numpy_concatenate(array_a, array_b) -> np.ndarray:
    """
    numba function to replace np.concatenate
    Concatenates two arrays along the first axis (axis=0).
    Supports 1D and 2D arrays.
    """
    # Handle 1D arrays
    if array_a.ndim == 1 and array_b.ndim == 1:
        result = np.empty(array_a.shape[0] + array_b.shape[0], dtype=array_a.dtype)
        for i in nb.prange(array_a.shape[0]):
            result[i] = array_a[i]
        for i in nb.prange(array_b.shape[0]):
            result[array_a.shape[0] + i] = array_b[i]
        return result
    
    # Handle 2D arrays
    elif array_a.ndim == 2 and array_b.ndim == 2:
        # Check if they have the same number of columns
        if array_a.shape[1] != array_b.shape[1]:
            raise ValueError("Arrays must have the same number of columns for concatenation")
        
        result = np.empty((array_a.shape[0] + array_b.shape[0], array_a.shape[1]), dtype=array_a.dtype)
        for i in nb.prange(array_a.shape[0]):
            for j in range(array_a.shape[1]):
                result[i, j] = array_a[i, j]
        for i in nb.prange(array_b.shape[0]):
            for j in range(array_b_b.shape[1]):
                result[array_a.shape[0] + i, j] = array_b[i, j]
        return result
    
    # Handle mixed dimensions (e.g., 1D + 2D)
    elif array_a.ndim == 1 and array_b.ndim == 2:
        # Convert 1D to 2D if needed
        if array_b.shape[1] == 1:
            result = np.empty((array_a.shape[0] + array_b.shape[0], 1), dtype=array_a.dtype)
            for i in nb.prange(array_a.shape[0]):
                result[i, 0] = array_a[i]
            for i in nb.prange(array_b.shape[0]):
                result[array_a.shape[0] + i, 0] = array_b[i,  0]
            return result
        else:
            raise ValueError("Cannot concatenate 1D array with 2D array of different column count")
    
    elif array_a.ndim == 2 and array_b.ndim == 1:
        # Convert 1D to 2D if needed
        if array_a.shape[1] == 1:
            result = np.empty((array_a.shape[0] + array_b.shape[0], 1), dtype=array_a.dtype)
            for i in nb.prange(array_a.shape[0]):
                result[i, 0] = array_a[i, 0]
            for i in nb.prange(array_b.shape[0]):
                result[array_a.shape[0] + i, 0] = array_b[i]
            return result
        else:
            raise ValueError("Cannot concatenate 2D array with 1D array of different column count")
    
    else:
        # General case for higher dimensions
        result = np.empty((array_a.shape[0] + array_b.shape[0],) + array_a.shape[1:], dtype=array_a.dtype)
        for i in nb.prange(array_a.shape[0]):
            result[i] = array_a[i]
        for i in nb.prange(array_b.shape[0]):
            result[array_a.shape[0] + i] = array_b[i]
        return result

@nb.njit(parallel=True, fastmath=True, cache=True)
def _pyramid_collision_numba(vertices: np.ndarray, faces: np.ndarray, contact_point: np.ndarray, center: np.ndarray, collision_margin: float) -> Tuple[np.ndarray, float]:
    """
    Fast pyramid collision detection using barycentric coordinates.
    Pyramid has 4 vertices and 4 faces.
    """
    # Pyramid structure: apex at index 0, base at indices 1,2,3
    direction = contact_point - center
    direction_norm = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
   
    if direction_norm < 1e-10:
        return np.array([0, 1, 2, 3], dtype=np.int32), 1.0
   
    direction = direction / direction_norm
   
    # Compute face normals
    face_normals = _compute_face_normals(vertices, faces)
   
    # Find which face the contact point is closest to using dot product
    dot_products = np.zeros(face_normals.shape[0], dtype=np.float64)
    for i in nb.prange(face_normals.shape[0]):
        dot_products[i] = face_normals[i, 0] * direction[0] + face_normals[i, 1] * direction[1] + face_normals[i, 2] * direction[2]

    closest_face = np.argmax(dot_products)

    # Get vertices for this face
    face_vertices = faces[closest_face]

    # Get unique vertices
    unique_vertices = np.unique(face_vertices)

    # Add neighboring vertices for smooth transition
    if collision_margin > 0.01:
        for i in range(faces.shape[0]):
            if i != closest_face:
                # Check if faces share vertices
                shared = False
                for j in range(face_vertices.shape[0]):
                    for k in range(faces[i].shape[0]):
                        if face_vertices[j] == faces[i, k]:
                            shared = True
                            break
                    if shared:
                        break
                if shared:
                    # Add vertices from adjacent face
                    new_vertices = _numpy_concatenate(unique_vertices, faces[i])
#                    new_vertices = np.concatenate([unique_vertices, faces[i]])
                    unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]
   
    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _octahedron_collision_numba(vertices: np.ndarray, faces: np.ndarray,
                                contact_point: np.ndarray, center: np.ndarray,
                                collision_margin: float) -> Tuple[np.ndarray, float]:
    """
    Fast octahedron collision detection using sign-based face selection.
    Octahedron has 6 vertices and 8 faces.
    """
    direction = contact_point - center
    octant_signs = np.zeros(3, dtype=np.int32)
    for i in nb.prange(3):
        if direction[i] >= 0:
            octant_signs[i] = 1
        else:
            octant_signs[i] = -1

    # Map octant to face index (pre-computed mapping for octahedron)
    # Faces are indexed by their normal direction
    # Face 0: (1,1,1), Face 1: (1,1,-1), Face 2: (1,-1,1), Face 3: (1,-1,-1)
    # Face 4: (-1,1,1), Face 5: (-1,1,-1), Face 6: (-1,-1,1), Face 7: (-1,-1,-1)

    face_idx = 0
    if octant_signs[0] == 1:
        if octant_signs[1] == 1:
            if octant_signs[2] == 1:
                face_idx = 0
            else:
                face_idx = 1
        else:
            if octant_signs[2] == 1:
                face_idx = 2
            else:
                face_idx = 3
    else:
        if octant_signs[1] == 1:
            if octant_signs[2] == 1:
                face_idx = 4
            else:
                face_idx = 5
        else:
            if octant_signs[2] == 1:
                face_idx = 6
            else:
                face_idx = 7

    # Get vertices for this face
    face_vertices = faces[face_idx]
    unique_vertices = np.unique(face_vertices)

    # Add adjacent face vertices for smooth transition
    if collision_margin > 0.01:
        for i in range(faces.shape[0]):
            if i != face_idx:
                # Check if faces share vertices
                shared = False
                for j in range(face_vertices.shape[0]):
                    for k in range(faces[i].shape[0]):
                        if face_vertices[j] == faces[i, k]:
                            shared = True
                            break
                    if shared:
                        break
                if shared:
                    new_vertices = _numpy_concatenate(unique_vertices, faces[i])
#                    new_vertices = np.concatenate([unique_vertices, faces[i]])
                    unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _cube_collision_numba(vertices: np.ndarray, faces: np.ndarray,
                          contact_point: np.ndarray, center: np.ndarray,
                          collision_margin: float) -> Tuple[np.ndarray, float]:
    """
    Fast cube collision detection using axis-aligned bounding box.
    Cube has 8 vertices and 6 faces (12 triangles).
    """
    direction = contact_point - center

    # Find the dominant axis (which face is closest)
    abs_direction = np.abs(direction)
    dominant_axis = np.argmax(abs_direction)
    face_sign = 1 if direction[dominant_axis] >= 0 else -1

    # For a cube, each face has 2 triangles
    # Face indices are grouped: [0,1] for +x, [2,3] for -x, etc.
    face_indices = np.zeros(2, dtype=np.int32)

    if dominant_axis == 0:  # x-axis
        if face_sign > 0:
            face_indices[0] = 0
            face_indices[1] = 1
        else:
            face_indices[0] = 2
            face_indices[1] = 3
    elif dominant_axis == 1:  # y-axis
        if face_sign > 0:
            face_indices[0] = 4
            face_indices[1] = 5
        else:
            face_indices[0] = 6
            face_indices[1] = 7
    else:  # z-axis
        if face_sign > 0:
            face_indices[0] = 8
            face_indices[1] = 9
        else:
            face_indices[0] = 10
            face_indices[1] = 11

    # Get all vertices from these faces
    face_vertices = _numpy_concatenate(faces[face_indices[0]], faces[face_indices[1]])
#    face_vertices = np.concatenate([faces[face_indices[0]], faces[face_indices[1]]])
    unique_vertices = np.unique(face_vertices)

    # Add edge vertices for smooth transition
    if collision_margin > 0.01:
        # Find adjacent faces (sharing an edge)
        for i in range(faces.shape[0]):
            if i != face_indices[0] and i != face_indices[1]:
                # Check if face shares vertices with our faces
                shared = False
                for j in range(face_vertices.shape[0]):
                    for k in range(faces[i].shape[0]):
                        if face_vertices[j] == faces[i, k]:
                            shared = True
                            break
                    if shared:
                        break
                if shared:
#                    new_vertices = np.concatenate([unique_vertices, faces[i]])
                    new_vertices = _numpy_concatenate(unique_vertices, faces[i])
                    unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _icosahedron_collision_numba(vertices: np.ndarray, faces: np.ndarray, contact_point: np.ndarray, center: np.ndarray, collision_margin: float) -> Tuple[np.ndarray, float]:
    """
    Fast icosahedron collision detection using face-based approach.
    """
    direction = contact_point - center
    direction_norm = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

    if direction_norm < 1e-10:
        return np.arange(vertices.shape[0], dtype=np.int32), 1.0

    direction = direction / direction_norm

    # Compute face normals
    face_normals = _compute_face_normals(vertices, faces)

    # Find the face whose normal is most aligned with the contact direction
    dot_products = np.zeros(face_normals.shape[0], dtype=np.float64)
    for i in nb.prange(face_normals.shape[0]):
        dot_products[i] = face_normals[i, 0] * direction[0] + face_normals[i, 1] * direction[1] + face_normals[i, 2] * direction[2]

    closest_face = np.argmax(dot_products)

    # Get vertices for this face
    face_vertices = faces[closest_face]
    unique_vertices = np.unique(face_vertices)

    # Add vertices from adjacent faces
    if collision_margin > 0.01:
        for i in range(faces.shape[0]):
            if i != closest_face:
                # Check if faces share vertices
                shared = False
                for j in range(face_vertices.shape[0]):
                    for k in range(faces[i].shape[0]):
                        if face_vertices[j] == faces[i, k]:
                            shared = True
                            break
                    if shared:
                        break
                if shared:
#                    new_vertices = np.concatenate([unique_vertices, faces[i]])
                    new_vertices = _numpy_concatenate(unique_vertices, faces[i])
                    unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _icosahedron_collision_subdivided_numba(vertices: np.ndarray, faces: np.ndarray, contact_point: np.ndarray, center: np.ndarray, collision_margin: float, subdivisions: int) -> Tuple[np.ndarray, float]:
    """
    Collision detection for subdivided icosahedron.
    """
    direction = contact_point - center
    direction_norm = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

    if direction_norm < 1e-10:
        return np.arange(vertices.shape[0], dtype=np.int32), 1.0

    direction = direction / direction_norm

    # Compute face normals
    face_normals = _compute_face_normals(vertices, faces)

    # Find the face whose normal is most aligned with the contact direction
    dot_products = np.zeros(face_normals.shape[0], dtype=np.float64)
    for i in nb.prange(face_normals.shape[0]):
        dot_products[i] = face_normals[i, 0] * direction[0] + face_normals[i, 1] * direction[1] + face_normals[i, 2] * direction[2]

    closest_face = np.argmax(dot_products)

    # Get vertices for this face
    face_vertices = faces[closest_face]
    unique_vertices = np.unique(face_vertices)

    # For subdivided icosahedron, include more vertices based on subdivision level
    sub_scale = 1.0 + subdivisions * 0.5

    # Find all vertices within the collision margin (scaled)
    search_radius = collision_margin * 2.0 * sub_scale

    # Find nearby vertices using a boolean mask instead of list
    distances = np.zeros(vertices.shape[0], dtype=np.float64)
    for i in nb.prange(vertices.shape[0]):
        dist = np.sqrt((vertices[i, 0] - contact_point[0])**2 + (vertices[i, 1] - contact_point[1])**2 + (vertices[i, 2] - contact_point[2])**2)
        distances[i] = dist
    
    # Create mask and get indices
    mask = distances < search_radius
    nearby_vertices = np.where(mask)[0]

    if nearby_vertices.shape[0] > 0:
        new_vertices = _numpy_concatenate(unique_vertices, nearby_vertices)
        unique_vertices = np.unique(new_vertices)

    # Add vertices from adjacent faces (sharing an edge)
    for i in range(faces.shape[0]):
        if i != closest_face:
            # Check if faces share at least 2 vertices (edge)
            shared_count = 0
            for j in range(face_vertices.shape[0]):
                for k in range(faces[i].shape[0]):
                    if face_vertices[j] == faces[i, k]:
                        shared_count += 1
                        break
            if shared_count >= 2:  # Share an edge
                new_vertices = _numpy_concatenate(unique_vertices, faces[i])
                unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _convex_hull_collision_numba(vertices: np.ndarray, faces: np.ndarray, contact_point: np.ndarray, center: np.ndarray, collision_margin: float) -> Tuple[np.ndarray, float]:
    """
    Fast convex hull collision detection using face-based approach.
    """
    direction = contact_point - center
    direction_norm = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

    if direction_norm < 1e-10:
        return np.arange(vertices.shape[0], dtype=np.int32), 1.0

    direction = direction / direction_norm

    # Compute face normals
    face_normals = _compute_face_normals(vertices, faces)

    # Find the face whose normal is most aligned with the contact direction
    dot_products = np.zeros(face_normals.shape[0], dtype=np.float64)
    for i in nb.prange(face_normals.shape[0]):
        dot_products[i] = face_normals[i, 0] * direction[0] + face_normals[i, 1] * direction[1] + face_normals[i, 2] * direction[2]

    closest_face = np.argmax(dot_products)

    # Get vertices for this face
    face_vertices = faces[closest_face]
    unique_vertices = np.unique(face_vertices)

    # Add vertices from adjacent faces
    if collision_margin > 0.01:
        for i in range(faces.shape[0]):
            if i != closest_face:
                # Check if faces share vertices
                shared = False
                for j in range(face_vertices.shape[0]):
                    for k in range(faces[i].shape[0]):
                        if face_vertices[j] == faces[i, k]:
                            shared = True
                            break
                    if shared:
                        break
                if shared:
                    new_vertices = _numpy_concatenate(unique_vertices, faces[i])
                    unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@dataclass 
class ProxyPhysics:
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')

    def _compute_face_area(self, vertices_idx, faces):
        """Compute face area ratio from vertex indices."""
        if len(vertices_idx) == 0:
            return 0.0

        mesh_faces_idx = np.where(np.any(np.isin(faces, vertices_idx), axis=1))[0]
        return len(mesh_faces_idx) / len(faces)

    def optimized_proxy_collision(self, obj1_idx, obj2_idx, cp1, cp2, collision_margin, contact_type, trajectory1, trajectory2, mesh1_faces, mesh2_faces, sample_idx):
        """
        Ultra-optimized collision detection for proxy meshes.
        Uses analytical geometry instead of KDTree queries.
        """
        # Get proxy mesh properties
        for config_obj in self.config.objects:
            if config_obj.idx == obj1_idx:
                proxy1 = config_obj.proxy_type
            elif config_obj.idx == obj2_idx:
                proxy2 = config_obj.proxy_type

        # Get current mesh vertices
        mesh1_vertices = trajectory1.get_vertices(sample_idx)
        mesh2_vertices = trajectory2.get_vertices(sample_idx)

        # Pre-compute centers once
        center1 = np.mean(mesh1_vertices, axis=0)
        center2 = np.mean(mesh2_vertices, axis=0)

        # Use analytical collision for all proxy types
        if proxy1 is not False:
            vertices1_idx, face_area1 = self.analytical_proxy_collision(proxy1, mesh1_vertices, mesh1_faces, cp1, center1, collision_margin)
        else:
            # Fallback to KDTree for other types
            tree1 = cKDTree(mesh1_vertices)
            radius = collision_margin * (4.0 if contact_type in [4, 5] else 2.0)
            vertices1_idx = np.array(tree1.query_ball_point(cp1, radius, workers=-1))
            face_area1 = self._compute_face_area(vertices1_idx, mesh1_faces)

        if proxy2 is not False:
            vertices2_idx, face_area2 = self.analytical_proxy_collision(proxy2, mesh2_vertices, mesh2_faces, cp2, center2, collision_margin)
        else:
            tree2 = cKDTree(mesh2_vertices)
            radius = collision_margin * (4.0 if contact_type in [4, 5] else 2.0)
            vertices2_idx = np.array(tree2.query_ball_point(cp2, radius, workers=-1))
            face_area2 = self._compute_face_area(vertices2_idx, mesh2_faces)
   
        return vertices1_idx, vertices2_idx, face_area1, face_area2

    def analytical_proxy_collision(self, proxy_type, vertices, faces, contact_point, center, collision_margin):
        """ 
        Analytical collision detection for proxy meshes.
        Uses geometric relationships instead of KDTree.
        """ 
        if proxy_type == 0:  # Pyramid (4 vertices)
            return _pyramid_collision_numba(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 1:  # Octahedron (6 vertices)
            return _octahedron_collision_numba(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 2:  # Cube (8 vertices)
            return _cube_collision_numba(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 3:  # Icosahedron (12 vertices, 20 faces)
            return _icosahedron_collision_numba(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 4:  # Icosahedron subdiv 1 (42 vertices, 80 faces)
            return _icosahedron_collision_subdivided_numba(vertices, faces, contact_point, center, collision_margin, subdivisions=1)
        elif proxy_type == 5:  # Icosahedron subdiv 2 (162 vertices, 320 faces)
            return _icosahedron_collision_subdivided_numba(vertices, faces, contact_point, center, collision_margin, subdivisions=2)
        elif proxy_type == 6:  # Convex hull
            return _convex_hull_collision_numba(vertices, faces, contact_point, center, collision_margin)

        return np.array([], dtype=np.int32), 0.0
