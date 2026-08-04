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
    shape = (len(array_a) + len(array_b), len(array_a))
    dtype = array_a.dtype
    new_array = np.zeros(shape, dtype=dtype)
    new_array[:array_a.shape[0]] = array_a
    new_array[-array_b.shape[0]:] = array_b
    return new_array

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
        for i in nb.prange(faces.shape[0]):
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
        for i in nb.prange(faces.shape[0]):
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
        for i in nb.prange(faces.shape[0]):
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
def _icosahedron_collision_numba(vertices: np.ndarray, faces: np.ndarray,
                                 contact_point: np.ndarray, center: np.ndarray,
                                 collision_margin: float) -> Tuple[np.ndarray, float]:
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
    if collision_marginargin > 0.01:
        for i in nb.prange(faces.shape[0]):
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
def _icosahedron_collision_subdivided_numba(vertices: np.ndarray, faces: np.ndarray,
                                            contact_point: np.ndarray, center: np.ndarray,
                                            collision_margin: float, subdivisions: int) -> Tuple[np.ndarray, float]:
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

    # Find nearby vertices
    nearby_vertices = []
    for i in nb.prange(vertices.shape[0]):
        dist = np.sqrt((vertices[i, 0] - contact_point[0])**2 +
                       (vertices[i, 1] - contact_point[1])**2 +
                       (vertices[i, 2] - contact_point[2])**2)
        if dist < search_radius:
            nearby_vertices.append(i)

    if len(nearby_vertices) > 0:
#        new_vertices = np.concatenate([unique_vertices, np.array(nearby_vertices, dtype=np.int32)])
        new_vertices = _numpy_concatenate(unique_vertices, np.array(nearby_vertices, dtype=np.int32))
        unique_vertices = np.unique(new_vertices)

    # Add vertices from adjacent faces (sharing an edge)
    for i in nb.prange(faces.shape[0]):
        if i != closest_face:
            # Check if faces share at least 2 vertices (edge)
            shared_count = 0
            for j in range(face_vertices.shape[0]):
                for k in range(faces[i].shape[0]):
                    if face_vertices[j] == faces[i, k]:
                        shared_count += 1
                        break
            if shared_count >= 2:  # Share an edge
#                new_vertices = np.concatenate([unique_vertices, faces[i]])
                new_vertices = _numpy_concatenate(unique_vertices, faces[i])
                unique_vertices = np.unique(new_vertices)

    face_area = unique_vertices.shape[0] / vertices.shape[0]

    return unique_vertices, face_area

@nb.njit(parallel=True, fastmath=True, cache=True)
def _get_adjacent_faces_numba(faces: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """
    Get faces adjacent to the given face indices.
    """
    # Collect all vertices from the given faces
    face_vertices = np.zeros(0, dtype=np.int32)
    for f_idx in nb.prange(face_indices.shape[0]):
        idx = face_indices[f_idx]
        face_vertices = _numpy_concatenate(face_vertices, faces[idx])
    for idx in face_indices:
#        face_vertices = np.concatenate([face_vertices, faces[idx]])

    face_vertices_set = np.unique(face_vertices)

    # Find adjacent faces
    adjacent = []
    for i in nb.prange(faces.shape[0]):
        # Skip if this face is already in face_indices
        is_in_indices = False
        for idx in face_indices:
            if i == idx:
                is_in_indices = True
                break
        if is_in_indices:
            continue

        # Count shared vertices
        shared_count = 0
        for j in range(faces[i].shape[0]):
            for k in range(face_vertices_set.shape[0]):
                if faces[i, j] == face_vertices_set[k]:
                    shared_count += 1
                    break

        if shared_count >= 2:  # Share an edge
            adjacent.append(i)

    return np.array(adjacent, dtype=np.int32)

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
            face_area1 = _compute_face_area(vertices1_idx, mesh1_faces)

        if proxy2 is not False:
            vertices2_idx, face_area2 = self.analytical_proxy_collision(proxy2, mesh2_vertices, mesh2_faces, cp2, center2, collision_margin)
        else:
            tree2 = cKDTree(mesh2_vertices)
            radius = collision_margin * (4.0 if contact_type in [4, 5] else 2.0)
            vertices2_idx = np.array(tree2.query_ball_point(cp2, radius, workers=-1))
            face_area2 = _compute_face_area(vertices2_idx, mesh2_faces)
   
        return vertices1_idx, vertices2_idx, face_area1, face_area2

    def analytical_proxy_collision(self, proxy_type, vertices, faces, contact_point, center, collision_margin):
        """ 
        Analytical collision detection for proxy meshes.
        Uses geometric relationships instead of KDTree.
        """ 
        if proxy_type == 0:  # Pyramid (4 vertices)
            try:
                return _pyramid_collision_numba(vertices, faces, contact_point, center, collision_margin)
            except Exception as e:
                print(e)
                return self._pyramid_collision(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 1:  # Octahedron (6 vertices)
            try:
                return _octahedron_collision_numba(vertices, faces, contact_point, center, collision_margin)
            except Exception as e:
                print(e)
                return self._octahedron_collision(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 2:  # Cube (8 vertices)
            try:
                return _cube_collision_numba(vertices, faces, contact_point, center, collision_margin)
            except Exception as e:
                print(e)
                return self._cube_collision(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 3:  # Icosahedron (12 vertices, 20 faces)
            try:
                return _icosahedron_collision_numba(vertices, faces, contact_point, center, collision_margin)
            except Exception as e:
                print(e)
                return self._icosahedron_collision(vertices, faces, contact_point, center, collision_margin, subdivisions=0)
        elif proxy_type == 4:  # Icosahedron subdiv 1 (42 vertices, 80 faces)
            try:
                return _icosahedron_collision_subdivided_numba(vertices, faces, contact_point, center, collision_margin, subdivisions=1)
            except Exception as e:
                print(e)
                return self._icosahedron_collision_subdivided(vertices, faces, contact_point, center, collision_margin, subdivisions=1)
        elif proxy_type == 5:  # Icosahedron subdiv 2 (162 vertices, 320 faces)
            try:
                return _icosahedron_collision_subdivided_numba(vertices, faces, contact_point, center, collision_margin, subdivisions=2)
            except Exception as e:
                print(e)
                return self._icosahedron_collision_subdivided(vertices, faces, contact_point, center, collision_margin, subdivisions=2)

        return np.array([], dtype=np.int32), 0.0

    def _pyramid_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast pyramid collision detection using barycentric coordinates.
        Pyramid has 4 vertices and 4 faces.
        """
        # Pyramid structure: apex at index 0, base at indices 1,2,3
        # The contact point direction from center tells us which face is hit

        direction = contact_point - center
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-10:
            return np.array([0, 1, 2, 3], dtype=np.int32), 1.0

        direction = direction / direction_norm

        # Pre-computed face normals for pyramid (from cache or compute on the fly)
        # Face 0: apex to base front (vertices 0,1,2)
        # Face 1: apex to base right (vertices 0,2,3)
        # Face 2: apex to base left (vertices 0,3,1)
        # Face 3: base (vertices 1,3,2)

        # Find which face the contact point is closest to using dot product
        face_normals = _compute_face_normals(vertices, faces)
        dot_products = np.dot(face_normals, direction)

        # The closest face has the highest dot product
        closest_face = np.argmax(dot_products)
   
        # Get vertices for this face
        face_vertices = faces[closest_face]
        unique_vertices = np.unique(face_vertices)

        # Add neighboring vertices for smooth transition
        # This prevents artifacts at face boundaries
        if collision_margin > 0.01:
            # Add vertices from adjacent faces
            for i, face in enumerate(faces):
                if i != closest_face:
                    if len(np.intersect1d(face, face_vertices)) > 0:
#                        unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
                        unique_vertices = np.unique(_numpy_concatenate(unique_vertices, face))

        face_area = len(unique_vertices) / len(vertices)

        return unique_vertices, face_area

    def _octahedron_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast octahedron collision detection using sign-based face selection.
        Octahedron has 6 vertices and 8 faces.
        """
        # Octahedron: vertices at (±1,0,0), (0,±1,0), (0,0,±1)
        # The contact point's octant tells us which face is hit

        direction = contact_point - center
        octant_signs = np.sign(direction)

        # Map octant to face index (pre-computed mapping for octahedron)
        # Faces are indexed by their normal direction
        octant_to_face = {
            (1, 1, 1): 0,   # Face normal: (1,1,1)/sqrt(3)
            (1, 1, -1): 1,  # Face normal: (1,1,-1)/sqrt(3)
            (1, -1, 1): 2,  # etc.
            (1, -1, -1): 3,
            (-1, 1, 1): 4,
            (-1, 1, -1): 5,
            (-1, -1, 1): 6,
            (-1, -1, -1): 7
        }

        octant_key = (octant_signs[0], octant_signs[1], octant_signs[2])
        face_idx = octant_to_face.get(octant_key, 0)

        # Get vertices for this face
        face_vertices = faces[face_idx]
        unique_vertices = np.unique(face_vertices)

        # Add adjacent face vertices for smooth transition
        if collision_margin > 0.01:
            for i, face in enumerate(faces):
                if i != face_idx:
                    if len(np.intersect1d(face, face_vertices)) > 0:
#                        unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
                        unique_vertices = np.unique(_numpy_concatenate(unique_vertices, face))

        face_area = len(unique_vertices) / len(vertices)

        return unique_vertices, face_area

    def _cube_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast cube collision detection using axis-aligned bounding box.
        Cube has 8 vertices and 6 faces (12 triangles).
        """
        # Cube is axis-aligned, so we can use simple coordinate checks
        direction = contact_point - center

        # Find the dominant axis (which face is closest)
        abs_direction = np.abs(direction)
        dominant_axis = np.argmax(abs_direction)
        face_sign = np.sign(direction[dominant_axis])

        # For a cube, each face has 2 triangles
        # Face indices are grouped: [0,1] for +x, [2,3] for -x, etc.
        if dominant_axis == 0:  # x-axis
            face_indices = np.array([0, 1]) if face_sign > 0 else np.array([2, 3])
        elif dominant_axis == 1:  # y-axis
            face_indices = np.array([4, 5]) if face_sign > 0 else np.array([6, 7])
        else:  # z-axis
            face_indices = np.array([8, 9]) if face_sign > 0 else np.array([10, 11])

        # Get all vertices from these faces
        face_vertices = faces[face_indices].flatten()
        unique_vertices = np.unique(face_vertices)

        # Add edge vertices for smooth transition
        if collision_margin > 0.01:
            # Add vertices from adjacent faces (sharing an edge)
            try:
                edge_faces = _get_adjacent_faces_numba(faces, face_indices)
            except Exception as e:
                print(e)
                edge_faces = self._get_adjacent_faces(faces, face_indices)
            for edge_face in edge_faces:
#                unique_vertices = np.unique(np.concatenate([unique_vertices, faces[edge_face]]))
                unique_vertices = np.unique(_numpy_concatenate(unique_vertices, faces[edge_face]))

        face_area = len(unique_vertices) / len(vertices)

        return unique_vertices, face_area

    def _icosahedron_collision(self, vertices, faces, contact_point, center, collision_margin, subdivisions=0):
        """
        Fast icosahedron collision detection using face-based approach.
        
        Icosahedron has 12 vertices and 20 faces (base), with subdivisions
        increasing vertex/face count. Uses the fact that all faces are
        equilateral triangles on a sphere-like surface.
        """
        direction = contact_point - center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-10:
            return np.arange(len(vertices), dtype=np.int32), 1.0
        
        direction = direction / direction_norm
        
        # Compute face normals
        face_normals = _compute_face_normals(vertices, faces)
        
        # Find the face whose normal is most aligned with the contact direction
        dot_products = np.dot(face_normals, direction)
        closest_face = np.argmax(dot_products)
        
        # Get vertices for this face
        face_vertices = faces[closest_face]
        unique_vertices = np.unique(face_vertices)
        
        # Add vertices from adjacent faces
        if collision_margin > 0.01:
            for i, face in enumerate(faces):
                if i != closest_face:
                    if len(np.intersect1d(face, face_vertices)) > 0:
#                        unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
                        unique_vertices = np.unique(_numpy_concatenate(unique_vertices, face))
        
        face_area = len(unique_vertices) / len(vertices)
        
        return unique_vertices, face_area

    def _icosahedron_collision_subdivided(self, vertices, faces, contact_point, center, collision_margin, subdivisions):
        """
        Collision detection for subdivided icosahedron (proxy types 3, 4, 5).
        
        Uses a hierarchical approach:
        1. Find the base icosahedron face (using 12 base vertices)
        2. For subdivided meshes, find the specific sub-face containing the contact point
        3. Include vertices from surrounding faces based on collision margin
        """
        direction = contact_point - center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-10:
            return np.arange(len(vertices), dtype=np.int32), 1.0
        
        direction = direction / direction_norm
        
        # Compute face normals
        face_normals = _compute_face_normals(vertices, faces)
        
        # Find the face whose normal is most aligned with the contact direction
        dot_products = np.dot(face_normals, direction)
        closest_face = np.argmax(dot_products)
        
        # Get vertices for this face
        face_vertices = faces[closest_face]
        unique_vertices = np.unique(face_vertices)
        
        # For subdivided icosahedron, include more vertices based on subdivision level
        sub_scale = 1.0 + subdivisions * 0.5
        
        # Find all vertices within the collision margin (scaled)
        search_radius = collision_margin * 2.0 * sub_scale
        distances = np.linalg.norm(vertices - contact_point, axis=1)
        nearby_vertices = np.where(distances < search_radius)[0]
        
        if len(nearby_vertices) > 0:
#            unique_vertices = np.unique(np.concatenate([unique_vertices, nearby_vertices]))
            unique_vertices = np.unique(_numpy_concatenate(unique_vertices, nearby_vertices))
        
        # Add vertices from adjacent faces (sharing an edge)
        face_vertices_set = set(face_vertices)
        for i, face in enumerate(faces):
            if i != closest_face:
                if len(set(face) & face_vertices_set) >= 2:  # Share an edge
#                    unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
                    unique_vertices = np.unique(_numpy_concatenate(unique_vertices, face))
        
        face_area = len(unique_vertices) / len(vertices)
        
        return unique_vertices, face_area

    def _get_adjacent_faces(self, faces, face_indices):
        """Get faces adjacent to the given face indices."""
        adjacent = set()
        face_vertices_set = set(faces[face_indices].flatten())

        for i, face in enumerate(faces):
            if i not in face_indices:
                if len(set(face.flatten()) & face_vertices_set) >= 2:  # Share an edge
                    adjacent.add(i)

        return list(adjacent)

    def _octant_to_face_indices(self, octant: np.ndarray, proxy_type: int = 0) -> np.ndarray:
        """
        Map octant sign to face indices for proxy meshes.
    
        For pyramid (proxy_type=0): 4 vertex, 4 faces
        For octahedron (proxy_type=1): 6 vertex, 8 faces
        For octant/cube (proxy_type=2): 8 vertex, 6 faces
    
        Returns face indices for the given octant.
        """
        if proxy_type == 0:
            # Pyramid has 4 faces
            # Faces are: [apex, base1, base2], [apex, base2, base3], [apex, base3, base1], [base1, base3, base2]
            # Apex is at +x, base is at -x
            # Map based on which quadrant the contact point is in
            if octant[1] >= 0 and octant[2] >= 0:
                return np.array([0])  # Front face
            elif octant[1] >= 0 and octant[2] < 0:
                return np.array([1])  # Right face
            elif octant[1] < 0 and octant[2] >= 00:
                return np.array([2])  # Left face
            else:
                return np.array([3])  # Base face
        elif proxy_type == 1:
            # Octahedron has 8 faces
            octant_key = (octant[0] >= 0, octant[1] >= 0, octant[2] >= 0)
            face_map = {
                (True, True, True): 0,
                (True, True, False): 1,
                (True, False, True): 2,
                (True, False, False): 3,
                (False, True, True): 4,
                (False, True, False): 5,
                (False, False, True): 6,
                (False, False, False): 7
            }
            return np.array([face_map.get(octant_key, 0)])
        elif proxy_type == 2:
            # Cube has 6 faces
            # Determine which face is closest based on the dominant axis
            abs_octant = np.abs(octant)
            dominant_axis = np.argmax(abs_octant)
            if dominant_axis == 0:  # x-axis
                return np.array([0, 1]) if octant[0] >= 0 else np.array([2, 3])
            elif dominant_axis == 1:  # y-axis
                return np.array([4, 5]) if octant[1] >= 0 else np.array([6, 7])
            else:  # z-axis
                return np.array([8, 9]) if octant[2] >= 0 else np.array([10, 11])
        else:
            # Default to octahedron behavior
            return np.array([0])

