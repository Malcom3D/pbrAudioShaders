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
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
import trimesh

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _load_pose


@dataclass
class ProxyMesh:
    """
    Generate low-resolution proxy meshes for objects.

    Replaces each selected object's mesh with a low-res axis-aligned proxy mesh
    based on the proxy_type parameter, that inscribscribes the object.

    proxy_type values:
    - 0: octahedron (no subdivision)
    - 1: dodecahedron (no subdivision)
    - 2,3,4: icosahedron with subdivision of (proxy_type - 2)
    """
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> List[int]:
        """
        Compute object and replace those with proxy_type enabled.
        """
        config = self.entity_manager.get('config')
        proxy_objects = []

        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                if config_obj.proxy_type is not False:
                    print(f"Creating proxy mesh for {config_obj.name} idx={config_obj.idx} proxy_type={config_obj.proxy_type}")
                    self._create_proxy_sequence(config_obj)
                    proxy_objects.append(config_obj.idx)

        return proxy_objects

    def _create_proxy_sequence(self, config_obj: Any) -> None:
        """
        Create proxy mesh sequence for an object.

        For each frame, generates a proxy mesh that inscribes the object
        at that frame's position and orientation.
        """
        # Load pose data to get number of frames
        positions, rotations = _load_pose(config_obj)
        n_frames = len(positions)

        # Create output directory for proxy meshes
        obj_proxy_path = f"{config_obj.obj_path}/proxy"
        os.makedirs(obj_proxy_path, exist_ok=True)

        # Process each frame
        for frame_idx in range(n_frames):
            # Load original mesh for this frame
            vertices, normals, faces = _load_mesh(config_obj, frame_idx, use_proxy_path=False)

            # Get pose for this frame
            position = positions[frame_idx]
            rotation_euler = rotations[frame_idx]

            # Transform vertices to local coordinates
            from scipy.spatial.transform import Rotation
            R = Rotation.from_euler('XYZ', rotation_euler).as_matrix()
            vertices_local = (R.T @ (vertices - position).T).T

            # Compute bounding box extents in local coordinates
            min_coords = np.min(vertices_local, axis=0)
            max_coords = np.max(vertices_local, axis=0)
            extents = max_coords - min_coords

            # Center of bounding box
            center_local = (min_coords + max_coords) / 2

            # Generate proxy mesh based on proxy_type
            proxy_vertices_local, proxy_faces = self._generate_proxy_mesh(
                proxy_type=config_obj.proxy_type,
                extents=extents
            )

            # Center the proxy at the object's local center
            proxy_vertices_local = proxy_vertices_local + center_local

            # Compute normals for the proxy
            proxy_normals = self._compute_vertex_normals(proxy_vertices_local, proxy_faces)

            # Transform proxy back to world coordinates
            proxy_vertices_world = (R @ proxy_vertices_local.T).T + position
            proxy_normals_world = (R @ proxy_normals.T).T

            # Save proxy mesh
            output_file = f"{obj_proxy_path}/{config_obj.name}_{frame_idx:04d}.npz"
            np.savez_compressed(
                output_file,
                vertices=proxy_vertices_world.astype(np.float32),
                normals=proxy_normals_world.astype(np.float32),
                faces=proxy_faces.astype(np.int32)
            )

        print(f"Created {n_frames} proxy frames for {config_obj.name} at {obj_proxy_path}")

    def _generate_proxy_mesh(self, proxy_type: int, extents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a proxy mesh based on proxy_type, scaled to fit extents.

        Args:
            proxy_type: 0=octahedron, 1=dodecahedron, 2,3,4=icosahedron with subdivisions
            extents:: (dx, dy, dz) bounding box extents

        Returns:
            Tuple of (vertices, faces) for the proxy mesh
        """
        if proxy_type == 0:
            # Octahedron - no subdivision
            vertices, faces = self._create_octahedron()
        elif proxy_type == 1:
            # Dodecahedron - no subdivision
            vertices, faces = self._create_dodecahedron()
        elif proxy_type in [2, 3, 4]:
            # Icosahedron with subdivision
            subdivisions = proxy_type - 2
            vertices, faces = self._create_icosahedron(subdivisions=subdivisions)
        else:
            # Default to octahedron for unknown types
            print(f"Warning: Unknown proxy_type {proxy_type}, using octahedron")
            vertices, faces = self._create_octahedron()

        # Scale vertices to match extents (axis-aligned)
        # Normalize to unit sphere first, then scale
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vertices_normalized = vertices / norms

        # Scale by half-extents to inscribe the object
        half_extents = extents / 2.0
        vertices_scaled = vertices_normalized * half_extents[np.newaxis, :]

        return vertices_scaled, faces

    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Compute vertex normals for a mesh.

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (M, 3)

        Returns:
            Vertex normals (N, 3)
        """
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)

        # Initialize vertex normals
        vertex_normals = np.zeros_like(vertices)

        # Accumulate face normals for each vertex
        for i, face in enumerate(faces):
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normals[i]

        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vertex_normals = vertex_normals / norms

        return vertex_normals

    def _create_octahedron(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a regular octahedron centered at origin with unit circumradius.

        Returns:
            Tuple of (vertices, faces)
        """
        # Vertices of a unit octahedron
        vertices = np.array([
            [0, 0, 1],   # top
            [1, 0, 0],   # right
            [0, 1, 0],   # front
            [-1, 0, 0],  # left
            [0, -1, 0],  # back
            [0, 0, -1]   # bottom
        ], dtype=np.float64)

        # Faces (triangles)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4]
        ], dtype=np.int32)

        return vertices, faces

    def _create_dodecahedron(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a regular dodecahedron centered at origin with unit circumradius.

        Returns:
            Tuple of (vertices, faces)
        """
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        # All All 20 vertices of a regular dodecahedron
        vertices = np.array([
            # (±1, ±1, ±1)
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            # (0, ±1/φ, ±φ)
            [0, 1/phi, phi], [0, 1/phi, - -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
            # (±1/φ, ±φ, 0)
            [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
            # (±φ, 0, ±1/φ)
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ], dtype=np.float64)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # 12 pentagonal faces, each split into 3 triangles
        # Face definitions (pentagons)
        pentagon_faces = [
            [0, 8, 4, 14, 12],    # top front
            [0, 12, 2, 16, 8],    # top right
            [0, 16, 3, 17, 1],    # top back
            [0, 1, 13, 5, 14],    # top left
            [2, 10, 6, 18, 16],   # right front
            [2, 12, 0, 8, 10],    # right top
            [3, 16, 18, 7, 19],   # back right
            [3, 17, 1, 9, 19],    # back bottom
            [4, 14, 5, 9, 15],    # left front
            [4, 8, 0, 14],        # left top (incomplete in standard)
            [5, 13, 1, 9],        # left bottom (incomplete)
            [6, 18, 7, 15, 11],   # bottom front
            [7, 19, 9, 15],       # bottom back
            [10, 6, 11, 15, 4],   # bottom left
            [11, 15, 5, 9],       # bottom middle
            [12, 2, 10, 4],       # right front top
            [13, 5, 14, 12],      # front left
            [16, 3, 19, 7, 18],   # back right
            [17, 1, 13, 5, 9],    # back left
            [18, 7, 15, 4, 10],   # front bottom
        ]

        # Convert pentagons to triangles
        faces = []
        for pentagon in pentagon_faces:
            if len(pentagon) == 5:
                # Fan triangulation from first vertex
                for i in range(1, len(pentagon) - 1):
                    faces.append([pentagon[0], pentagon[i], pentagon[i + 1]])
            elif len(pentagon) == 4:
                # Quadrilateral to two triangles
                faces.append([pentagon[0], pentagon[1], pentagon[2]])
                faces.append([pentagon[0], pentagon[2], pentagon[3]])

        return vertices, np.array(faces, dtype=np.int32)

    def _create_icosahedron(self, subdivisions: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an icosahedron with optional subdivisions.

        Args:
            subdivisions: Number of subdivision steps (0=base icosahedron)

        Returns:
            Tuple of (vertices, faces)
        """
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        # Base icosahedron vertices
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Base icosahedron faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)

        # Subdivide if requested
        for _ in range(subdivisions):
            vertices, faces = self._subdivide_mesh(vertices, faces)

        return vertices, faces

    def _subdivide_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subdivide a triangular mesh once (each face -> 4 faces).

        Args:
            vertices: Current vertices
            faces: Current faces

        Returns:
            Tuple of (new_vertices, new_faces)
        """
        new_vertices = list(vertices)
        edge_map = {}
        new_faces = []

        for face in faces:
            v0, v1, v2 = face

            # Get or create midpoints for each edge
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            midpoints = []

            for edge in edges:
                key = tuple(sorted(edge))
                if key not in edge_map:
                    # Create new vertex at midpoint and project to sphere
                    midpoint = (new_vertices[edge[0]] + new_vertices[edge[1]]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    edge_map[key] = len(new_vertices)
                    new_vertices.append(midpoint)
                midpoints.append(edge_map[key])

            # Create 4 new faces
            v01, v12, v20 = midpoints
            new_faces.append([v0, v01, v20])
            new_faces.append([v1, v12, v01])
            new_faces.append([v2, v20, v12])
            new_faces.append([v01, v12, v20])

        return np.array(new_vertices), np.array(new_faces)
