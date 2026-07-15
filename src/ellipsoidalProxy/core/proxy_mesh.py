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

# ./ellipsoidalProxy/core/proxy_mesh.py

import os
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _load_pose


@dataclass
class ProxyMesh:
    """
    Generate low-resolution proxy meshes with consistent vertex indexing.

    proxy_type values:
    - 0: 6-vertex octahedron (axis-aligned, 6 vertices at extents)
    - 1: 8-vertex hexahedron/cube (axis-aligned, 8 vertices at corners)
    - 2,3,4: icosahedron with subdivision of (proxy_type - 2)

    The proxy mesh vertices maintain consistent indexing across frames
    by mapping to the original mesh's extremal vertices.
    """
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> List[int]:
        """
        Compute proxy mesh for an object.

        Args:
            obj_idx: Object index

        Returns:
            List of object indices that had proxies created
        """
        config = self.entity_manager.get('config')
        proxy_objects = []

        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                if config_obj.proxy_type is not False:
                    print(f"Creating proxy mesh for {config_obj.name} idx={config_obj.idx} proxy_type={config_obj.proxy_type}")
                    self._create_pro_proxy_sequence(config_obj)
                    proxy_objects.append(config_obj.idx)

        return proxy_objects

    def _create_proxy_sequence(self, config_obj: Any) -> None:
        """
        Create proxy mesh sequence for an object.

        For each frame, generates a proxy mesh that maintains consistent
        vertex indexing by mapping to the original mesh's extremal vertices.
        """
        # Load pose data to get number of frames
        positions, rotations = _load_pose(config_obj)
        n_frames = len(positions)

        # Create output directory for proxy meshes
        obj_proxy_path = f"{config_obj.obj_path}/proxy"
        os.makedirs(obj_proxy_path, exist_ok=True)

        # Process first frame to establish vertex mapping
        vertices_0, normals_0, faces_0 = _load_mesh(config_obj, 0, use_proxy_path=False)
        position_0 = positions[0]
        rotation_0 = Rotation.from_euler('XYZ', rotations[0])

        # Transform first frame vertices to local coordinates
        R0 = rotation_0.as_matrix()
        vertices_local_0 = (R0.T @ (vertices_0 - position_0).T).T

        # Compute bounding box in in local coordinates for first frame
        min_coords_0 = np.min(vertices_local_0, axis=0)
        max_coords_0 = np.max(vertices_local_0, axis=0)
        extents_0 = max_coords_0 - min_coords_0
        center_local_0 = (min_coords_0 + max_coords_0) / 2

        # Generate proxy vertices for first frame (reference)
        proxy_vertices_local_0, proxy_faces = self._generate_proxy_mesh(
            proxy_type=config_obj.proxy_type,
            extents=extents_0,
            center=center_local_0
        )

        # Build KD-tree for first frame's original vertices
        tree_original_0 = cKDTree(vertices_local_0 - center_local_0)

        # For each proxy vertex, find the nearest original vertex
        # This establishes the consistent vertex mapping
        proxy_vertex_mapping = []
        for pv in proxy_vertices_local_0:
            # Normalize proxy vertex to match original vertex search space
            pv_normalized = pv - center_local_0
            _, idx = tree_original_0.query(pv_normalized)
            proxy_vertex_mapping.append(idx)

        # Process each frame
        for frame_idx in range(n_frames):
            # Load original mesh for this frame
            vertices, normals, faces = _load_mesh(config_obj, frame_idx, use_proxy_path=False)

            # Get pose for this frame
            position = positions[frame_idx]
            rotation_euler = rotations[frame_idx]

            # Transform vertices to local coordinates
            R = Rotation.from_euler('XYZ', rotation_euler).as_matrix()
            vertices_local = (R.T @ (vertices - position).T).T

            # Compute bounding box extents in local coordinates
            min_coords = np.min(vertices_local, axis=0)
            max_coords = np.max(vertices_local, axis=0)
            extents = max_coords - min_coords
            center_local = (min_coords + max_coords) / 2

            # Generate proxy mesh based on proxy_type
            proxy_vertices_local, proxy_faces = self._generate_proxy_mesh(
                proxy_type=config_obj.proxy_type,
                extents=extents,
                center=center_local
            )

            # Now re-index proxy vertices to maintain consistency
            # For each proxy vertex, find the corresponding original vertex
            # using the mapping established from the first frame
            tree_original = cKDTree(vertices_local - center_local)
            
            reindexed_proxy_vertices = np.zeros_like(proxy_vertices_local)
            for i, orig_idx in enumerate(proxy_vertex_mapping):
                # Use the original vertex position as the proxy vertex
                # This ensures consistent indexing across frames
                reindexed_proxy_vertices[i] = vertices_local[orig_idx]

            # Center the proxy at the object's local center
            reindexed_proxy_vertices = reindexed_proxy_vertices

            # Compute normals for the proxy
            proxy_normals = self._compute_vertex_normals(reindexed_proxy_vertices, proxy_faces)

            # Transform proxy back to world coordinates
            proxy_vertices_world = (R @ reindexed_proxy_vertices.T).T + position
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

    def _generate_proxy_mesh(self, proxy_type: int, extents: np.ndarray, 
                             center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a proxy mesh based on proxy_type, scaled to fit extents.

        Args:
            proxy_type: 0=6-vertex octahedron, 1=8-vertex cube, 2,3,4=icosahedron
            extents: (dx, dy, dz) bounding box extents
            center: Center of the bounding box in local coordinates

        Returns:
            Tuple of (vertices, faces) for the proxy mesh
        """
        if proxy_type == 0:
            # 6-vertex octahedron - vertices at axis extents
            vertices, faces = self._create_octahedron_6v(extents, center)
        elif proxy_type == 1:
            # 8-vertex cube - vertices at bounding box corners
            vertices, faces = self._create_cube_8v(extents, center)
        elif proxy_type in [2, 3, 4]:
            # Icosahedron with subdivision
            subdivisions = proxy_type - 2
            vertices, faces = self._create_icosahedron(subdivisions=subdivisions)
            # Scale to match extents
            half_extents = extents / 2.0
            vertices = vertices * half_extents[np.newaxis, :] + center
        else:
            # Default to 6-vertex octahedron for unknown types
            print(f"Warning: Unknown proxy_type {proxy_type}, using 6-vertex octaahedron")
            vertices, faces = self._create_octahedron_6v(extents, center)

        return vertices, faces

    def _create_octahedron_6v(self, extents: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 6-vertex octahedron (axis-aligned).
        
        Vertices are at the extents of each axis:
        - Vertex 0: +x extent
        - Vertex 1: -x extent
        - Vertex 2: +y extent
        - Vertex 3: -y extent
        - Vertex 4: +z extent
        - Vertex 5: -z extent

        Args:
            extents: (dx, dy, dz) bounding box extents
            center: Center position

        Returns:
            Tuple of (6 vertices, 8 triangular faces)
        """
        half_extents = extents / 2.0

        # 6 vertices at axis extents
        vertices = np.array([
            [half_extents[0], 0.0, 0.0],   # 0: +x
            [-half_extents[0], 0.0, 0.0],  # 1: -x
            [0.0, half_extents[1], 0.0],   # 2: +y
            [0.0, -half_extents[1], 0.0],  # 3: -y
            [0.0, 0.0, half_extents[2]],   # 4: +z
            [0.0, 0.0, -half_extents[2]]   # 5: -z
        ], dtype=np.float64)

        # Center the vertices
        vertices = vertices + center

        # 8 triangular faces forming an octahedron
        faces = np.array([
            [0, 2, 4],  # +x, +y, +z
            [0, 4, 3],  # +x, +z, -y
            [0, 3, 5],  # +x, -y, -z
            [0, 5, 2],  # +x, -z, +y
            [1, 4, 2],  # -x, +z, +y
            [1, 3, 4],  # -x, -y, +z
            [1, 5, 3],  # -x, -z, -y
            [1, 2, 5]   # -x, +y, -z
        ], dtype=np.int32)

        return vertices, faces

    def _create_cube_8v(self, extents: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an 8-vertex cube (hexahedron) axis-aligned.
        
        Vertices are at the 8 corners of the bounding box:
        - Vertex 0: (-x, -y, -z)
        - Vertex 1: (+x, -y, -z)
        - Vertex 2: (+x, +y, -z)
        - Vertex 3: (-x, +y, -z)
        - Vertex 4: (-x, -y, +z)
        - Vertex 5: (+x, -y, +z)
        - Vertex 6: (+x, +y, +z)
        - Vertex 7: (-x, +y, +z)

        Args:
            extents: (dx, dy, dz) bounding box extents
            center: Center position

        Returns:
            Tuple of (8 vertices, 12 triangular faces)
        """
        half_extents = extents / 2.0
        hx, hy, hz = half_extents

        # 8 vertices at bounding box corners
        vertices = np.array([
            [-hx, -hy, -hz],  # 0: bottom-back-left
            [ hx, -hy, -hz],  # 1: bottom-back-right
            [ hx,  hy, -hz],  # 2: bottom-front-right
            [-hx,  hy, -hz],  # 3: bottom-front-left
            [-hx, -hy,  hz],  # 4: top-back-left
            [ hx, -hy,  hz],  # 5: top-back-right
            [ hx,  hy,  hz],  # 6: top-front-right
            [-hx,  hy,  hz]   # 7: top-front-left
        ], dtype=np.float64)

        # Center the vertices
        vertices = vertices + center

        # 12 triangular faces (2 triangles per face of the cube)
        faces = np.array([
            # Bottom face (z = -hz)
            [0, 1, 2],
            [0, 2, 3],
            # Top face (z = +hz)
            [4, 6, 5],
            [4, 7, 6],
            # Front face (y = +hy)
            [3, 2, 6],
            [3, 6, 7],
            # Back face (y = -hy)
            [0, 5, 1],
            [0, 4, 5],
            # Left face (x = -hx)
            [0, 3, 7],
            [0, 7, 4],
            # Right face (x = +hx)
            [1, 6, 2],
            [1, 5, 6]
        ], dtype=np.int32)

        return vertices, faces

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

    def _create_icosahedron(self, subdivisions: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an icosahedron with optional subdivisions.

        Args:
            subdivisions: Number of subdivision steps (0=base icosahedron)

        Returns:
            Tuple of (vertices, faces) centered at origin with unit circumradius
        """
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        # Base icosahedron vertices
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi,  0, -1], [-phi, 0, 1]
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
        for _ in range(sub(subdivisions):
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
