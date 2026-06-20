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
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _load_pose

@dataclass
class EllipsoidalProxy:
    """
    Generate ellipsoidal proxies for small objects.

    For objects with maximum dimension < size_threshold, replaces the detailed
    mesh with an axis-aligned ellipsoid that inscribes the object.
    """
    entity_manager: EntityManager
    size_threshold: float = 0.1  # meters - objects smaller than this become proxies

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.cache_path = config.system.cache_path
        self.proxy_path = f"{self.cache_path}/ellipsoidal_proxy"
        os.makedirs(self.proxy_path, exist_ok=True)

    def process_all_objects(self) -> List[int]:
        """
        Process all objects and replace small ones with ellipsoidal proxies.

        Returns:
            List of object indices that were replaced with proxies
        """
        config = self.entity_manager.get('config')
        proxy_objects = []

        for config_obj in config.objects:
            if self._should_replace_with_proxy(config_obj):
                print(f"Replacing {config_obj.name} (idx={config_obj.idx}) with ellipsoidal proxy")
                self._create_proxy_sequence(config_obj)
                proxy_objects.append(config_obj.idx)

        return proxy_objects

    def _should_replace_with_proxy(self, config_obj: Any) -> bool:
        """
        Determine if an object should be replaced with an ellipsoidal proxy.

        Criteria:
        - Object is not static
        - Maximum dimension < size_threshold
        """
        if config_obj.static:
            return False

        # Load first frame to check size
        try:
            vertices, _, _ = _load_mesh(config_obj, 0)

            # Compute bounding box dimensions
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
            max_dimension = np.max(dimensions)

            # Check if object is small enough
            if max_dimension < self.size_threshold:
                # Also compute volume ratio to avoid replacing flat/long objects
                # that might have one dimension large but are still small in volume
                volume = self._compute_volume(vertices)
                bounding_box_volume = np.prod(dimensions)

                if bounding_box_volume > 0:
                    volume_ratio = volume / bounding_box_volume
                    # Replace if object fills reasonable portion of bounding box
                    # (avoids replacing thin shells or sparse point clouds)
                    return volume_ratio > 0.1

            return False

        except Exception as e:
            print(f"Error checking size for {config_obj.name}: {e}")
            return False

    def _compute_volume(self, vertices: np.ndarray) -> float:
        """Estimate volume from vertex cloud using convex hull."""
        if len(vertices) < 4:
            return 0.0

        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            # Fallback: approximate as bounding box volume / 2
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
            return np.prod(dimensions) / 2.0

    def _create_proxy_sequence(self, config_obj: Any) -> None:
        """
        Create ellipsoidal proxy mesh sequence for an object.

        For each frame, generates an ellipsoid that inscribes the object
        at that frame's position and orientation.
        """
        # Load pose data to get number of frames
        positions, rotations = _load_pose(config_obj)
        n_frames = len(positions)

        # Create output directory for proxy meshes
        obj_proxy_path = f"{self.proxy_path}/{config_obj.name}"
        os.makedirs(obj_proxy_path, exist_ok=True)

        # Process each frame
        for frame_idx in range(n_frames):
            # Load original mesh for this frame
            vertices, normals, faces = _load_mesh(config_obj, frame_idx)

            # Get pose for this frame
            position = positions[frame_idx]
            rotation_euler = rotations[frame_idx]

            # Transform vertices to local coordinates
            R = Rotation.from_euler('XYZ', rotation_euler).as_matrix()
            vertices_local = (R.T @ (vertices - position).T).T

            # Compute ellipsoid parameters in local coordinates
            ellipsoid_params = self._compute_inscribing_ellipsoid(vertices_local)

            # Generate proxy mesh (low-resolution ellipsoid)
            proxy_vertices, proxy_normals, proxy_faces = self._generate_ellipsoid_mesh(
                ellipsoid_params, resolution=2  # Very low resolution (2 subdivisions)
            )

            # Transform proxy back to world coordinates
            proxy_vertices_world = (R @ proxy_vertices.T).T + position

            # Save proxy mesh
            output_file = f"{obj_proxy_path}/frame_{frame_idx:04d}.npz"
            np.savez_compressed(
                output_file,
                vertices=proxy_vertices_world.astype(np.float32),
                normals=proxy_normals.astype(np.float32),
                faces=proxy_faces.astype(np.int32)
            )

        print(f"Created {n_frames} proxy frames for {config_obj.name}")

    def _compute_inscribing_ellipsoid(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute an axis-aligned ellipsoid that inscribes the vertex cloud.

        Returns:
            Tuple of (center, radii) where radii are half-lengths along each axis
        """
        # Center at origin (vertices are already in local coordinates)
        center = np.zeros(3)

        # Find bounding box extents
        max_coords = np.max(vertices, axis=0)
        min_coords = np.min(vertices, axis=0)

        # Radii are half the bounding box dimensions
        radii = (max_coords - min_coords) / 2.0

        # Ensure minimum radius to avoid degenerate ellipsoids
        radii = np.maximum(radii, 0.001)

        return center, radii

    def _generate_ellipsoid_mesh(self, ellipsoid_params: Tuple[np.ndarray, np.ndarray],
                                 resolution: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a low-resolution ellipsoid mesh.

        Args:
            ellipsoid_params: (center, radii) tuple
            resolution: Number of subdivisions (2 = very low resolution, 3 = low, 4 = medium)

        Returns:
            Tuple of (vertices, normals, faces)
        """
        center, radii = ellipsoid_params

        # Start with a unit sphere (icosphere for better triangle distribution)
        if resolution <= 2:
            # Very low resolution - use octahedron
            vertices, faces = self._create_octahedron()
        elif resolution == 3:
            # Low resolution - use icosphere with 1 subdivision
            vertices, faces = self._create_icosphere(subdivisions=1)
        else:
            # Medium resolution - use icosphere with 2 subdivisions
            vertices, faces = self._create_icosphere(subdivisions=2)

        # Scale vertices by radii to create ellipsoid
        vertices_scaled = vertices * radii

        # Compute normals (for ellipsoid, normals are just normalized vertex positions)
        normals = vertices_scaled / (np.linalg.norm(vertices_scaled, axis=1, keepdims=True) + 1e-10)

        # Translate to center
        vertices_scaled += center

        return vertices_scaled.astype(np.float32), normals.astype(np.float32), faces.astype(np.int32)

    def _create_octahedron(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simple octahedron (8 faces)."""
        # Vertices of a unit octahedron
        vertices = np.array([
            [0, 0, 1],   # top
            [1, 0, 0],   # right
            [0, 1, 0],   # front
            [-1, 0, 0],  # left
            [0, -1, 0],  # back
            [0, 0, -1]   # bottom
        ])

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
        ])

        return vertices, faces

    def _create_icosphere(self, subdivisions: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an icosphere by subdividing an icosahedron.

        Args:
            subdivisions: Number of subdivision steps

        Returns:
            Tuple of (vertices, faces)
        """
        # Start with icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])

        # Normalize vertices to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])

        # Subdivide
        for _ in range(subdivisions):
            vertices, faces = self._subdivide_mesh(vertices, faces)

        return vertices, faces

    def _subdivide_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subdivide a triangular mesh once.

        Each triangle is split into 4 smaller triangles by adding vertices
        at the midpoints of each edge.
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
                    # Create new vertex at midpoint
                    midpoint = (new_vertices[edge[0]] + new_vertices[edge[1]]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)  # Project to sphere
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

    def integrate_with_physics_solver(self) -> None:
        """
        Update the physics solver configuration to use proxy meshes.

        This modifies the object paths in the entity manager to point to
        the proxy mesh sequences instead of the original meshes.
        """
        config = self.entity_manager.get('config')

        for config_obj in config.objects:
            if self._should_replace_with_proxy(config_obj):
                # Update obj_path to point to proxy meshes
                config_obj.obj_path = f"{self.proxy_path}/{config_obj.name}"
                print(f"Updated {config_obj.name} obj_path to proxy meshes")

        # Re-register config to ensure changes are picked up
        self.entity_manager.register('config', config)
