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
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _load_mesh, _load_pose
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix


@dataclass
class ProxyMesh:
    """
    Generate low-resolution proxy meshes with consistent vertex indexing.

    proxy_type values:
    - 0: 4-vertex pyramid (axis-aligned)
    - 1: 6-vertex octahedron (axis-aligned)
    - 2: 8-vertex hexahedron/cube (axis-aligned)
    - 3,4,5: icosahedron with subdivision of (proxy_type - 3)

    The proxy mesh is generated once in local coordinates based on the
    bounding box of the original mesh at frame 0, then transformed to
    world coordinates for each frame using the pose data.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        # Cache for mesh data to avoid reloading
        self._mesh_cache = {}
        
        # Cache for proxy geometry templates (generated once per proxy_type)
        self._proxy_templates = {}

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
                    debug_print(
                        f"Creating proxy mesh for {config_obj.name} "
                        f"idx={config_obj.idx} proxy_type={config_obj.proxy_type}"
                    )
                    self._create_proxy_sequence(config_obj)
                    proxy_objects.append(config_obj.idx)

        return proxy_objects

    def _create_proxy_sequence(self, config_obj: Any) -> None:
        """
        Create proxy mesh sequence for an object.

        Strategy:
        1. Load pose data to get number of frames and transformations
        2. Load the original mesh at frame 0
        3. Generate proxy mesh template in local coordinates from frame 0 bbox
        4. For each frame, transform the proxy template to world coordinates
        """
        # Load pose data
        positions, rotations = _load_pose(config_obj)
        
        # Determine number of frames
        if config_obj.static:
            n_frames = 1
            # For static, positions/rotations are single arrays
            position_0 = positions
            rotation_0 = Rotation.from_euler('XYZ', rotations)
        else:
            n_frames = len(positions)
            position_0 = positions[0]
            rotation_0 = Rotation.from_euler('XYZ', rotations[0])

        # Create output directory
        obj_proxy_path = f"{config_obj.obj_path}/proxy"
        os.makedirs(obj_proxy_path, exist_ok=True)

        # Load original mesh at frame 0
        vertices_0, normals_0, faces_0 = self._load_mesh_cached(
            config_obj, 0, use_proxy_path=False
        )

        # Transform frame 0 vertices to local coordinates
        R0 = rotation_0.as_matrix()
        vertices_local_0 = (R0.T @ (vertices_0 - position_0).T).T

        # Compute bounding box in local coordinates
        min_coords = np.min(vertices_local_0, axis=0)
        max_coords = np.max(vertices_local_0, axis=0)
        extents = max_coords - min_coords
        center_local = (min_coords + max_coords) / 2

        # Generate proxy mesh template in local coordinates
        # This is done ONCE and reused for all frames
        proxy_vertices_local, proxy_faces = self._get_proxy_template(
            proxy_type=config_obj.proxy_type,
            extents=extents,
            center=center_local
        )

        # Validate and fix the proxy template
        proxy_vertices_local, proxy_faces = self._validate_and_fix_mesh(
            vertices=proxy_vertices_local,
            faces=proxy_faces,
            proxy_type=config_obj.proxy_type,
            extents=extents,
            center=center_local
        )

        # Compute vertex normals for the template (in local space)
        proxy_normals_local = self._compute_vertex_normals(
            proxy_vertices_local, proxy_faces
        )

        # Store expected dimensions for consistency
        expected_num_vertices = len(proxy_vertices_local)
        expected_num_faces = len(proxy_faces)

        debug_print(
            f"Proxy template for {config_obj.name}: "
            f"{expected_num_vertices} vertices, {expected_num_faces} faces"
        )

        # Process each frame
        for frame_idx in range(n_frames):
            # Get pose for this frame
            if config_obj.static:
                position = positions
                rotation = rotation_0
            else:
                position = positions[frame_idx]
                rotation = Rotation.from_euler('XYZ', rotations[frame_idx])

            # Transform proxy template from local to world coordinates
            R = rotation.as_matrix()
            proxy_vertices_world = (R @ proxy_vertices_local.T).T + position
            proxy_normals_world = (R @ proxy_normals_local.T).T

            # Ensure normals are normalized
            norm_mags = np.linalg.norm(proxy_normals_world, axis=1, keepdims=True)
            norm_mags[norm_mags == 0] = 1.0
            proxy_normals_world = proxy_normals_world / norm_mags

            # Save proxy mesh for this frame
            output_file = f"{obj_proxy_path}/{config_obj.name}_{frame_idx:04d}.npz"
            if config_obj.static:
                output_file = f"{obj_proxy_path}/{config_obj.name}.npz"
            
            np.savez_compressed(
                output_file,
                vertices=proxy_vertices_world.astype(np.float32),
                normals=proxy_normals_world.astype(np.float32),
                faces=proxy_faces.astype(np.int32)
            )

        debug_print(
            f"Created {n_frames} proxy frames for {config_obj.name} "
            f"at {obj_proxy_path}"
        )

    def _load_mesh_cached(self, config_obj: Any, frame_idx: int, 
                          use_proxy_path: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load mesh with caching to avoid repeated file file I/O.
        """
        cache_key = (config_obj.idx, frame_idx, use_proxy_path)
        
        if cache_key not in self._mesh_cache:
            vertices, normals, faces = _load_mesh(config_obj, frame_idx, use_proxy_path)
            self._mesh_cache[cache_key] = (vertices, normals, faces)
        
        return self._mesh_cache[cache_key]

    def _get_proxy_template(self, proxy_type: int, extents: np.ndarray, 
                            center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get or generate proxy mesh template.

        Templates are cached by proxy_type to avoid regeneration.
        
        Args:
            proxy_type: 0=pyramid, 1=octahedron, 2=cube, 3,4,5=icosahedron
            extents: (dx, dy, dz) bounding box extents
            center: Center of the bounding box in local coordinates

        Returns:
            Tuple of (vertices, faces) for the proxy mesh
        """
        # Check cache first
        cache_key = proxy_type
        if cache_key not in self._proxy_templates:
            # Generate template based on proxy_type
            if proxy_type == 0:
                vertices, faces = self._create_pyramid_4v(extents, center)
            elif proxy_type == 1:
                vertices, faces = self._create_octahedron_6v(extents, center)
            elif proxy_type == 2:
                vertices, faces = self._create_cube_8v(extents, center)
            else:
                # Icosahedron with subdivision
                subdivisions = proxy_type - 3 if proxy_type in [3, 4, 5] else 0
                vertices, faces = self._create_icosahedron(subdivisions)
                # Scale to match extents
                half_extents = extents / 2.0
                vertices = vertices * half_extents[np.newaxis, :] + center
            
            self._proxy_templates[cache_key] = (vertices, faces)
        
        return self._proxy_templates[cache_key]

    def _create_pyramid_4v(self, extents: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 4-vertex pyramid (axis-aligned).
        
        Vertices:
        - Vertex 0: apex at +x extent
        - Vertices 1-3: base at -x extent
        
        The pyramid has a triangular base in the YZ plane and apex at +x.
        """
        half_extents = extents / 2.0
        hx, hy, hz = half_extents

        # 4 vertices forming a pyramid
        vertices = np.array([
            [hx, 0.0, 0.0],                    # 0: Apex (+x)
            [-hx, -hy, -hz],                   # 1: Base corner 1
            [-hx, hy, -hz],                    # 2: Base corner 2
            [-hx, 0.0, hz],                    # 3: Base corner 3
        ], dtype=np.float64)

        # Center the vertices
        vertices = vertices + center

        # 4 triangular faces
        faces = np.array([
            [0, 1, 2],  # Front face
            [0, 2, 3],  # Right face  
            [0, 3, 1],  # Left face
            [1, 3, 2],  # Base face
        ], dtype=np.int32)

        return vertices, faces

    def _create_octahedron_6v(self, extents: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 6-vertex octahedron (axis-aligned).
        
        Vertices at the extents of each axis:
        - Vertex 0: +x, Vertex 1: -x
        - Vertex 2: +y, Vertex 3: -y
        - Vertex 4: +z, Vertex 5: -z
        """
        half_extents = extents / 2.0
        hx, hy, hz = half_extents

        # 6 vertices at axis extents
        vertices = np.array([
            [hx, 0.0, 0.0],    # 0: +x
            [-hx, 0.0, 0.0],   # 1: -x
            [0.0, hy, 0.0],    # 2: +y
            [0.0, -hy, 0.0],   # 3: - -y
            [0.0, 0.0, hz],    # 4: +z
            [0.0, 0.0, -hz]    # 5: -z
        ], dtype=np.float64)

        # Center the vertices
        vertices = vertices + center

        # 8 triangular faces
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
        
        Vertices are at the 8 corners of the bounding box.
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

        # 12 triangular faces (2 per cube face)
        faces = np.array([
            # Bottom face (z = -hz)
            [0, 1, 2], [0, 2, 3],
            # Top face (z = +hz)
            [4, 6, 5], [4, 7, 6],
            # Front face (y = +hy)
            [3, 2, 6], [3, 6, 7],
            # Back face (y = -hy)
            [0, 5, 1], [0, 4, 5],
            # Left face (x = -hx)
            [0, 3, 7], [0, 7, 4],
            # Right face (x = +hx)
            [1, 6, 2], [1, 5, 6]
        ], dtype=np.int32)

        return vertices, faces

    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Compute vertex normals for a mesh.
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
        
        Returns vertices centered at origin with unit circumradius.
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
                    norm = np.linalg.norm(midpoint)
                    if norm > 0:
                        midpoint = midpoint / norm
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

    def _validate_and_fix_mesh(self, vertices: np.ndarray, faces: np.ndarray, 
                               proxy_type: int, extents: np.ndarray, 
                               center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate that the proxy mesh has a valid volume (volume > 0) and fix it if not.
        """
        import trimesh

        # Create mesh for validation
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Check if volume is valid
        volume = mesh.volume
        is_valid = (
            not np.isnan(volume) and
            not np.isinf(volume) and
            volume > 0 and
            mesh.is_watertight and
            mesh.is_winding_consistent
        )

        if is_valid:
            return vertices, faces

        debug_print(f"Invalid proxy mesh detected (volume={volume}). Attempting to fix...")

        # Try different fixing strategies
        fixed_mesh = None

        # Strategy 1: Fix normals and fill holes
        try:
            if not mesh.is_winding_consistent:
                mesh.fix_normals()
            if not mesh.is_watertight:
                mesh.fill_holes()

            mesh.process(validate=True)

            if mesh.is_watertight and mesh.is_winding_consistent:
                volume = mesh.volume
                if not np.isnan(volume) and volume > 0:
                    fixed_mesh = mesh
                    debug_print(f"Mesh fixed via normal/hole fixing. New volume: {volume}")
        except Exception as e:
            debug_print(f"Fixing strategy 1 failed: {e}")

        # Strategy 2: Regenerate with minimum size
        if fixed_mesh is None:
            debug_print("Attempting to regenerate proxy mesh with minimum size...")

            # Ensure minimum extents
            min_extent = 0.001  # 1mm minimum
            fixed_extents = np.maximum(extents, min_extent)

            # Regenerate the proxy mesh
            fixed_vertices, fixed_faces = self._get_proxy_template(
                proxy_type=proxy_type,
                extents=fixed_extents,
                center=center
            )

            # Validate the regenerated mesh
            fixed_mesh = trimesh.Trimesh(vertices=fixed_vertices, faces=fixed_faces)
            volume = fixed_mesh.volume

            if np.isnan(volume) or volume <= 0:
                # Strategy 3: Use a simple cube as fallback
                debug_print("Regeneration failed. Using cube fallback...")
                half_extents = fixed_extents / 2
                hx, hy, hz = half_extents
                
                fixed_vertices = np.array([
                    [-hx, -hy, -hz], [hx, -hy, -hz],
                    [hx, hy, -hz], [-hx, hy, -hz],
                    [-hx, -hy, hz], [hx, -hy, hz],
                    [hx, hy, hz], [-hx, hy, hz]
                ]) + center

                fixed_faces = np.array([
                    [0, 1, 2], [0, 2, 3],  # Bottom
                    [4, 6, 5], [4, 7, 6],  # Top
                    [0, 3, 7], [0, 7, 4],  # Left
                    [1, 5, 6], [1, 6, 2],  # Right
                    [0, 4, 5], [0, 5, 1],  # Back
                    [3, 2, 6], [3, 6, 7]   # Front
                ])

                fixed_mesh = trimesh.Trimesh(vertices=fixed_vertices, faces=fixed_faces)
                debug_print(f"Created fallback cube with volume: {fixed_mesh.volume}")

        # Final validation
        if fixed_mesh is not None:
            volume = fixed_mesh.volume
            if not np.isnan(volume) and volume > 0:
                debug_print(f"Mesh validation passed. Final volume: {volume}")
                return fixed_mesh.vertices, fixed_mesh.faces

        # If all strategies fail, scale up the original mesh slightly
        debug_print("All fixing strategies failed. Scaling mesh slightly...")
        scale_factor = 1.1  # 10% scale up
        scaled_vertices = (vertices - center) * scale_factor + center

        # Recreate mesh with scaled vertices
        scaled_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=faces)

        if not scaled_mesh.is_winding_consistent:
            scaled_mesh.fix_normals()

        volume = scaled_mesh.volume
        if not np.isnan(volume) and volume > 0:
            debug_print(f"Scaled mesh has valid volume: {volume}")
            return scaled_mesh.vertices, scaled_mesh.faces

        # Absolute last resort: return a small valid cube
        debug_print("Using absolute fallback: minimal cube")
        size = max(np.max(extents), 0.01)  # At least 1cm
        half = size / 2
        fallback_vertices = np.array([
            [-half, -half, -half], [half, -half, -half],
            [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half],
            [half, half, half], [-half, half, half]
        ]) + center

        fallback_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
            [0, 4, 5], [0, 5, 1],
            [3, 2, 6], [3, 6, 7]
        ])

        return fallback_vertices, fallback_faces
