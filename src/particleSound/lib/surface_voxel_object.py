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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class SurfaceVoxelObject:
    """
    Manages a time-varying voxel grid representation of a single object's surface.

    The grid is computed once in the object's local space and then transformed
    for each frame using the object's trajectory data.
    """
    entity_manager: EntityManager
    voxel_size: float
    obj_idx: int

    # Internal state
    _base_voxel_grid: Optional[trimesh.voxel.VoxelGrid] = None
    _material_map: Optional[Dict[Tuple[int, int, int], Any]] = None

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.config = config
        
        # Pre-compute the base voxel grid in local coordinates
        self._compute_base_voxel_grid()

    def _compute_base_voxel_grid(self):
        """
        Computes the initial voxel grid in the object's local coordinate frame.
        This is done once and then transformed for each frame.
        """
        debug_print(f"Computing base surface voxel grid for '{self.config_obj.name}'...")
        
        config = self.entity_manager.get('config')
        trajectories = self.entity_manager.get('trajectories')
        for config_obj in self.config.objects:
            if config_obj.idx == self.obj_idx:
                for t_idx in trajectories.keys():
                    if hasattr(trajectories[t_idx], 'obj_idx'):
                        if trajectories[t_idx].obj_idx == self.obj_idx:
                            self.trajectory = trajectories[t_idx]
                break
        
        try:
            vertices, normals, faces = _load_mesh(config_obj, 0, use_proxy_path=False)
            if vertices is None or len(vertices) == 0:
                debug_print(f"Could not load mesh for '{config_obj.name}'.")
                return

            pos_0 = self.trajectory.get_position(0)
            rot_0 = Rotation.from_euler('XYZ', self.trajectory.get_rotation(0))
            
            # Transform world-space vertices to local space
            R_inv = rot_0.inv().as_matrix()
            local_vertices = (R_inv @ (vertices - pos_0).T).T

            local_mesh = trimesh.Trimesh(vertices=local_vertices, faces=faces, vertex_normals=normals)

            # Create the voxel grid from the local mesh
            # `surface=True` ensures we only get voxels on the surface, not the volume.
            self._base_voxel_grid = local_mesh.voxelized(pitch=self.voxel_size, method='subdivide').fill()
            
            # For now, we assume a single material for the whole object.
            # If per-face materials were supported, we would populate this map
            # by checking which material each face belongs to.
            self._material_map = {}
            for voxel_coord in self._base_voxel_grid.sparse_indices:
                self._material_map[tuple(voxel_coord)] = config_obj.acoustic_shader

            debug_print(f"Base grid for '{config_obj.name}' created with {len(self._base_voxel_grid.sparse_indices)} voxels.")

        except Exception as e:
            debug_print(f"Failed to compute base voxel grid for '{config_obj.name}': {e}")
            self._base_voxel_grid = None

    def update_voxels(self, sample_idx: float) -> Tuple[Optional[trimesh.voxel.VoxelGrid], Optional[np.ndarray]]:
        """
        Returns the base voxel grid and a transformation matrix for the given sample.

        The caller can use these to transform the grid into world space without
        needing to re-voxelize the mesh at every timestep.

        Parameters:
        -----------
        sample_idx : float
            The time sample at which to get the voxel grid's state.

        Returns:
        --------
        Tuple[Optional[trimesh.voxel.VoxelGrid], Optional[np.ndarray]]
            A tuple containing the base VoxelGrid and a 4x4 transformation matrix,
            or (None, None) if the grid could not be created.
        """
        if self._base_voxel_grid is None:
            return None, None

        # Get the object's pose at the requested sample
        position = self.trajectory.get_position(sample_idx)
        rotation = Rotation.from_euler('XYZ', self.trajectory.get_rotation(sample_idx))
        
        # Create the transformation matrix (rotation + translation)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = position

        return self._base_voxel_grid, transform_matrix

    def get_material_at_voxel(self, local_voxel_coord: Tuple[int, int, int]) -> Optional[Any]:
        """
        Retrieves the acoustic material associated with a voxel in the base grid.

        Parameters:
        -----------
        local_voxel_coord : Tuple[int, int, int]
            The integer coordinates of the voxel in the local grid.

        Returns:
        --------
        Optional[Any]
            The AcousticShader for that voxel, or None if the voxel is empty.
        """
        if self._material_map is None:
            return None
        return self._material_map.get(local_voxel_coord)

    @property
    def base_voxel_grid(self) -> Optional[trimesh.voxel.VoxelGrid]:
        """Returns the base VoxelGrid object."""
        return self._base_voxel_grid

