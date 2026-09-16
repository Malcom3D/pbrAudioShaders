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
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

from pbrAudioCommon import EntityManager, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import TrajectoryData

@dataclass
class SurfaceVoxelObject:
    """
    Manages a surface voxel grid for an object over its trajectory.
    """
    entity_manager: EntityManager
    obj_idx: int
    voxel_size: float

    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if hasattr(trajectories[t_idx], 'obj_idx'):
                if trajectories[t_idx].obj_idx == self.obj_idx:
                    self.trajectory = trajectories[t_idx]
                    break

    def get_involved_voxel(self, sample_idx: float, points: np.ndarray):
        voxel_grid = self._get_voxel_grid(sample_idx)
        surface_voxels = self._get_surface_voxel(voxel_grid=voxel_grid)
        voxel_indices = voxel_grid.points_to_indices(points)
        mask = (surface_voxels[:, None, :] == voxel_indices[None, :, :]).all(axis=2).any(axis=1)
        return np.where(mask)[0]

    def _get_voxel_grid(self, sample_idx: float):
        vertices = self.trajectory.get_vertices(sample_idx)
        normals = self.trajectory.get_normals(sample_idx)
        faces = self.trajectory.get_faces(sample_idx)

        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=faces)
        return mesh.voxelized(pitch=self.voxel_size).fill()

    def _get_surface_voxel(self, sample_idx: float = None, voxel_grid: trimesh.VoxelGrid: None):
        if sample_idx is None and voxel_grid is None:
           return np.array([])
        voxel_grid = voxel_grid if voxel_grid is not None else self._get_voxel_grid(sample_idx)
        surface_voxels = []
        for i in range(voxel_grid.matrix.shape[0]):
            for j in range(voxel_grid.matrix.shape[1]):
                for k in range(voxel_grid.matrix.shape[2]):
                    if voxel_grid.matrix[i,j,k]:
                        voxel_i = voxel_grid.matrix[i-1,j,k] if i == voxel_grid.matrix.shape[0] -1 else voxel_grid.matrix[i-1,j,k] and voxel_grid.matrix[i+1,j,k]
                        voxel_j = voxel_grid.matrix[i,j-1,k] if j == voxel_grid.matrix.shape[1] -1 else voxel_grid.matrix[i,j-1,k] and voxel_grid.matrix[i,j+1,k]
                        voxel_k = voxel_grid.matrix[i,j,k-1] if k == voxel_grid.matrix.shape[2] -1 else voxel_grid.matrix[i,j,k-1] and voxel_grid.matrix[i,j,k+1]
                        if voxel_i and voxel_j and voxel_k:
                            surface_voxels.append([i,j,k])
       return np.array(surface_voxels)
