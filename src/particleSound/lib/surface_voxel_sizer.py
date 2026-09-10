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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class SurfaceVoxelSizer:
    """
    Computes an optimal voxel size for the entire scene to represent object
    surfaces with a low-density grid that respects material boundaries.
    """
    entity_manager: EntityManager
    target_percentile: float = 90.0  # Use the 90th percentile of feature sizes
    min_voxels_per_object: int = 1000 # A soft target to avoid overly coarse grids

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.voxel_size = None

    def compute(self) -> float:
        """
        Parses all objects, computes characteristic feature sizes of their
        material regions, and determines a single, optimal voxel size for the scene.

        Returns:
        --------
        float
            The computed voxel size for the scene.
        """
        if self.voxel_size is not None:
            return self.voxel_size

        debug_print("Computing optimal surface voxel size for the scene...")
        
        all_feature_sizes = []

        for config_obj in self.config.objects:
            try:
                # Load mesh at frame 0 as a representative
                vertices, normals, faces = _load_mesh(config_obj, 0)
                if vertices is None or len(vertices) == 0:
                    continue
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

                # Feature size can be approximated by the average edge length.
                edge_lengths = mesh.edges_unique_length
                if len(edge_lengths) > 0:
                    avg_feature_size = np.mean(edge_lengths)
                    all_feature_sizes.append(avg_feature_size)
                    debug_print(f"Object '{config_obj.name}': avg_feature_size={avg_feature_size:.5f}m")

            except Exception as e:
                debug_print(f"Could not process object '{config_obj.name}': {e}")
                continue

        if not all_feature_sizes:
            self.voxel_size = 0.05  # 5cm
            debug_print(f"No valid feature sizes found. Fallback to default voxel size: {self.voxel_size}m")
            return self.voxel_size

        # Use a percentile to be robust against very large or very small objects.
        # A low percentile ensures we capture the details of the most detailed objects.
        self.voxel_size = np.percentile(all_feature_sizes, self.target_percentile)
        
        # Clamp the voxel size to a reasonable range to prevent extremes
        self.voxel_size = np.clip(self.voxel_size, 0.01, 1) # 1cm to 1m

        debug_print(f"Computed scene-wide surface voxel size: {self.voxel_size:.5f}m")
        return self.voxel_size

