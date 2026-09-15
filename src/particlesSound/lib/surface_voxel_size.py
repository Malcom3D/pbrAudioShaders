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

import numpy as np
from typing import Any, List, Dict
from dataclasses import dataclass
from psutil import cpu_count

from pbrAudioCommon import EntityManager, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class SurfaceVoxelSize:
    """
    Computes a common voxel size for the scene based on computational power
    and object complexity, ensuring each voxel predominantly contains a single material.
    """
    entity_manager: EntityManager
    voxel_size: float = None

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.physical_core = config.system.physical_core or cpu_count(logical=False)

    def compute(self) -> float:
        """
        Computes and returns the common surface voxel size for the scene.
        """
        config = self.entity_manager.get('config')
        
        total_surface_area = 0
        total_objects = 0
        
        # Estimate total surface area of all objects in the scene
        for config_obj in config.objects:
            if config_obj.static:
                try:
                    vertices, _, faces = _load_mesh(config_obj, 0)
                    if vertices.size > 0 and faces.size > 0:
                        # A more robust area calculation would use trimesh, but this is a quick estimate
                        # For now, we'll just use the bounding box surface area as a proxy
                        min_coords = np.min(vertices, axis=0)
                        max_coords = np.max(vertices, axis=0)
                        extents = max_coords - min_coords
                        surface_area_estimate = 2 * (extents[0]*extents[1] + extents[1]*extents[2] + extents[0]*extents[2])
                        total_surface_area += surface_area_estimate
                        total_objects += 1
                except Exception as e:
                    debug_print(f"Could not load mesh for {config_obj.name} to estimate surface area: {e}")

        if total_objects == 0:
            self.voxel_size = 0.1  # Default if no objects
            debug_print(f"No static objects found. Defaulting voxel size to {self.voxel_size}m")
            return self.voxel_size

        # Heuristic: We want to distribute the voxelization workload across CPU cores.
        # Let's assume each core can handle a certain number of voxels efficiently.
        # This is a tunable parameter based on performance testing.
        voxels_per_core = 500000 
        
        max_total_voxels = self.physical_core * voxels_per_core
        
        # voxel_size = sqrt(surface_area / num_voxels)
        # We want num_voxels <= max_total_voxels
        # so voxel_size >= sqrt(surface_area / max_total_voxels)
        
        min_voxel_size = np.sqrt(total_surface_area / max_total_voxels)

        # Clamp to a reasonable range (e.g., 1cm to 1m)
        self.voxel_size = np.clip(min_voxel_size, 0.01, 1.0)
        
        debug_print(f"Computed common voxel size: {self.voxel_size:.4f}m for {total_objects} objects with total area ~{total_surface_area:.2f}m²")
        
        return self.voxel_size
