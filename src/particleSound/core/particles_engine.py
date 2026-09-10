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
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass, field
from dask import delayed, compute

# Configure Dask to use more threads
from dask import config as dask_config
dask_config.set({'num_workers': 1024, 'optimization.fuse.active': True, 'optimization.fuse.max_depth': 10,})

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..core.particles_trajectory_solver import ParticlesTrajectorySolver
from ..core.particles_collision_solver import ParticlesCollisionSolver
from ..lib.surface_voxel_sizer import SurfaceVoxelSizer
from ..lib.surface_voxel_object import SurfaceVoxelObject

@dataclass
class ParticlesEngine:
    entity_manager: EntityManager

    def __post_init__(self):
        # ToDo: from pbrAudioCommon functions charge EntityManager if empty
        pass

    def bake(self):
        config = self.entity_manager.get('config')
        tasks_traj = [self._traj(particle_cfg.idx) for particle_cfg in config.particles]
        results_traj = compute(*tasks_traj)

        # Initialize the voxel size calculator
        voxel_sizer = SurfaceVoxelSizer(self.entity_manager)
        self.scene_voxel_size = voxel_sizer.compute()

        tasks_svobj = [self._svobj(config_obj.idx) for config_obj in config.objects]
        results_svobj = compute(*tasks_svobj)

        tasks_colls = [self._colls(particle_cfg.idx) for particle_cfg in config.particles]
        results_colls = compute(*tasks_colls)

    @delayed
    def _traj(self, particle_idx: int):
        parts = ParticlesTrajectorySolver(self.entity_manager)
        parts.compute(particle_idx)

    @delayed
    def _svobj(self, obj_idx: int):
        surf_voxel_obj = SurfaceVoxelObject(entity_manager=self.entity_manager, voxel_size=self.scene_voxel_size, obj_idx=obj_idx)
        _ = self.entity_manager.register('objects', surf_voxel_obj)

    @delayed
    def _colls(self, particle_idx: int):
        colls = ParticlesCollisionSolver(self.entity_manager)
        colls.compute(particle_idx)
        colls.save_all_data()
