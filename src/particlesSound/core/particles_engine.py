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

from dask import config as dask_config
dask_config.set({'num_workers': 1024, 'optimization.fuse.active': True, 'optimization.fuse.max_depth': 10,})

from pbrAudioCommon import EntityManager, ParticlesTrajectoryData, ResumeData
from pbrAudioCommon import _update_status
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.surface_voxel_size import SurfaceVoxelSize
from ..lib.surface_voxel_object import SurfaceVoxelObject
from ..core.particles_trajectory_solver import ParticlesTrajectorySolver
from ..core.particles_collisions import ParticlesCollisions
from ..core.particles_composer import ParticlesComposer
from ..core.particles_luthier import ParticlesLuthier
from ..core.particles_player import ParticlesPlayer

@dataclass
class particlesEngine:
    entity_manager: EntityManager
    particles_obj_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        resume_data = ResumeData(self.entity_manager)
        resume_data.load_data()

        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        os.makedirs(self.status_dir, exist_ok=True)
        
        self.voxel_size = None

        for p_config in config.particles:
            self.particles_obj_indices.append(p_config.idx)

    def bake(self):
        """Main bake function to run the entire particles sound synthesis pipeline."""
        _update_status(f"{self.status_dir}", "/bake", 0)

        with open(f"{self.status_dir}/step_done", 'r') as file:
            step_done = file.read().split()

        if not '_voxel_size' in step_done:
            self._voxel_size()
        if not '_voxel_obj' in step_done:
            self._voxel_size()
        if not '_traj' in step_done:
            self._traj()
        if not '_colls' in step_done:
            self._colls()
        if not '_comps' in step_done:
            self._comps()
        if not '_luthier' in step_done:
            self._luthier()
        if not '_player' in step_done:
            self._player()

    def _voxel_size(self):
        # 1. Compute a common voxel size for the scene
        voxel_size_calc = SurfaceVoxelSize(self.entity_manager)
        self.voxel_size = voxel_size_calc.compute()
        _update_status(f"{self.status_dir}", "/bake", 5)

    def _voxel_obj(self):
        # 2. Create and compute SurfaceVoxelObjects for all static/dynamic objects
        self._compute_surface_voxel_objects()
        _update_status(f"{self.status_dir}", "/bake", 20)

    def _traj(self):
        # 3. Compute ParticlesTrajectoryData for all particles objects
        self._compute_particles_trajectories()
        _update_status(f"{self.status_dir}", "/bake", 40)

    def _colls(self):
        # 4. Detect collisions for each particles object
        tasks_collisions = [self._detect_collisions(p_idx) for p_idx in self.particles_obj_indices]
        compute(*tasks_collisions)
        _update_status(f"{self.status_dir}", "/bake", 60)

    def _comps(self):
        # 5. Compose and determine convolver needs
        tasks_composer = [self._compose(p_idx) for p_idx in self.particles_obj_indices]
        compute(*tasks_composer)
        _update_status(f"{self.status_dir}", "/bake", 70)

    def _luthier(self):
        # 6. Luthier: Prepare convolver pools
        tasks_luthier = [self._luthier(p_idx) for p_idx in self.particles_obj_indices]
        compute(*tasks_luthier)
        _update_status(f"{self.status_dir}", "/bake", 80)

    def _player(self):
        # 7. Play: Synthesize the final audio
        tasks_player = [self._play(p_idx) for p_idx in self.particles_obj_indices]
        compute(*tasks_player)
        _update_status(f"{self.status_dir}", "/bake", 100)

    def _compute_surface_voxel_objects(self):
        """Creates and computes SurfaceVoxelObject for all mesh objects in the scene."""
        config = self.entity_manager.get('config')
        tasks = [self._voxelize_object(config_obj.idx) for config_obj in config.objects]
        compute(*tasks)

    def _compute_particles_trajectories(self):
        """Creates and computes ParticlesTrajectoryData for all particles objects."""
        config = self.entity_manager.get('config')
        for particles_config in config.particles:
            solver = ParticlesTrajectorySolver(self.entity_manager)
            solver.compute(particles_config.idx)

    @delayed
    def _voxelize_object(self, obj_idx: int):
        """Delayed task to compute a SurfaceVoxelObject."""
        voxel_obj = SurfaceVoxelObject(self.entity_manager, obj_idx, self.voxel_size)
        self.entity_manager.register('objects', voxel_obj)

    @delayed
    def _detect_collisions(self, particles_obj_idx: int):
        """Delayed task for collision detection."""
        collision_detector = ParticlesCollisions(self.entity_manager)
        collision_detector.compute(particles_obj_idx)

    @delayed
    def _compose(self, particles_obj_idx: int):
        """Delayed task for composing convolver requirements."""
        composer = ParticlesComposer(self.entity_manager)
        composer.compute(particles_obj_idx)

    @delayed
    def _luthier(self, particles_obj_idx: int):
        """Delayed task for preparing convolver pools."""
        luthier = ParticlesLuthier(self.entity_manager)
        luthier.compute(particles_obj_idx)

    @delayed
    def _play(self, particles_obj_idx: int):
        """Delayed task for audio synthesis."""
        player = ParticlesPlayer(self.entity_manager, particles_obj_idx)
        player.compute()

