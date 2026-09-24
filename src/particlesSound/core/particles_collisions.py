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
from typing import Any, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy.spatial import cKDTree

from pbrAudioCommon import EntityManager, ParticlesTrajectoryData
from pbrAudioCommon import _load_mesh, _load_particle
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from pbrAudioCommon import TrajectoryData

from ..lib.surface_voxel_object import SurfaceVoxelObject
from ..lib.particles_collisions_points import ParticlesCollisionsPoints
from ..lib.particles_collisions_voxels import ParticlesCollisionsVoxels

@dataclass
class ParticlesCollisions:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.collisions_dir = f"{config.system.cache_path}/particles_collisions"
        os.makedirs(self.collisions_dir, exist_ok=True)

    def compute(self, particles_obj_idx: int):
        config = self.entity_manager.get('config')

        particles_config_obj = next((p for p in config.particles if p.idx == particles_obj_idx), None)
        if not particles_config_obj:
            debug_print(f"No particles config found for idx {particles_obj_idx}")
            return

        # Get the particles trajectory data
        particles_trajectories = self.entity_manager.get('trajectories')
        for p_key in particles_trajectories.keys():
            if isinstance(particles_trajectories[p_key], ParticlesTrajectoryData):
                if particles_trajectories[p_key].particles_idx == particles_obj_idx:
                    particles_traj = particles_trajectories[p_key]
                    break
            debug_print(f"No ParticlesTrajectoryData found for idx {particles_obj_idx}")
            return

        # Get all other objects' trajectories and voxel data
        voxel_objects = {} 
        objects = self.entity_manager.get('surface_voxel_objects')
        for o_idx in objects.keys():
            if isinstance(objects[o_idx], SurfaceVoxelObject):
                voxel_objects[objects[o_idx].obj_idx] = objects[o_idx]
        
        if particles_config_obj.proxy: # Massive particles
            collision_data = ParticlesCollisionsVoxels(particles_obj_idx=particles_obj_idx)
        else: # Hero particles
            collision_data = ParticlesCollisionsPoints(particles_obj_idx=particles_obj_idx)


     def _detect_collisions(self, particles_config_obj: Any, particles_traj: ParticlesTrajectoryData, voxel_objects: List[Any], collision_data: Union[ParticlesCollisionsPoints, ParticlesCollisionsVoxels]):
        frames = particles_traj.sampled_frames if not particles_config_obj.proxy else particles_traj.massive.get_sampled_frames()
        num_particles = particles_traj.positions.shape[0] if not particles_config_obj.proxy else particles_traj.massive.get_particle_count()

        points, voxel_ids = ({} for _ in range(2))
        for sample_idx in frames:
            # Get particle positions at this frame
            particle_positions = particles_traj.get_position(sample_idx)
            for o_idx in voxel_objects.keys():
                points, voxel_ids = voxel_objects[o_idx].query_collision(sample_idx, particle_positions)
                if len(points) > 0 or len(voxel_ids) > 0:
                    particles_vel = particles_traj.get_velocity(sample_idx)
                    particles_sizes = particles_traj.get_sizes(sample_idx)
                    particles_density = particles_config_obj.acoustic_shader.density
                    particles_mass = np.prod(particles_sizes, axis=0)
                    forces = np.linalg.norm(particles_vel, axis=0) * particles_mass

                if len(points) > 0 and not particles_config_obj.proxy:
                    collision_data.add_collision(sample_idx, o_idx, points, particles_vel, forces)
                if len(voxel_ids) > 0 and particles_config_obj.proxy:
                    collision_data.add_collision(sample_idx, o_idx, voxel_ids, particles_vel, forces)

        # Save the results
        if isinstance(collision_data, ParticlesCollisionsPoints):
            filepath = f"{self.collisions_dir}/points_{particles_obj_idx}.pkl"
        else:
            filepath = f"{self.collisions_dir}/voxels_{particles_obj_idx}.pkl"
        
        collision_data.save(filepath)
        self.entity_manager.register('collisions', collision_data)
