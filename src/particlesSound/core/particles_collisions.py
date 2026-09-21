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
from typing import Any, List, Dict, Tuple
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
        particles_traj = self.entity_manager.get('trajectories').get(particles_obj_idx)
        if not isinstance(particles_traj, ParticlesTrajectoryData):
            debug_print(f"No ParticlesTrajectoryData found for idx {particles_obj_idx}")
            return

        # Get all other objects' trajectories and voxel data
        scene_objects = {}
        trajectories = self.entity_manager.get('trajectories')
        voxel_objects = self.entity_manager.get('surface_voxel_objects')
        
        for obj_idx, traj in trajectories.items():
            if isinstance(traj, TrajectoryData) and obj_idx != particles_obj_idx:
                scene_objects[obj_idx] = {
                    'trajectory': traj,
                    'voxel_object': voxel_objects.get(obj_idx)
                }

        if not scene_objects:
            debug_print("No scene objects to collide with.")
            return

        if particles_config_obj.proxy: # Massive particles
            collision_data = ParticlesCollisionsVoxels(particles_obj_idx=particles_obj_idx)
        else: # Hero particles
            collision_data = ParticlesCollisionsPoints(particles_obj_idx=particles_obj_idx)

        frames = particles_traj.get_x()
        num_particles = len(particles_traj.positions)

        for frame in frames:
            # Get particle positions at this frame
            particle_positions = np.array([particles_traj.get_position(i, frame) for i in range(num_particles)])
            
            for obj_idx, obj_data in scene_objects.items():
                voxel_obj = obj_data['voxel_object']
                if not voxel_obj:
                    continue

                # Query collisions
                for p_idx, p_pos in enumerate(particle_positions):
                    if particles_traj.get_state(p_idx, frame) != 1: # Not alive
                        continue
                    
                    voxel_idx, distance = voxel_obj.query_collision(frame, p_pos)
                    
                    if voxel_idx.size > 0:
                        # Collision detected
                        particle_vel = (particles_traj.get_position(p_idx, frame + 1) - p_pos) * particles_traj.sample_rate
                        # Approximate force (Hertzian-like)
                        force = np.linalg.norm(particle_vel) * 0.001 # Placeholder for mass
                        
                        if isinstance(collision_data, ParticlesCollisionsPoints):
                            collision_data.add_collision(frame, p_idx, obj_idx, p_pos, particle_vel, force)
                        else:
                            collision_data.add_collision(frame, obj_idx, voxel_idx, force)

        # Save the results
        if isinstance(collision_data, ParticlesCollisionsPoints):
            filepath = f"{self.collisions_dir}/points_{particles_obj_idx}.pkl"
        else:
            filepath = f"{self.collisions_dir}/voxels_{particles_obj_idx}.pkl"
        
        collision_data.save(filepath)
        self.entity_manager.register('particles_collisions', collision_data)

