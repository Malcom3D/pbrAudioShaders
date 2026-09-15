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
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from pbrAudioCommon import EntityManager, _parse_lib
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.particles_collisions_points import ParticlesCollisionsPoints
from ..lib.particles_collisions_voxels import ParticlesCollisionsVoxels

@dataclass
class ParticlesComposer:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.dsp_path = f"{config.system.cache_path}/dsp"

    def compute(self, particles_obj_idx: int):
        """
        Analyze collision data to determine the maximum number of concurrent modal
        convolvers needed for each material.
        """
        config = self.entity_manager.get('config')
        particles_config_obj = next((p for p in config.particles if p.idx == particles_obj_idx), None)
        
        collisions_dir = f"{config.system.cache_path}/particles_collisions"
        if particles_config_obj.proxy:
            filepath = f"{collisions_dir}/voxels_{particles_obj_idx}.pkl"
        else:
            filepath = f"{collisions_dir}/points_{particles_obj_idx}.pkl"
            
        if not os.path.exists(filepath):
            debug_print(f"No collision data found for {particles_obj_idx}")
            return

        if particles_config_obj.proxy:
            collision_data = ParticlesCollisionsVoxels.load(filepath)
        else:
            collision_data = ParticlesCollisionsPoints.load(filepath)

        # Map to store max concurrent convolvers per object
        max_convolvers_needed = {}
        
        # Get all unique obj_idx involved in collisions
        involved_obj_indices = set()
        if isinstance(collision_data, ParticlesCollisionsPoints):
            for frame_collisions in collision_data.collisions.values():
                for collision in frame_collisions.values():
                    involved_obj_indices.add(collision['obj_idx'])
        else: # Voxels
            for frame_collisions in collision_data.collisions.values():
                for collision in frame_collisions.values():
                    involved_obj_indices.add(collision['obj_idx'])

        # For each object, determine the max T60 to know how long a convolver is active
        for obj_idx in involved_obj_indices:
            obj_config = next((o for o in config.objects if o.idx == obj_idx), None)
            if not obj_config: continue
            
            lib_path = f"{self.dsp_path}/{obj_config.name}.lib"
            if not os.path.exists(lib_path): continue
            
            modal_data = _parse_lib(lib_path)
            max_t60 = np.max(modal_data['t60s']) if modal_data['t60s'].size > 0 else 0
            max_convolvers_needed[obj_idx] = {'max_t60': max_t60, 'count': 0}

        # Sliding window to find max concurrent active convolvers
        active_convolvers = {obj_idx: [] for obj_idx in involved_obj_indices} # Store end times
        max_concurrent = {obj_idx: 0 for obj_idx in involved_obj_indices}
        
        all_frames = sorted(collision_data.collisions.keys())

        for frame in all_frames:
            # Remove expired convolvers
            for obj_idx in involved_obj_indices:
                active_convolvers[obj_idx] = [end_time for end_time in active_convolvers[obj_idx] if end_time > frame]

            # Add new convolvers from this frame's collisions
            frame_collisions = collision_data.collisions[frame]
            collided_objs_this_frame = set()
            if isinstance(collision_data, ParticlesCollisionsPoints):
                for collision in frame_collisions.values():
                    collided_objs_this_frame.add(collision['obj_idx'])
            else: # Voxels
                for collision in frame_collisions.values():
                    collided_objs_this_frame.add(collision['obj_idx'])
            
            for obj_idx in collided_objs_this_frame:
                # Each unique voxel/point collision collision starts a new convolver
                num_new_convolvers = len([c for c in frame_collisions.values() if c['obj_idx'] == obj_idx])
                t60_samples = max_convolvers_needed[obj_idx]['max_t60'] * config.system.sample_rate
                for _ in range(num_new_convolvers):
                    active_convolvers[obj_idx].append(frame + t60_samples)

            # Update max
            for obj_idx in involved_obj_indices:
                max_concurrent[obj_idx] = max(max_concurrent[obj_idx], len(active_convolvers[obj_idx]))

        # Save the results
        composer_dir = f"{config.system.cache_path}/particles_composer"
        os.makedirs(composer_dir, exist_ok=True)
        filepath = f"{composer_dir}/convolvers_{particles_obj_idx}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(max_concurrent, f)
        
        debug_print(f"Composer result for {particles_obj_idx}: {max_concurrent}")

