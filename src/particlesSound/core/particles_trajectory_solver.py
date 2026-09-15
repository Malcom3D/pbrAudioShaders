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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from pbrAudioCommon import EntityManager, _load_particle
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import PositionSolver, RotationSolver

from ..lib.particles_trajectory_data import ParticlesTrajectoryData

@dataclass
class ParticlesTrajectorySolver:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.output_dir = f"{config.system.cache_path}/trajectories"
        os.makedirs(self.output_dir, exist_ok=True)
        self.position_solver = PositionSolver(self.entity_manager)
        self.rotation_solver = RotationSolver(self.entity_manager)

    def compute(self, particles_obj_idx: int):
        """Computes and registers ParticlesTrajectoryData for a particles object."""
        config = self.entity_manager.get('config')
        
        particles_config_obj = next((p for p in config.particles if p.idx == particles_obj_idx), None)
        if not particles_config_obj:
            raise ValueError(f"Particles config for idx {particles_obj_idx} not found.")

        # Load the entire particle animation sequence
        positions, rotations, sizes, states = _load_particle(particles_config_obj)
        
        sfps = (config.system.fps / config.system.fps_base) * config.system.subframes
        sample_rate = config.system.sample_rate
        n_frames = positions.shape[0]
        n_particles = positions.shape[1]

        original_frames = (1 + np.arange(n_frames)) * sample_rate / sfps

        # Find unsampled intermediate positions for each particle
        # This is a simplified approach. A more optimized version would batch process particles.
        solved_frames_set = set()
        solved_positions = {} # frame -> {particle_idx: pos}
        solved_rotations = {} # frame -> {particle_idx: rot}
        
        # For now, we only solve positions for simplicity, as it's the most critical.
        # A full implementation would also solve rotations.
        for p_idx in range(n_particles):
            particle_positions = positions[:, p_idx, :]
            # This is inefficient, but a starting point
            intersection_data = self.position_solver._intersection(particle_positions, sample_rate, sfps)
            for frame, point in intersection_data:
                solved_frames_set.add(frame)
                if frame not in solved_positions:
                    solved_positions[frame] = {}
                solved_positions[frame][p_idx] = point

        solved_frames = np.array(sorted(list(solved_frames_set)))
        
        # Merge original and solved frames
        all_frames = np.unique(np.concatenate((original_frames, solved_frames)))
        all_frames.sort()

        # Build interpolators for each particle
        pos_interpolators = []
        rot_interpolators = []
        size_interpolators = []
        
        # Create a mapping from frame to index for original data
        original_frame_map = {f: i for i, f in enumerate(original_frames)}

        for p_idx in range(n_particles):
            # --- Build position data for interpolation ---
            interp_positions = []
            for frame in all_frames:
                if frame in original_frame_map:
                    interp_positions.append(positions[original_frame_map[frame], p_idx, :])
                elif frame in solved_positions and p_idx in solved_positions[frame]:
                    interp_positions.append(solved_positions[frame][p_idx])
                else:
                    # Fallback: interpolate from original frames
                    interp_positions.append(np.interp(frame, original_frames, positions[:, p_idx, :]))
            
            interp_positions = np.array(interp_positions)
            if len(all_frames) > 3:
                pos_interp = tuple(CubicSpline(all_frames, interp_positions[:, i], extrapolate=True) for i in range(3))
            else: # Fallback for very short sequences
                pos_interp = tuple(CubicSpline(all_frames, interp_positions[:, i], extrapolate=True) for i in range(3))

            pos_interpolators.append(pos_interp)

            # --- Build rotation data ---
            interp_rotations = [Rotation.from_euler('XYZ', rotations[original_frame_map[f], p_idx, :]) for f in all_frames if f in original_frame_map]
            # This is a simplification; we'd need to interpolate rotations for solved frames too.
            # For now, we'll just use the original rotations.
            valid_frames_for_rot = np.array([f for f in all_frames if f in original_frame_map])
            if len(valid_frames_for_rot) > 1:
                rot_interp = RotationSpline(valid_frames_for_rot, Rotation.concatenate(interp_rotations))
            else:
                rot_interp = None
            rot_interpolators.append(rot_interp)

            # --- Build size data ---
            interp_sizes = np.array([np.interp(frame, original_frames, sizes[:, p_idx, i]) for frame in all_frames for i in range(3)]).reshape(len(all_frames), 3)
            size_interp = tuple(CubicSpline(all_frames, interp_sizes[:, i], extrapolate=True) for i in range(3))
            size_interpolators.append(size_interp)

        # Create the ParticlesTrajectoryData object
        trajectory_data = ParticlesTrajectoryData(
            obj_idx=particles_obj_idx,
            static=particles_config_obj.static,
            sfps=sfps,
            sample_rate=sample_rate,
            positions=pos_interpolators,
            rotations=rot_interpolators,
            sizes=size_interpolators,
            states=states,
            original_frames=original_frames,
            solved_frames=solved_frames
        )
        
        # Register and save
        self.entity_manager.register('trajectories', trajectory_data)
        trajectory_data.save(f"{self.output_dir}/particles_{particles_obj_idx}.pkl")
        debug_print(f"Computed and registered ParticlesTrajectoryData for idx {particles_obj_idx}")

