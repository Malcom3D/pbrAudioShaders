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
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.trajectory_data import TrajectoryData, tmpTrajectoryData
from ..lib.functions import _load_mesh, _load_pose

@dataclass
class FlightPath:
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> None:
        """Compute flight path for an object and create TrajectoryData."""
        config = self.entity_manager.get('config')
        
        # Find the object config
        config_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                config_obj = obj
                break
        
        if not config_obj:
            raise ValueError(f"Object with idx {obj_idx} not found in config")
        
        # Get system parameters
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = (fps / fps_base) * subframes  # subframes per second
        sample_rate = config.system.sample_rate
        
        # Load pose data
        positions, rotations = _load_pose(config_obj)
        
        # Get all trajectory data for this object
        tmp_trajectories = self.entity_manager.get('trajectories')
        trajectory_frames = []
        
        for idx, traj in tmp_trajectories.items():
            if isinstance(traj, tmpTrajectoryData) and traj.obj_idx == obj_idx:
                trajectory_frames.append([sample_rate * traj.frame / sfps, traj.position, traj.rotation, traj.vertices, traj.normals])
        
        if config_obj.static:
            # Static object - single position/rotation
            trajectory_data = self._static_trajectory(config_obj, positions, rotations, sample_rate)
        else:
            # Dynamic object - interpolated trajectory
            if trajectory_frames:
                # Use solved trajectory frames
                trajectory_data = self._dynamic_trajectory_with_solved(config_obj, positions, rotations, trajectory_frames, sfps, sample_rate)
            else:
                # Use original frames (no collisions detected)
                trajectory_data = self._dynamic_trajectory_original(config_obj, positions, rotations, sfps, sample_rate)
        
        # Register the trajectory data
        trajectory_idx = len(self.entity_manager.get('trajectories')) + 1
        self.entity_manager.register('trajectories', trajectory_data, trajectory_idx)
        
        # Remove temporary trajectory data for this object
        self._cleanup_tmp_trajectories(obj_idx)

    def _static_trajectory(self, config_obj, positions: np.ndarray, rotations: np.ndarray, sample_rate: int) -> TrajectoryData:
        """Create trajectory data for static object."""
        # Load mesh data for static object
        vertices, normals, faces = _load_mesh(config_obj, 0)
        
        return TrajectoryData(obj_idx=config_obj.idx, static=True, sample_rate=sample_rate, positions=positions, rotations=rotations, vertices=vertices, normals=normals, faces=faces)

    def _dynamic_trajectory_original(self, config_obj, positions: np.ndarray, rotations: np.ndarray, sfps: float, sample_rate: int) -> TrajectoryData:
        """Create trajectory data for dynamic object using original frames."""
        n_frames = len(positions)
        
        # Create time points for original frames
        frame_times = sample_rate * ( 1 + np.arange(n_frames)) / sfps
        
        # Interpolate positions and rotations using CubicSpline
        pos_interp = [CubicSpline(frame_times, positions[:, i], extrapolate=1) for i in range(positions.shape[1])]
        rot_interp = [CubicSpline(frame_times, rotations[:, i], extrapolate=1) for i in range(positions.shape[1])]
        
        # Load mesh data for all frames and create interpolators
        all_vertices = []
        all_normals = []

        for frame_time in frame_times:
            # Find position at this time
            idx = np.where(frame_times == frame_time)[0][0]
            vert, norm, _ = _load_mesh(config_obj, idx)

            all_vertices.append(vert)
            all_normals.append(norm)

        all_vertices = np.array(all_vertices)  # Shape: (n_frames, n_vertices, 3)
        all_normals = np.array(all_normals)    # Shape: (n_frames, n_vertices, 3)
        vertices_interp, normals_interp = self._create_mesh_interpolators(config_obj, all_vertices, all_normals, frame_times)
        
        # Get faces (assume constant topology)
        _, _, faces = _load_mesh(config_obj, 1)
        
        return TrajectoryData(obj_idx=config_obj.idx, static=False, sample_rate=sample_rate, positions=pos_interp, rotations=rot_interp, vertices=vertices_interp, normals=normals_interp, faces=faces)

    def _dynamic_trajectory_with_solved(self, config_obj, positions: np.ndarray, rotations: np.ndarray, solved_frames: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], sfps: float, sample_rate: int) -> TrajectoryData:
        """Create trajectory data for dynamic object with solved collision frames."""
        n_original_frames = len(positions)
        original_frames = 1 + np.arange(n_original_frames)
        original_frame_times = sample_rate * original_frames / sfps

        frames = []
        for idx in range(len(solved_frames)):
            frames.append(solved_frames[idx][0])

        # Combine original and solved frames
        all_frames = sorted(set(list(original_frame_times) + frames))
        all_frame_times = np.array(all_frames)
        
        # Get positions and rotations at all frame times
        all_positions = []
        all_rotations = []
        all_vertices = []
        all_normals = []
        
        for frame_time in all_frame_times:
            # Find position at this time
            if frame_time in original_frame_times:
                idx = np.where(original_frame_times == frame_time)[0][0]
                pos = positions[idx]
                rot = rotations[idx]
                vert, norm, faces = _load_mesh(config_obj, idx)
            else:
                # solved position
                idx = next((i for i in range(len(solved_frames)) if frame_time == solved_frames[i][0]), None)
                pos = solved_frames[idx][1]
                rot = solved_frames[idx][2]
                vert = solved_frames[idx][3]
                norm = solved_frames[idx][4]
            
            all_positions.append(pos)
            all_rotations.append(rot)
            all_vertices.append(vert)
            all_normals.append(norm)
        
        all_positions = np.array(all_positions)
        all_rotations = np.array(all_rotations)
        all_vertices = np.array(all_vertices)  # Shape: (n_frames, n_vertices, 3)
        all_normals = np.array(all_normals)    # Shape: (n_frames, n_vertices, 3)
        
        # Interpolate positions and rotations using CubicSpline
        pos_interp = [CubicSpline(all_frame_times, all_positions[:, i], extrapolate=1) for i in range(all_positions.shape[1])]
        rot_interp = [CubicSpline(all_frame_times, all_rotations[:, i], extrapolate=1) for i in range(all_positions.shape[1])]
        
        # Create mesh interpolators using the combined frame times
        vertices_interp, normals_interp = self._create_mesh_interpolators(config_obj, all_vertices, all_normals, all_frame_times)
        
        # Get faces
        _, _, faces = _load_mesh(config_obj, 1)
        
        return TrajectoryData(obj_idx=config_obj.idx, static=False, sample_rate=sample_rate, positions=pos_interp, rotations=rot_interp, vertices=vertices_interp, normals=normals_interp, faces=faces)

    def _create_mesh_interpolators(self, config_obj, all_vertices: np.ndarray, all_normals: np.ndarray, frame_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create interpolators for vertices and normals across frames.
        
        Returns:
            Tuple of (vertices_interp, normals_interp) where each is a 2D array
            of interpolation functions for each vertex coordinate.
        """
        n_vertices = all_vertices.shape[1]
        
        # Create interpolation functions for each vertex coordinate
        vertices_interp = np.zeros((n_vertices, 3), dtype=object)
        normals_interp = np.zeros((n_vertices, 3), dtype=object)
        
        for v_idx in range(n_vertices):
            for coord_idx in range(3):
                # Interpolate vertex coordinates
                vertex_coords = all_vertices[:, v_idx, coord_idx]
                vertices_interp[v_idx, coord_idx] = CubicSpline(frame_times, vertex_coords, extrapolate=1)
                
                # Interpolate normal coordinates
                normal_coords = all_normals[:, v_idx, coord_idx]
                normals_interp[v_idx, coord_idx] = CubicSpline(frame_times, normal_coords, extrapolate=1)
        
        return vertices_interp, normals_interp

    def _cleanup_tmp_trajectories(self, obj_idx: int) -> None:
        """Remove temporary trajectory data for the given object."""
        tmp_trajectories = self.entity_manager.get('trajectories')
        indices_to_remove = []
        
        for idx, traj in tmp_trajectories.items():
            if isinstance(traj, tmpTrajectoryData) and traj.obj_idx == obj_idx:
                indices_to_remove.append(idx)
        
        for idx in indices_to_remove:
            self.entity_manager.unregister('trajectories', idx)
