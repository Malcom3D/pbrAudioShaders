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

import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_pose, _load_mesh

@dataclass
class NormalSolver:
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        tmp_trajectories = self.entity_manager.get('trajectories')
        for index in tmp_trajectories:
            if 'tmpTrajectoryData' in str(type(tmp_trajectories[index])) and tmp_trajectories[index].obj_idx == obj_idx:
                tmp_trajectory = tmp_trajectories[index]
                frame = tmp_trajectory.frame
                impact_position = tmp_trajectory.position
                impact_rotation = tmp_trajectory.rotation
                # Find the object config
                config_obj = None
                for obj in config.objects:
                    if obj.idx == obj_idx:
                        config_obj = obj
                        break

                if config_obj:
                    # Load pose data
                    positions, rotations = _load_pose(config_obj)

                    # Get normals at impact time
                    normals_impact = self._impact_normals(config_obj, positions, rotations, impact_position, impact_rotation, frame)
                    tmp_trajectory.add_data('normals', normals_impact)

    def _impact_normals(self, config_obj, positions: np.ndarray, rotations: np.ndarray, impact_position: np.ndarray, impact_rotation: np.ndarray, frame: float, n_frames: int = 3) -> np.ndarray:
        """
        Compute vertex normals at impact time by loading and interpolating normals from multiple surrounding frames.
        
        Parameters:
        -----------
        config_obj : ObjectConfig
            Configuration object containing mesh information
        positions : np.ndarray
            Array of positions for each frame
        rotations : np.ndarray
            Array of rotations (Euler angles) for each frame
        impact_position : np.ndarray
            Position at impact time
        impact_rotation : np.ndarray
            Rotation (Euler angles) at impact time
        frame : float
            Impact frame (can be fractional)
        n_frames : int
            Number of frames to use on each side of impact for averaging
            
        Returns:
        --------
        np.ndarray
            Vertex normals at impact time in world coordinates
        """
        # Get frame range for averaging
        frame_idx = int(round(frame))
        start_frame = max(0, frame_idx - n_frames)
        end_frame = min(len(positions), frame_idx + n_frames + 1)
        
        all_local_normals = []
        
        # Collect local normals from multiple frames
        for i in range(start_frame, end_frame):
            vertices_i, normals_i, faces_i = _load_mesh(config_obj, i)
            pos_i = positions[i]
            rot_i = Rotation.from_euler('XYZ', rotations[i])
            R_i = rot_i.as_matrix()
            
            # Convert normals to local coordinates
            # Note: Normals transform differently than vertices - they use the inverse transpose
            # For rotation matrices (which are orthogonal), inverse transpose = the matrix itself
            local_normals_i = (R_i.T @ normals_i.T).T
            all_local_normals.append(local_normals_i)
        
        # Average local normals
        avg_local_normals = np.mean(all_local_normals, axis=0)
        
        # Normalize the averaged normals to ensure unit length
        norm_magnitudes = np.linalg.norm(avg_local_normals, axis=1, keepdims=True)
        norm_magnitudes[norm_magnitudes == 0] = 1  # Avoid division by zero
        avg_local_normals = avg_local_normals / norm_magnitudes
        
        # Apply impact transformation to normals
        R_impact = Rotation.from_euler('XYZ', impact_rotation).as_matrix()
        normals_impact = (R_impact @ avg_local_normals.T).T
        
        # Normalize again after transformation
        norm_magnitudes = np.linalg.norm(normals_impact, axis=1, keepdims=True)
        norm_magnitudes[norm_magnitudes == 0] = 1
        normals_impact = normals_impact / norm_magnitudes
        
        return normals_impact
