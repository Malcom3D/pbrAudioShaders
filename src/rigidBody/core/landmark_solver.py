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
from ..lib.functions import _load_pose

@dataclass
class LandmarkSolver:
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        tmp_trajectories = self.entity_manager.get('trajectories')
        for index in tmp_trajectories:
            tmp_trajectory = tmp_trajectories[index]
            if hasattr(tmp_trajectory, 'add_data') and tmp_trajectory.obj_idx == obj_idx:
                frame = tmp_trajectory.frame
                impact_position = tmp_trajectory.position
                impact_rotation = tmp_trajectory.rotation
                for config_obj in config.objects:
                    if config_obj.idx == obj_idx:
                            positions, rotations, landmarks_vertices = _load_pose(config_obj)
                            landmarks_0, landmarks_1, landmarks_2 = np.hsplit(landmarks_vertices, 3)
                            landmark0 = self._impact_landmark(positions, rotations, landmarks_0, impact_position, impact_rotation, frame)
                            landmark1 = self._impact_landmark(positions, rotations, landmarks_1, impact_position, impact_rotation, frame)
                            landmark2 = self._impact_landmark(positions, rotations, landmarks_2, impact_position, impact_rotation, frame)
                            print('landmark0', obj_idx, type(landmark0), landmark0)
                            print('landmark1', obj_idx, type(landmark1), landmark1)
                            print('landmark2', obj_idx, type(landmark2), landmark2)
                            tmp_trajectory.add_data('landmark0', landmark0)
                            tmp_trajectory.add_data('landmark1', landmark1)
                            tmp_trajectory.add_data('landmark2', landmark2)

    def _impact_landmark(self, positions: np.ndarray, rotations: np.ndarray, landmarks: np.ndarray, impact_position: np.ndarray, impact_rotation: np.ndarray, frame: float):
        """
        Alternative method that enforces rigidity constraints more strictly.
        """
        # Pre-impact data
        frame_before = math.floor(frame)
        pre_impact_pos = positions[frame_before]
        pre_impact_rot = Rotation.from_euler('XYZ', rotations[frame_before])

        # Post-impact data (if available)
        frame_after = math.ceil(frame)
        post_impact_pos = positions[frame_after]
        post_impact_rot = Rotation.from_euler('XYZ', rotations[frame_before])

        R_impact = self.euler_to_rotation_matrix(*impact_rotation)
    
        # Calculate average local coordinates from multiple frames for better accuracy
        n_frames_around = min(3, len(landmarks))  # Use up to 3 frames around impact
        start_frame = max(0, frame_before - n_frames_around + 1)
        end_frame = min(len(landmarks), frame_after + n_frames_around)
    
        all_local_coords = []
    
        for i in range(start_frame, end_frame):
            pos_i = positions[i]
            rot_i = rotations[i]
            R_i = self.euler_to_rotation_matrix(*rot_i)
            landmarks_i = landmarks[i]
        
            # Convert to local coordinates
            R_inv = R_i.T
            local_coords = (R_inv @ (landmarks_i - pos_i).T).T
            all_local_coords.append(local_coords)
    
        # Average local coordinates (assuming object is rigid)
        avg_local_coords = np.mean(all_local_coords, axis=0)
    
        # Apply impact transformation
        landmarks_impact = (R_impact @ avg_local_coords.T).T + impact_position
    
        return landmarks_impact

    def euler_to_rotation_matrix(self, rx, ry, rz, order='xyz'):
        """Convert Euler angles to rotation matrix."""
        return Rotation.from_euler(order, [rx, ry, rz]).as_matrix()
