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
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class TrajectoryData:
    """Container for trajectory and orientation data."""
    obj_idx: int
    sample_rate: int
    positions: np.ndarray   # Shape: (num_samples, 3) - world coordinates in meters
    orientations: np.ndarray  # Shape: (num_samples, 3) - eurler rotations (x, y, z)
    landmarks: Optional[np.ndarray] = None  # Shape: (num_samples, 3, 3) - landmarks for each sample
    
    def get_num_samples(self) -> int:
        return self.positions.shape[0]

    def get_position(self, sample_idx: int) -> np.ndarray:
        """Get interpolated position at specific sample_idx."""
        if sample_idx < 0:
            return self.positions[0]
        elif sample_idx >= len(self.positions) - 1:
            return self.positions[-1]
        
        return self.positions[sample_idx]

    def get_orientation(self, sample_idx: int) -> np.ndarray:
        """Get interpolated orientation at specific sample_idx."""
        if sample_idx < 0:
            return self.orientations[0]
        elif sample_idx >= len(self.orientations) - 1:
            return self.orientations[-1]
        
        return self.orientations[sample_idx]

    def _get_transformation(self, sample_idx: int) -> np.ndarray:
        """Get transformation matrix for a specific sample_idx."""
        # Get position and orientation for this sample_idx
        position = self.positions[sample_idx]
        quaternion = self.orientations[sample_idx]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self._euler_to_rotation_matrix(quaternion)
        
        # Create homogeneous transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = position
        
        return transformation

    @staticmethod
    def _euler_to_rotation_matrix(q: np.ndarray, degrees=False):
        """
        Convert Euler angles to rotation matrix (ZYX convention).
    
        Parameters:
        -----------
        q : np.ndarray
            yaw, pitch, roll
            yaw : float
                Rotation around Z axis (yaw)
            pitch : float
                Rotation around Y axis (pitch)
            roll : float
                Rotation around X axis (roll)
        degrees : bool, optional
            If True, input angles are in degrees. Default is False (radians).
    
        Returns:
        --------
        R : np.ndarray
            3x3 rotation matrix
        """

        roll, pitch, yaw = q

        if degrees:
            # Convert degrees to radians
            yaw = np.radians(yaw)
            pitch = np.radians(pitch)
            roll = np.radians(roll)
    
        # Precompute trigonometric functions
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)
    
        # ZYX rotation matrix (R = Rz(yaw) * Ry(pitch) * Rx(roll))
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
    
        return R

    def get_relative_transformation(self, from_sample: int, to_sample: int) -> np.ndarray:
        """
        Get relative transformation from one sample_idx to another.
        
        Parameters:
        -----------
        from_sample : int or float
            Source sample_idx index
        to_sample : int or float
            Target sample_idx index
        
        Returns:
        --------
        np.ndarray
            Relative transformation matrix T_to_from such that:
            P_to = T_to_from @ P_from
        """
        # Get transformations in world coordinates
        T_world_from = self._get_transformation(from_sample)
        T_world_to = self._get_transformation(to_sample)
        
        # Compute relative transformation: T_to_from = T_world_to @ inv(T_world_from)
        T_world_from_inv = np.linalg.inv(T_world_from)
        T_to_from = T_world_to @ T_world_from_inv
        
        return T_to_from
