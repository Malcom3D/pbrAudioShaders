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
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_pose
from ..lib.trajectory_data import tmpTrajectoryData

@dataclass
class PositionSolver:
    """ Find unsampled intermediate positions """
    entity_manager: EntityManager

    def compute(self, obj_idx: int):
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                positions, rotations = _load_pose(config_obj)

        position_intersection = self._intersection(sequence_array=positions, sfps=sfps)

        for index in range(len(position_intersection)):
            frame, point = position_intersection[index]
            tmp_trajectory_data = tmpTrajectoryData(obj_idx=obj_idx, sfps=sfps, frame=frame, position=point)
            trajectory_idx = len(self.entity_manager.get('trajectories')) + 1
            self.entity_manager.register('trajectories', tmp_trajectory_data, trajectory_idx)

    def _intersection(self, sequence_array: np.ndarray, sfps: int):
        intersection_data = []
        intersection_points = []
        position_old = np.array([0,0,0])
        for index in range(2, len(sequence_array) - 2):
            position = self._intersection_point(sequence_array, index)
            error = np.sum((position_old - position)**2)
            if error < 1e-06:
                intersection_point = position + (position_old - position)/2 
                intersection_points.append([index, intersection_point])
            position_old = position

        for frame, intersection_point in intersection_points:
            intersection_time = self._intersection_time(motion_path=sequence_array, frame=frame, intersection_point=intersection_point, sfps=sfps)
            intersection_data.append([intersection_time['absolute_frame'], intersection_point])
        return intersection_data

    def _intersection_time(self, motion_path: np.ndarray, frame: int, intersection_point: np.ndarray, sfps: float) -> Dict:
        """
        Find intersection time from computed intersection_point.
        
        Parameters:
        -----------
        motion_path : np.ndarray
            Array of shape (n_frames, 3) containing position samples
        frame : int
            Index of the frame where intersection was detected
        intersection_point : np.ndarray
            The computed intersection point P
        sfps : float
            Subframes per second (sampling rate)
        
        Returns:
        --------
        result : Dict
            Dictionary containing:
            - 'time': intersection time in seconds
            - 'method': used method
            - 'confidence': confidence score (0-1)
            - 'frame_offset': fractional frame offset
        """
        # Get surrounding points for interpolation
        # Use linear interpolation between P2 and P3
        P2 = motion_path[frame - 1]
        P3 = motion_path[frame]
            
        # Project intersection point onto P2-P3 segment
        v = P3 - P2
        w = intersection_point - P2
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
            
        if c2 < 1e-10:
            alpha = 0.5  # Midpoint if segment is too short
        else:
            alpha = np.clip(c1 / c2, 0.0, 1.0)
            
        # Calculate confidence based on how close intersection is to the segment
        dist_to_segment = np.linalg.norm(intersection_point - (P2 + alpha * v))
        segment_length = np.linalg.norm(v)
        confidence = max(0, 1 - dist_to_segment / max(segment_length, 1e-10))
            
        # Clamp alpha and confidence
        alpha = np.clip(alpha, 0.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Calculate exact time
        time_at_P2 = (frame - 1) / sfps
        time_at_P3 = frame / sfps
        intersection_time = time_at_P2 + alpha * (time_at_P3 - time_at_P2)
        
        return {
            'time': intersection_time,
            'confidence': confidence,
            'frame_offset': alpha,
            'absolute_frame': (frame) + alpha
        }
                            
    def _intersection_point(self, motion_path: np.ndarray, frame: int, tolerance=1e-10):
        """
        Find the vertex P of triangle P,P2,P3 where:
        - P lies on the line through P2 in direction (P1 - P2)
        - P lies on the line through P3 in direction (P4 - P3)
    
        Parameters:
        -----------
        P1, P2, P3, P4 : tuples or lists of length 3
            Coordinates of points P1, P2, P3, P4
        tolerance : float
            Tolerance for checking if lines are parallel
    
        Returns:
        --------
        P : numpy array of shape (3,)
            Coordinates of point P
        """
        # Convert to numpy arrays
        P1 = motion_path[frame - 2]
        P2 = motion_path[frame - 1]
        P3 = motion_path[frame + 1]
        P4 = motion_path[frame + 2]
    
        # Direction vectors
        d1 = P1 - P2  # Direction from P2 toward P1
        d2 = P4 - P3  # Direction from P3 toward P4
    
        # Check if direction vectors are parallel
        cross_product = np.cross(d1, d2)
        if np.linalg.norm(cross_product) < tolerance:
            raise ValueError("Lines are parallel or coincident, no unique intersection point")
    
        # Parametric equations:
        # Line 1: P = P2 + t * d1, where t is a scalar parameter
        # Line 2: P = P3 + s * d2, where s is a scalar scalar parameter
    
        # We need to solve: P2 + t*d1 = P3 + s*d2
        # Rearranged: t*d1 - s*d2 = P3 - P2
    
        # Create a system of linear equations
        # For 3D, we have 3 equations but 2 unknowns, so we need to solve in least squares sense
        # or use two of the equations that are independent
    
        # Let's solve using linear algebra approach
        A = np.column_stack((d1, -d2))  # Columns are d1 and -d2
        b = P3 - P2
    
        # Solve for t and s using least squares
        ts, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
        t = ts[0]
        s = ts[1]
    
        # Calculate P using either line equation
        P_line1 = P2 + t * d1
        P_line2 = P3 + s * d2
    
        # Check consistency (should be very close if lines intersect)
        if np.linalg.norm(P_line1 - P_line2) > tolerance:
            #print(f"Warning: Lines don't intersect perfectly. Distance: {np.linalg.norm(P_line1 - P_line2)}")
            # Return the midpoint as best estimate
            P = (P_line1 + P_line2) / 2
        else:
            P = P_line1
    
        return P
