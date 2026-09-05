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
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _load_particle
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.particle_trajectory_data import ParticleTrajectoryData

@dataclass
class ParticlesSolver:
    """
    Solver for particle system trajectories and orientations.
    
    Computes unsampled intermediate positions and rotations for particles
    using algorithms from PositionSolver and RotationSolver.
    
    The solver:
    1. Loads particle data from npz files (positions, rotations, sizes, states)
    2. Detects unsampled intermediate positions using the PositionSolver algorithm
    3. Estimates rotations at those positions using the RotationSolver algorithm
    4. Creates ParticleTrajectoryData with interpolated positions and rotations
    """
    entity_manager: EntityManager
    
    # Detection parameters
    position_tolerance: float = 1e-6  # Position matching tolerance
    velocity_threshold: float = 0.01   # Minimum velocity change for detection
    sampling_interval: float = 0.001   # Sampling interval for search
    
       def __post_init__(self):
        config = self.entity_manager.get('config')
        
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        self.sample_rate = config.system.sample_rate
        self.fps = config.system.fps
        self.fps_base = config.system.fps_base
        self.subframes = config.system.subframes
        self.sfps = (self.fps / self.fps_base) * self.subframes
        
        self.output_dir = f"{config.system.cache_path}/particle_trajectories"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache for particle data
        self._particle_data_cache = {}
    
    def compute(self, particle_idx: int, particle_name: str = None) -> Optional[ParticleTrajectoryData]:
        """
        Compute particle trajectories from particle sequence
        
        Parameters:
        -----------
        particle_idx : int
            ID of particles object
        particle_name : str, optional
            Object name for output naming
            
        Returns:
        --------
        Optional[ParticleTrajectoryData]
            Computed particle trajectory data, or None if failed
        """
        config = self.entity_manager.get('config')
        for particle_cfg in config.particles:
            if particle_cfg.idx == particle_idx:
            # Load particle data from files
            positions, rotations, states = self._load_particle(particle_cfg)
            break
        
        # Extract frame indices
        frame_indices = np.arange(positions.shape[0])
        frame_times = np.array(frame_indices) * self.sample_rate / self.sfps
        
        # Initialize particle trajectory data
        particle_count = positions[0].shape[0]
        particle_data = ParticleTrajectoryData(frames=frame_times, particle_idx=particle_idx, is_static=particle_cfg.static, sfps=self.sfps, sample_rate=self.sample_rate, particle_count=particle_count)

        debug_print(f"Processing {particle_count} particles across {positions.shape[0]} frames")
        
        if particle_cfg.static:
            particle_data.positions = positions
            particle_data.rotations = rotations
            particle_data.states = states

        else:
            # Process each particle
            particles_positions, particles_rotations, particles_states = ([] for _ in range(3))
            for particle_idx in range(particle_count):
                for frame_idx in frame_indices:
                    # Extract particle data across all frames
                    particle_positions, particle_rotations, particle_states = ([] for _ in range(3))
            
                    particle_positions.append(positions[frame_idx][particle_idx])
                    particle_rotations.append(rotations[frame_idx][particle_idx])
                    particle_states.append(states[frame_idx][particle_idx])

                particles_positions.append(particle_positions)
                particles_rotations.append(particle_rotations)
                particles_states.append(particle_states)
            
            # Convert to numpy arrays
            positions_array = np.array(particles_positions)
            rotations_array = np.array(particles_rotations)
            states_array = np.array(particles_states)
            
            # Detect unsampled intermediate positions using PositionSolver algorithm
            for particle_idx in range(particle_count):
                unsampled_positions = self._detect_unsampled_positions(positions=positions_array[particle_idx], times=frame_times)
            
                # If unsampled positions found, insert them into the data
                if len(unsampled_positions) > 0:
                    debug_print(f"Particle {particle_idx}: Found {len(unsampled_positions)} unsampled positions")
                
                    # Create combined data with unsampled positions
                    all_times = np.sort(np.concatenate([frame_times, [p['time'] for p in unsampled_positions]]))
                
                    # Interpolate positions at all times
                    all_positions = self._interpolate_positions(times=frame_times, positions=positions_array[particle_idx], eval_times=all_times)
                
                    # Estimate rotations at unsampled positions using RotationSolver algorithm
                    all_rotations = self._estimate_rotations(times=valid_times, rotations=rotations_array[particle_idx], positions=positions_array[particle_idx], eval_times=all_times, unsampled_positions=unsampled_positions)

                    # Create interpolation functions
                    for coord_idx in range(3):
                        particle_data.positions[particle_idx, coord_idx] = CubicSpline(all_times, all_positions[:, coord_idx], extrapolate=1)
                        particle_data.rotations[particle_idx, coord_idx] = CubicSpline(all_times, all_rotations[:, coord_idx], extrapolate=1)
                else:
                    # No unsampled positions - use original data
                    for coord_idx in range(3):
                        particle_data.positions[particle_idx, coord_idx] = CubicSpline(frame_times, positions_array[particle_idx][:, coord_idx], extrapolate=1)
                        particle_data.rotations[particle_idx, coord_idx] = CubicSpline(frame_times, rotations_array[particle_idx][:, coord_idx], extrapolate=1)
            
            particle_data.states = states_array
        
        # Register with entity manager
        _ = self.entity_manager.register('trajectories', particle_data)
        
        # Save to file
        if obj_name:
            output_file = f"{self.output_dir}/{obj_name}_particles.pkl"
            particle_data.save(output_file)
        
    def _detect_unsampled_positions(self, positions: np.ndarray, times: np.ndarray) -> List[Dict]:
        """
        Detect unsampled intermediate positions using the PositionSolver algorithm.
        
        This implements the _intersection method from PositionSolver:
        - Finds where consecutive position segments intersect
        - These intersection points represent unsampled positions
        
        Parameters:
        -----------
        positions : np.ndarray
            Array of shape (n_frames, 3)
        times : np.ndarray
            Array of frame times
            
        Returns:
        --------
        List of dictionaries with 'time' and 'position' keys
        """
        unsampled = []
        
        if len(positions) < 4:
            return unsampled
        
        for index in range(2, positions.shape[0] - 2):
            # Find intersection point using the algorithm from PositionSolver
            intersection_point = self._intersection_point(positions, index)
            
            if intersection_point is not None:
                intersection_time = self._intersection_time(positions=positions, times=times, frame=index, intersection_point=intersection_point)
                    
                # Check if this time is sufficiently far from existing samples
                time_diffs = np.abs(times - intersection_time)
                if np.min(time_diffs) > 1.0 / self.sample_rate:
                    unsampled.append({'time': intersection_time, 'position': intersection_point})
        
        return unsampled
    
    def _intersection_point(self, positions: np.ndarray, frame: int, tolerance: float = 1e-10) -> Optional[np.ndarray]:
        """
        Find the vertex P of triangle P,P2,P3 where:
        - P lies on the line through P2 in direction (P1 - P2)
        - P lies on the line through P3 in direction (P4 - P3)
        
        This is the same algorithm as in PositionSolver._intersection_point
        
        Parameters:
        -----------
        positions : np.ndarray
            Array of positions
        frame : int
            Frame index
        tolerance : float
            Tolerance for checking if lines are parallel
            
        Returns:
        --------
        Optional[np.ndarray]
            Intersection point, or None if lines are parallel
        """
        if frame < 2 or frame >= positions.shape[0] - 2:
            return None
        
        # Get surrounding points
        P1 = positions[frame - 2]
        P2 = positions[frame - 1]
        P3 = positions[frame + 1]
        P4 = positions[frame + 2]
        
        # Direction vectors
        d1 = P1 - P2  # Direction from P2 toward P1
        d2 = P4 - P3  # Direction from P3 toward P4
        
        # Check if direction vectors are parallel
        cross_product = np.cross(d1, d2)
        if np.linalg.norm(cross_product) < tolerance:
            return None
        
        # Solve for intersection: P2 + t*d1 = P3 + s*d2
        A = np.column_stack((d1, -d2))
        b = P3 - P2
        
        # Solve using least squares
        ts, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        t = ts[0]
        s = ts[1]
        
        # Calculate P using either line equation
        P_line1 = P2 + t * d1
        P_line2 = P3 + s * d2
        
        # Check consistency
        if np.linalg.norm(P_line1 - P_line2) > tolerance * 100:
            # Lines don't intersect perfectly - use midpoint
            return (P_line1 + P_line2) / 2
        
        return P_line1
    
    def _intersection_time(self, positions: np.ndarray, times: np.ndarray, frame: int, intersection_point: np.ndarray) -> float:
        """
        Find intersection time from computed intersection point.
        
        This implements the _intersection_time method from PositionSolver.
        
        Parameters:
        -----------
        positions : np.ndarray
            Array of positions
        times : np.ndarray
            Array of frame times @sample_rate
        frame : int
            Frame index
        intersection_point : np.ndarray
            The computed intersection point
            
        Returns:
        --------
        float
            Time of intersection
        """
        # Get surrounding points for interpolation
        P2 = positions[frame - 1]
        P3 = positions[frame]
        
        # Project intersection point onto P2-P3 segment
        v = P3 - P2
        w = intersection_point - P2
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        
        if c2 < 1e-10:
            alpha = 0.5  # Midpoint if segment is too short
        else:
            alpha = np.clip(c1 / c2, 0.0, 1.0)
        
        # Calculate exact time
        time_at_P2 = times[frame - 1]
        time_at_P3 = times[frame]
        intersection_time = time_at_P2 + alpha * (time_at_P3 - time_at_P2)
        
        return intersection_time
    
    def _interpolate_positions(self, times: np.ndarray, positions: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
        """
        Interpolate positions at evaluation times.
        
        Parameters:
        -----------
        times : np.ndarray
            Original time points
        positions : np.ndarray
            Original positions (n_times, 3)
        eval_times : np.ndarray
            Times to evaluate at
            
        Returns:
        --------
        np.ndarray
            Interpolated positions (n_eval_times, 3)
        """
        n_eval = len(eval_times)
        result = np.zeros((n_eval, 3))
        
        for coord_idx in range(3):
            spline = CubicSpline(times, positions[:, coord_idx], extrapolate=1)
            result[:, coord_idx] = spline(eval_times)
        
        return result
    
    def _estimate_rotations(self, times: np.ndarray, rotations: np.ndarray, positions: np.ndarray, eval_times: np.ndarray, unsampled_positions: List[Dict]) -> np.ndarray:
        """
        Estimate rotations at evaluation times using the RotationSolver algorithm.
        
        For unsampled positions, we estimate the rotation by:
        1. Interpolating between known rotations
        2. Accounting for angular velocity changes at intersection points
        
        Parameters:
        -----------
        times : np.ndarray
            Original time points
        rotations : np.ndarray
            Original rotations as Euler angles (n_times, 3)
        positions : np.ndarray
            Original positions (n_times, 3)
        eval_times : np.ndarray
            Times to evaluate at
        unsampled_positions : List[Dict]
            List of unsampled positions with 'time' and 'position'
            
        Returns:
        --------
        np.ndarray
            Estimated rotations (n_eval_times, 3)
        """
        n_eval = len(eval_times)
        result = np.zeros((n_eval, 3))
        
        # Convert rotations to Rotation objects for interpolation
        rotations_objects = Rotation.from_euler('XYZ', rotations)
        
        # Create Slerp interpolator for smooth rotation interpolation
        if len(times) >= 2:
            slerp = Slerp(times, rotations_objects)
        else:
            # Not enough data - return first rotation repeated
            for i in range(n_eval):
                result[i] = rotations[0]
            return result
        
        # Interpolate rotations at all evaluation times
        for i, eval_time in enumerate(eval_times):
            # Check if this is an unsampled position
            is_unsampled = False
            for unsampled_pos in unsampled_positions:
                if abs(unsampled_pos['time'] - eval_time) < 1e-6:
                    is_unsampled = True
                    break
            
            if is_unsampled:
                # Estimate rotation at unsampled position using the RotationSolver algorithm approach
                # Find surrounding frame indices
                before_idx = np.searchsorted(times, eval_time) - 1
                after_idx = min(before_idx + 1, len(times) - 1)
                before_idx = max(before_idx, 0)
                
                if before_idx == after_idx:
                    # At the boundary
                    result[i] = rotations[before_idx]
                else:
                    # Time between frames
                    dt = times[after_idx] - times[before_idx]
                    if dt > 0:
                        # Fraction of the way between frames
                        frac = (eval_time - times[before_idx]) / dt
                        
                        # Get rotations
                        rot_before = Rotation.from_euler('XYZ', rotations[before_idx])
                        rot_after = Rotation.from_euler('XYZ', rotations[after_idx])
                        
                        # Estimate angular velocity
                        # delta_rot = rot_after * rot_before.inv()
                        # ang_vel = delta_rot.as_rotvec() / dt
                        delta_rot = rot_after * rot_before.inv()
                        ang_vel = delta_rot.as_rotvec() / dt
                        
                        # Integrate from before to eval_time
                        time_to_eval = eval_time - times[before_idx]
                        delta_rot_vec = ang_vel * time_to_eval
                        delta_rot = Rotation.from_rotvec(delta_rot_vec)
                        
                        # Estimated rotation
                        estimated_rot = rot_before * delta_rot
                        result[i] = estimated_rot.as_euler('XYZ')
                    else:
                        result[i] = rotations[before_idx]
            else:
                # Regular interpolation using Slerp
                try:
                    rot = slerp(eval_time)
                    result[i] = rot.as_euler('XYZ')
                except:
                    # Fallback to linear interpolation
                    before_idx = np.searchsorted(times, eval_time) - 1
                    after_idx = min(before_idx + 1, len(times) - 1)
                    before_idx = max(before_idx, 0)
                    
                    if before_idx == after_idx:
                        result[i] = rotations[before_idx]
                    else:
                        dt = times[after_idx] - times[before_idx]
                        if dt > 0:
                            frac = (eval_time - times[before_idx]) / dt
                            result[i] = rotations[before_idx] * (1 - frac) + rotations[after_idx] * frac
                        else:
                            result[i] = rotations[before_idx]
        
        return result
