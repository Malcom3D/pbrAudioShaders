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
            particle_frames = self._load_particle(particle_cfg)
        
        if len(particle_frames) == 0:
            debug_print(f"No particle data found in {particle_data_path}")
            return None
        
        # Build master particle list
        master_particles = self._build_master_particle_list(particle_frames)
        
        if len(master_particles) == 0:
            debug_print("No particles found")
            return None
        
        debug_print(f"Processing {len(master_particles)} particles across {len(particle_frames)} frames")
        
        # Extract frame indices
        frame_indices = sorted(particle_frames.keys())
        frame_times = np.array(frame_indices) * self.sample_rate / self.sfps
        
        # Initialize particle trajectory data
        particle_count = len(master_particles)
        particle_data = ParticleTrajectoryData(particle_idx=particle_idx, particle_count=particle_count, sfps=self.sfps, sample_rate=self.sample_rate, frames=frame_indices)
        
        # Process each particle
        for p_idx, particle_id in enumerate(master_particles):
            # Extract particle data across all frames
            particle_positions = []
            particle_rotations = []
            particle_states = []
            valid_frames = []
            
            for frame_idx in frame_indices:
                if frame_idx in particle_frames:
                    frame_data = particle_frames[frame_idx]
                    if particle_id in frame_data:
                        data = frame_data[particle_id]
                        particle_positions.append(data['position'])
                        particle_rotations.append(data['rotation'])
                        particle_states.append(data['state'])
                        valid_frames.append(frame_idx)
            
            if len(valid_frames) < 2:
                # Not enough data for interpolation
                particle_data.is_static[p_idx] = True
                if len(valid_frames) == 1:
                    particle_data.static_positions[p_idx] = particle_positions[0]
                    particle_data.static_rotations[p_idx] = particle_rotations[0]
                continue
            
            # Convert to numpy arrays
            positions_array = np.array(particle_positions)
            rotations_array = np.array(particle_rotations)
            states_array = np.array(particle_states)
            valid_times = np.array(valid_frames) * self.sample_rate / self.sfps
            
            # Check if particle is static
            if self._is_particle_static(positions_array, rotations_array):
                particle_data.is_static[p_idx] = True
                particle_data.static_positions[p_idx] = positions_array[0]
                particle_data.static_rotations[p_idx] = rotations_array[0]
                particle_data.states[p_idx] = states_array[0]
                continue
            
            # Detect unsampled intermediate positions using PositionSolver algorithm
            unsampled_positions = self._detect_unsampled_positions(positions=positions_array, times=valid_times)
            
            # If unsampled positions found, insert them into the data
            if len(unsampled_positions) > 0:
                debug_print(f"Particle {p_idx}: Found {len(unsampled_positions)} unsampled positions")
                
                # Create combined data with unsampled positions
                all_times = np.sort(np.concatenate([valid_times, [p['time'] for p in unsampled_positions]]))
                
                # Interpolate positions at all times
                all_positions = self._interpolate_positions(times=valid_times, positions=positions_array, eval_times=all_times)
                
                # Estimate rotations at unsampled positions using RotationSolver algorithm
                all_rotations = self._estimate_rotations(times=valid_times, rotations=rotations_array, positions=positions_array, eval_times=all_times, unsampled_positions=unsampled_positions)
                
                # Create interpolation functions
                for coord_idx in range(3):
                    particle_data.positions[p_idx, coord_idx] = CubicSpline(all_times, all_positions[:, coord_idx], extrapolate=1)
                    particle_data.rotations[p_idx, coord_idx] = CubicSpline(all_times, all_rotations[:, coord_idx], extrapolate=1)
            else:
                # No unsampled positions - use original data
                for coord_idx in range(3):
                    particle_data.positions[p_idx, coord_idx] = CubicSpline(valid_times, positions_array[:, coord_idx], extrapolate=1)
                    particle_data.rotations[p_idx, coord_idx] = CubicSpline(valid_times, rotations_array[:, coord_idx], extrapolate=1)
            
            # Set particle state (use most common state)
            particle_data.states[p_idx] = self._get_most_common_state(states_array)
        
        # Register with entity manager
        _ = self.entity_manager.register('trajectories', particle_data)
        
        # Save to file
        if obj_name:
            output_file = f"{self.output_dir}/{obj_name}_particles.pkl"
            particle_data.save(output_file)
        
    def _build_master_particle_list(self, particle_frames: Dict[int, Dict[str, Dict]]) -> List[str]:
        """
        Build master list of all particle identifiers.
        
        Returns:
        --------
        List of particle identifiers
        """
        master_list = []
        seen = set()
        
        for frame_idx in sorted(particle_frames.keys()):
            frame_data = particle_frames[frame_idx]
            for particle_id in frame_data.keys():
                if particle_id not in seen:
                    seen.add(particle_id)
                    master_list.append(particle_id)
        
        return master_list
    
    def _is_particle_static(self, positions: np.ndarray, rotations: np.ndarray) -> bool:
        """
        Check if a particle is static across all frames.
        
        Parameters:
        -----------
        positions : np.ndarray
            Array of shape (n_frames, 3)
        rotations : np.ndarray
            Array of shape (n_frames, 3)
            
        Returns:
        --------
        bool
            True if particle is static
        """
        if len(positions) < 2:
            return True
        
        position_static = np.all(np.abs(positions - positions[0]) < self.position_tolerance)
        rotation_static = np.all(np.abs(rotations - rotations[0]) < self.position_tolerance)
        
        return position_static and rotation_static
    
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
        
        position_old = np.zeros(3)
        
        for index in range(2, len(positions) - 2):
            # Find intersection point using the algorithm from PositionSolver
            intersection_point = self._intersection_point(positions, index)
            
            if intersection_point is not None:
                error = np.sum((position_old - intersection_point)**2)
                
                if error < 1e-06:
                    # Found an intersection - this is an unsampled position
                    intersection_time = self._intersection_time(positions=positions, times=times, frame=index, intersection_point=intersection_point)
                    
                    # Check if this time is sufficiently far from existing samples
                    time_diffs = np.abs(times - intersection_time)
                    if np.min(time_diffs) > 1.0 / self.sfps:
                        unsampled.append({'time': intersection_time, 'position': intersection_point})
                
                position_old = intersection_point
        
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
        if if frame < 2 or frame >= len(positions) - 2:
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
            Array of frame times
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
    
    def _get_most_common_state(self, states: np.ndarray) -> int:
        """
        Get the most common state from an array of states.
        
        Parameters:
        -----------
        states : np.ndarray
            Array of particle states
            
        Returns:
        --------
        int
            Most common state
        """
        if len(states) == 0:
            return 0
        
        # Count occurrences of each state
        unique, counts = np.unique(states, return_counts=True)
        
        # Return the most common state
        return int(unique[np.argmax(counts)])

