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
from dask import delayed, compute
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import procrustes

from ..core.entity_manager import EntityManager
from ..lib.trajectory_data import TrajectoryData
from ..utils.config import Config, ObjectConfig

from ..lib.functions import _load_pose

@dataclass
class FlightPath:
    """Class to compute flight paths from OBJ sequences."""
    
    def __init__(self, entity_manager: EntityManager):
        self.em = entity_manager
        self.config = self.em.get('config')
        
    @delayed
    def _compute_trajectory_from_pose(self, obj_idx: int, positions: np.ndarray, rotations: np.ndarray, landmarks_vertices: np.ndarray) -> TrajectoryData:
        """Compute trajectory from a sequence of meshes with Procrustes refinement."""
        fps = self.config.system.fps
        fps_base = self.config.system.fps_base
        subframes = self.config.system.subframes
        sample_rate = self.config.system.sample_rate
        
        # samples at sample_rate of full trajectories
        sfps = ( fps / fps_base ) * subframes #subframes per seconds
        num_samples = int(( positions.shape[0] / sfps ) * sample_rate)
        
        # Interpolate positions
        interp_positions = self._interpolate_positions(positions, num_samples, obj_idx)

        # Interpolate rotations
        interp_rotations = self._interpolate_rotations(rotations, num_samples, obj_idx)
        
        # Interpolate landmarks
        interp_landmarks = self._interpolate_landmarks(landmarks_vertices, num_samples, obj_idx)
        
        # Refine trajectory using Procrustes analysis
        refined_positions, refined_rotations = self._refine_with_procrustes(interp_positions, interp_rotations, interp_landmarks, obj_idx)
        
        # Create TrajectoryData
        trajectory_data = TrajectoryData(obj_idx=obj_idx, sample_rate=sample_rate, positions=refined_positions, orientations=refined_rotations, landmarks=interp_landmarks)
        
        return trajectory_data

    def _interpolate_landmarks(self, landmarks_vertices: np.ndarray, num_samples: int, obj_idx: int) -> np.ndarray:
        """
        Interpolate landmark positions using arc length parameterization.
        
        Parameters:
        -----------
        landmarks_vertices : numpy.ndarray
            Input landmarks of shape (k, 3, 3) where:
            - k: number of frames
            - 3: number of landmarks (0, 1, 2)
            - 3: x, y, z coordinates
        num_samples : int
            Number of samples in the output array
            
        Returns:
        --------
        numpy.ndarray
            Interpolated landmarks of shape (num_samples, 3, 3)
        """
        k = landmarks_vertices.shape[0]
        
        # Calculate cumulative arc length using the centroid of landmarks
        centroids = np.mean(landmarks_vertices, axis=1)  # Shape: (k, 3)
        diffs = np.diff(centroids, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        
        # Normalize arc length to [0, 1]
        arc_length_normalized = arc_length / arc_length[-1]
        
        # Check if object is static
        if np.all(np.isnan(arc_length_normalized)) or arc_length[-1] < 1e-6:
            # Object is static, repeat first frame
            interpolated = np.full((num_samples, 3, 3), landmarks_vertices[0])
            return interpolated
        
        # Create new normalized arc length values
        new_arc_length = np.linspace(0, 1, num_samples)
        
        # Initialize array for interpolated landmarks
        interpolated = np.zeros((num_samples, 3, 3))
        
        # Interpolate each landmark point separately
        for landmark_idx in range(3):  # For each of the 3 landmarks
            for dim in range(3):  # For each dimension (x, y, z)
                interp_func = interp1d(arc_length_normalized, landmarks_vertices[:, landmark_idx, dim], kind='cubic', bounds_error=False, fill_value="extrapolate")
                interpolated[:, landmark_idx, dim] = interp_func(new_arc_length)
        
        return interpolated

    def _refine_with_procrustes(self, positions: np.ndarray, rotations: np.ndarray, 
                               landmarks: np.ndarray, obj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine trajectory using Procrustes analysis with landmarks.
        
        Parameters:
        -----------
        positions : numpy.ndarray
            Initial positions of shape (num_samples, 3)
        rotations : numpy.ndarray
            Initial rotations (Euler angles) of shape (num_samples, 3)
        landmarks : numpy.ndarray
            Landmarks of shape (num_samples, 3, 3)
        obj_idx : int
            Object index for debugging
            
        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray]
            Refined positions and rotations
        """
        num_samples = positions.shape[0]
        refined_positions = np.zeros_like(positions)
        refined_rotations = np.zeros_like(rotations)
        
        # Convert rotations to quaternions for easier manipulation
        quaternions = self._euler_to_quaternion(rotations)
        
        # Use first frame as reference
        reference_landmarks = landmarks[0]
        
        for i in range(num_samples):
            if i == 0:
                # Keep first frame as is
                refined_positions[i] = positions[i]
                refined_rotations[i] = rotations[i]
                continue
            
            # Get current frame landmarks
            current_landmarks = landmarks[i]
            
            # Perform Procrustes analysis
            # Note: scipy's procrustes returns (reference_aligned, target_aligned, disparity)
            # We need to extract the transformation
            mtx1, mtx2, disparity = procrustes(reference_landmarks, current_landmarks)
            
            # Calculate the transformation from mtx2 to current_landmarks
            # mtx2 = scale * current_landmarks @ R + translation
            
            # Since procrustes normalizes both sets, we need to reconstruct the transformation
            # We can compute the optimal rotation using SVD
            H = current_landmarks.T @ mtx2
            U, S, Vt = np.linalg.svd(H)
            R_optimal = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R_optimal) < 0:
                Vt[-1, :] *= -1
                R_optimal = Vt.T @ U.T
            
            # Calculate scale
            scale = np.trace(np.diag(S)) / np.trace(current_landmarks.T @ current_landmarks)
            
            # Calculate translation
            translation = mtx2.mean(axis=0) - scale * (current_landmarks @ R_optimal).mean(axis=00)
            
            # Apply transformation to refine position
            # The translation gives us the offset from the reference frame
            refined_positions[i] = positions[0] + translation
            
            # Update rotation: combine existing rotation with optimal rotation
            # Convert R_optimal to quaternion
            R_optimal_quat = Rotation.from_matrix(R_optimal).as_quat()
            
            # Combine rotations (quaternion multiplication)
            # Note: quaternions are in [x, y, z, w] format
            current_quat = quaternions[i]
            combined_quat = self._quaternion_multiply(current_quat, R_optimal_quat)
            
            # Convert back to Euler angles
            refined_rotations[i] = self._quaternion_to_euler(combined_quat[np.newaxis, :])[0]
        
        return refined_positions, refined_rotations

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Parameters:
        -----------
        q1, q2 : numpy.ndarray
            Quaternions in [x, y, z, w] format
            
        Returns:
        --------
        numpy.ndarray
            Resulting quaternion in [x, y, z, w] format
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])

    def _interpolate_positions(self, positions: np.ndarray, num_samples: int, obj_idx: int):
        """
        Interpolate 3D positions using arc length parameterization.
    
        Parameters:
        -----------
        positions : numpy.ndarray
            Input positions of shape (k, 3) in order [x, y, z]
        num_samples : int
            Number of samples in the output array (l)

        Returns:
        --------
        numpy.ndarray
            Interpolated positions of shape (l, 3) in order [x, y, z]
        """
        # Calculate cumulative arc length
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_length = np.concatenate(([0], np.cumsum(segment_lengths)))
    
        # Normalize arc length to [0, 1]
        arc_length_normalized = arc_length / arc_length[-1]
        all_nan = np.all(np.isnan(arc_length_normalized)) 
        if all_nan:
            # object is static
            interpolated = np.full((num_samples, 3), positions[0])
            return interpolated
    
        # Create new normalized arc length values
        new_arc_length = np.linspace(0, 1, num_samples)
    
        # Initialize array for interpolated positions
        interpolated = np.zeros((num_samples, 3))
    
        # Interpolate each dimension
        for dim in range(3):
            interp_func = interp1d(arc_length_normalized, positions[:, dim], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interpolated[:, dim] = interp_func(new_arc_length)
    
        return interpolated

    def _euler_to_quaternion(self, euler_angles, degrees=False):
        """Convert Euler angles (XYZ order) to quaternions."""
        return Rotation.from_euler('XYZ', euler_angles, degrees=degrees).as_quat()

    def _quaternion_to_euler(self, quaternions, degrees=False):
        """Convert quaternions to Euler angles (XYZ order)."""
        return Rotation.from_quat(quaternions).as_euler('XYZ', degrees=degrees)

    def _slerp_quaternions(self, q1, q2, t):
        """Spherical linear interpolation between two quaternions."""
        # Ensure quaternions are unit quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
    
        # Calculate dot product
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
        # If the dot product is negative, the quaternions have opposite handedness
        # and we need to negate one to take the shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
    
        # If the quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
    
        # Calculate the angle between the quaternions
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
    
        # Perform SLERP
        q3 = q2 - q1 * dot
        q3 = q3 / np.linalg.norm(q3)
    
        return q1 * np.cos(theta) + q3 * np.sin(theta)

    def _interpolate_rotations(self, euler_angles: np.ndarray, num_samples: int, degrees=False) -> np.ndarray:
        """
        Interpolate Euler rotations using SLERP.
    
        Parameters:
        -----------
        euler_angles : ndarray, shape (k, 3)
            Input Euler angles (XYZ order)
        num_samples : int
            Desired output length
        degrees : bool
            Whether the input angles are in degrees (default: False)
    
        Returns:
        --------
        ndarray, shape (num_samples, 3)
            Interpolated Euler angles
        """
        k = len(euler_angles)
    
        # Convert Euler angles to quaternions
        quaternions = self._euler_to_quaternion(euler_angles, degrees=degrees)
    
        # Generate interpolation positions
        t_positions = np.linspace(0, k-1, num_samples)
    
        # Prepare output array
        interpolated_quaternions = np.zeros((num_samples, 4))
    
        for i, t in enumerate(t_positions):
            # Find the segment containing this interpolation point
            idx = int(np.floor(t))
            if idx >= k - 1:
                # If at or beyond the last point, use the last quaternion
                interpolated_quaternions[i] = quaternions[-1]
            else:
                # Calculate interpolation parameter within the segment
                segment_t = t - idx
                # Perform SLERP between consecutive quaternions
                interpolated_quaternions[i] = self._slerp_quaternions(
                    quaternions[idx], 
                    quaternions[idx + 1], 
                    segment_t
                )
    
        # Convert back to Euler angles
        interpolated_euler = self._quaternion_to_euler(interpolated_quaternions, degrees=degrees)
    
        return interpolated_euler


    def _interpolate_rotations_quaternion(self, quaternions: np.ndarray, num_samples: int, obj_idx: int):
        """
        Interpolate quaternions using SLERP.
    
        Parameters:
        -----------
        quaternions : numpy.ndarray
            Input quaternions of shape (k, 4) in order [x, y, z, w]
        num_samples : int
            Number of samples in the output array (l)
        
        Returns:
        --------
        numpy.ndarray
            Interpolated quaternions of shape (l, 4) in order [x, y, z, w]
        """
        # Ensure quaternions are normalized
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
        # Create times array for interpolation
        times = np.linspace(0, 1, len(quaternions))
    
        # Create Slerp object (scipy expects [w, x, y, z] order)
        # Convert from [x, y, z, w] to [w, x, y, z]
        quaternions_wxyz = np.roll(quaternions, shift=1, axis=1)
        slerp = Slerp(times, Rotation.from_quat(quaternions_wxyz))
    
        # Create new times for interpolation
        new_times = np.linspace(0, 1, num_samples)
    
        # Interpolate
        interpolated_rotations = slerp(new_times)
    
        # Convert back to [x, y, z, w] order
        interpolated_quats = interpolated_rotations.as_quat()  # Returns [x, y, z, w]
    
        return interpolated_quats
    
    def compute(self, obj: int, detected_distances) -> None:
        """Compute trajectories for all objects in parallel using Dask."""
        config = self.config
    
        # Create delayed tasks for each object
        tasks = []
        
        # Compute all trajectories in parallel
        tasks = [self._dask_tasks(config_obj) for config_obj in config.objects]
        results = compute(*tasks)

        # Register each trajectory in EntityManager
        for trajectory_idx, trajectory_data in enumerate(results):
            self.em.register('trajectories', trajectory_data, trajectory_data.obj_idx)
        
    def _dask_tasks(self, config_obj) -> TrajectoryData:
        # Load positions, rotations, and landmarks sequence
        positions, rotations, landmarks_vertices = _load_pose(config_obj)
            
        # Compute trajectory with Procrustes refinement
        trajectory_data = self._compute_trajectory_from_pose(config_obj.idx, positions, rotations, landmarks_vertices)

        return trajectory_data
