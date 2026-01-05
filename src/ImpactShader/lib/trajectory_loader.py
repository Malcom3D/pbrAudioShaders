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

"""
Utility module for loading and processing trajectories for acceleration noise.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import glob
import os
import re

def load_trajectories_from_obj_files(obj_path: str, obj_name: str, fps: float) -> Dict:
    """
    Load trajectory data from a sequence of OBJ files.
    
    Args:
        obj_path: Directory containing OBJ files
        obj_name: Base name of the object
        fps: Frames per second for timing
        
    Returns:
        Dictionary with trajectory data including accelerations
    """
    # Get all obj files in the directory (excluding optimized mesh)
    obj_files = glob.glob(os.path.join(obj_path, "*.obj"))
    obj_files = [f for f in obj_files if not f.endswith(f"optimized_{obj_name}.obj")]
    
    if not obj_files:
        return {}
    
    # Sort files by frame number
    obj_files.sort(key=_extract_frame_number)
    
    positions = []
    rotations = []
    times = []
    
    for i, obj_file in enumerate(obj_files):
        try:
            mesh = trimesh.load(obj_file, force='mesh')
            
            # Get center of mass position
            position = mesh.center_mass
            positions.append(position)
            
            # Get rotation from mesh transformation
            if hasattr(mesh, 'principal_inertia_transform'):
                transform = mesh.principal_inertia_transform
                rotation = transform[:3, :3]
            else:
                # Estimate rotation from vertices
                # Use PCA to estimate principal axes
                vertices = mesh.vertices - mesh.center_mass
                if len(vertices) > 3:
                    cov = np.cov(vertices.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    rotation = eigvecs
                else:
                    rotation = np.eye(3)
            
            rotations.append(rotation)
            times.append(i / fps)
            
        except Exception as e:
            print(f"Warning: Could not load {obj_file}: {e}")
            # Use previous frame's data if available
            if positions:
                positions.append(positions[-1])
                rotations.append(rotations[-1])
                times.append(i / fps)
    
    if not positions:
        return {}
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    times = np.array(times)
    
    # Calculate velocities and accelerations
    velocities = _calculate_velocities(positions, times)
    accelerations = _calculate_accelerations(velocities, times)
    
    # Calculate angular velocities
    angular_velocities = _calculate_angular_velocities(rotations, times)
    
    return {
        'positions': positions,
        'rotations': rotations,
        'times': times,
        'velocities': velocities,
        'accelerations': accelerations,
        'angular_velocities': angular_velocities,
        'fps': fps,
        'num_frames': len(positions)
    }

def interpolate_trajectories(trajectories: Dict, sample_rate: float) -> Dict:
    """
    Interpolate trajectories to audio sample rate.
    
    Args:
        trajectories: Original trajectory data
        sample_rate: Target sample rate in Hz
        
    Returns:
        Interpolated trajectory data
    """
    times = trajectories['times']
    positions = trajectories['positions']
    velocities = trajectories['velocities']
    accelerations = trajectories['accelerations']
    
    if len(times) < 2:
        return trajectories
    
    # Create new time array at audio sample rate
    total_time = times[-1]
    new_times = np.arange(0, total_time, 1/sample_rate)
    if new_times[-1] < total_time:
        new_times = np.append(new_times, total_time)
    
    # Interpolate positions
    pos_interp = interp1d(times, positions, axis=0, kind='cubic', 
                         fill_value='extrapolate', bounds_error=False)
    new_positions = pos_interp(new_times)
    
    # Interpolate velocities
    vel_interp = interp1d(times, velocities, axis=0, kind='cubic',
                         fill_value='extrapolate', bounds_error=False)
    new_velocities = vel_interp(new_times)
    
    # Interpolate accelerations
    accel_interp = interp1d(times, accelerations, axis=0, kind='cubic',
                           fill_value='extrapolate', bounds_error=False)
    new_accelerations = accel_interp(new_times)
    
    # Interpolate rotations (using quaternions for proper interpolation)
    rot_matrices = trajectories['rotations']
    rots = Rotation.from_matrix(rot_matrices)
    quats = rots.as_quat()
    
    quat_interp = interp1d(times, quats, axis=0, kind='cubic',
                          fill_value='extrapolate', bounds_error=False)
    new_quats = quat_interp(new_times)
    new_rotations = Rotation.from_quat(new_quats).as_matrix()
    
    # Calculate new angular velocities
    new_angular_velocities = _calculate_angular_velocities(new_rotations, new_times)
    
    return {
        'positions': new_positions,
        'rotations': new_rotations,
        'times': new_times,
        'velocities': new_velocities,
        'accelerations': new_accelerations,
        'angular_velocities': new_angular_velocities,
        'sample_rate': sample_rate,
        'num_samples': len(new_times)
    }

def _calculate_velocities(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate velocities from positions using finite differences."""
    if len(positions) < 2:
        return np.zeros_like(positions)
    
    velocities = np.zeros_like(positions)
    for i in range(1, len(positions)):
        dt = times[i] - times[i-1]
        if dt > 0:
            velocities[i] = (positions[i] - positions[i-1]) / dt
    
    # Forward difference for first element
    if len(positions) > 1:
        velocities[0] = velocities[1]
    
    return velocities

def _calculate_accelerations(velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate accelerations from velocities using finite differences."""
    if len(velocities) < 2:
        return np.zeros_like(velocities)
    
    accelerations = np.zeros_like(velocities)
    for i in range(1, len(velocities)):
        dt = times[i] - times[i-1]
        if dt > 0:
            accelerations[i] = (velocities[i] - velocities[i-1]) / dt
    
    # Forward difference for first element
    if len(velocities) > 1:
        accelerations[0] = accelerations[1]
    
    return accelerations

def _calculate_angular_velocities(rotations: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate angular velocities from rotation matrices."""
    n_samples = len(rotations)
    angular_velocities = np.zeros((n_samples, 3))
    
    if n_samples < 2:
        return angular_velocities
    
    for i in range(1, n_samples):
        dt = times[i] - times[i-1]
        if dt > 0:
            # Calculate rotation difference
            R_diff = rotations[i] @ rotations[i-1].T
            
            # Extract angular velocity from skew-symmetric part
            skew_sym = 0.5 * (R_diff - R_diff.T)
            wx = skew_sym[2, 1]
            wy = skew_sym[0, 2]
            wz = skew_sym[1, 0]
            
            angular_velocities[i] = np.array([wx, wy, wz]) / dt
    
    # Forward difference for first element
    angular_velocities[0] = angular_velocities[1]
    
    return angular_velocities

def _extract_frame_number(filename: str) -> int:
    """Extract frame number from filename."""
    basename = os.path.basename(filename)
    numbers = re.findall(r'\d+', basename)
    return int(numbers[-1]) if numbers else 0
