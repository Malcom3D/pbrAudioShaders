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
import pickle
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

@dataclass
class ParticleTrajectoryData:
    """
    Container for particle trajectory and orientation data.
    
    Stores interpolated positions and rotations for multiple particles.
    Each particle has its own set of interpolation functions.
    """
    particle_idx: int
    particle_count: int
    sfps: float
    sample_rate: int
    # Array of interpolation functions for each particle's position
    # # Shape: (particle_count, 3) where each element is a CubicSpline
    positions: np.ndarray = None  # dtype=object, shape (particle_count, 3)
    # Array of interpolation functions for each particle's rotation
    # Shape: (particle_count, 3) where each element is a CubicSpline
    rotations: np.ndarray = None  # dtype=object, shape (particle_count, 3)
    # Static data for particles that don't move
    static_positions: np.ndarray = None  # Shape: (particle_count, 3)
    static_rotations: np.ndarray = None  # Shape: (particle_count, 3)
    # Particle states: 0=dead, 1=alive, 2=unborn
    states: np.ndarray = None  # Shape: (particle_count,) dtype=int8
    # Track which particles are static
    is_static: np.ndarray = None  # Shape: (particle_count,) dtype=bool
    # Original frame data for reference
    frames: np.ndarray = None  # Array of frame indices
    
    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.positions is None:
            self.positions = np.zeros((self.particle_count, 3), dtype=object)
        if self.rotations is None:
            self.rotations = np.zeros((self.particle_count, 3), dtype=object)
        if self.static_positions is None:
            self.static_positions = np.zeros((self.particle_count, 3), dtype=np.float32)
        if self.static_rotations is None:
            self.static_rotations = np.zeros((self.particle_count, 3), dtype=np.float32)
        if self.states is None:
            self.states = np.zeros(self.particle_count, dtype=np.int8)
        if self.is_static is None:
            self.is_static = np.zeros(self.particle_count, dtype=bool)
    
    def get_position(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """
        Get interpolated position for a specific particle at a sample index.
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Position (3,) at the given sample index
        """
        if self.is_static[particle_idx]:
            return self.static_positions[particle_idx].copy()
        else:
            x = self.positions[particle_idx, 0](sample_idx)
            y = self.positions[particle_idx, 1](sample_idx)
            z = self.positions[particle_idx, 2](sample_idx)
            return np.array([x, y, z])
    
    def get_rotation(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """
        Get interpolated rotation (Euler angles XYZ) for a specific particle.
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Euler angles (3,) in radians
        """
        if self.is_static[particle_idx]:
            return self.static_rotations[particle_idx].copy()
        else:
            x = self.rotations[particle_idx, 0](sample_idx)
            y = self.rotations[particle_idx, 1](sample_idx)
            z = self.rotations[particle_idx, 2](sample_idx)
            return np.array([x, y, z])
    
    def get_velocity(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """
        Get interpolated velocity for a specific particle.
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Velocity (3,) in m/s
        """
        if self.is_static[particle_idx]:
            return np.zeros(3)
        else:
            x = self.positions[particle_idx, 0](sample_idx, 1) * self.sample_rate
            y = self.positions[particle_idx, 1](sample_idx, 1) * self.sample_rate
            z = self.positions[particle_idx, 2](sample_idx, 1) * self.sample_rate
            return np.array([x, y, z])
    
    def get_acceleration(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """
        Get interpolated acceleration for a specific particle.
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Acceleration (3,) in m/s²
        """
        if self.is_static[particle_idx]:
            return np.zeros(3)
        else:
            x = self.positions[particle_idx, 0](sample_idx, 2) * self.sample_rate**2
            y = self.positions[particle_idx, 1](sample_idx, 2) * self.sample_rate**2
            z z = self.positions[particle_idx, 2](sample_idx, 2) * self.sample_rate**2
            return np.array([x, y, z])
    
    def get_angular_velocity(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """
        Get interpolated angular velocity for a specific particle.
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Angular velocity (3,) in rad/s
        """
        if self.is_static[particle_idx]:
            return np.zeros(3)
        else:
            x = self.rotations[particle_idx, 0](sample_idx, 1) * self.sample_rate
            y = self.rotations[particle_idx, 1](sample_idx, 1) * self.sample_rate
            z = self.rotations[particle_idx, 2](sample_idx, 1) * self.sample_rate
            return np.array([x, y, z])
    
    def get_state(self, particle_idx: int) -> int:
        """
        Get the state of a particle.
        
        Returns:
        --------
        int
            0=dead, 1=alive, 2=unborn
        """
        return self.states[particle_idx]
    
    def get_all_positions(self, sample_idx: float) -> np.ndarray:
        """
        Get positions for all particles at a sample index.
        
        Parameters:
        -----------
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Positions array of shape (particle_count, 3)
        """
        positions = np.zeros((self.particle_count, 3), dtype=np.float32)
        for i in range(self.particle_count):
            positions[i] = self.get_position(i, sample_idx)
        return positions
    
    def get_all_rotations(self, sample_idx: float) -> np.ndarray:
        """
        Get rotations for all particles at a sample index.
        
        Parameters:
        -----------
        sample_idx : float
            Sample index to evaluate
            
        Returns:
        --------
        np.ndarray
            Rotations array of shape (particle_count, 3)
        """
        rotations = np.zeros((self.particle_count, 3), dtype=np.float32)
        for i in range(self.particle_count):
            rotations[i] = self.get_rotation(i, sample_idx)
        return rotations
    
    def get_all_states(self) -> np.ndarray:
        """
        Get states for all particles.
        
        Returns:
        --------
        np.ndarray
            States array of shape (particle_count,)
        """
        return self.states.copy()
    
    def save(self, filepath: str) -> None:
        """
        Save data in pickle format (preserves interpolation objects).
        
        Parameters:
        -----------
        filepath : str
            Path to save the file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Create a serializable version of the object
        save_dict = {
            'particle_count': self.particle_count,
            'sfps': self.sfps,
            'sample_rate': self.sample_rate,
            'positions': self.positions,
            'rotations': self.rotations,
            'static_positions': self.static_positions,
            'static_rotations': self.static_rotations,
            'states': self.states,
            'is_static': self.is_static,
            'frames': self.frames,
            '_format': 'ParticleTrajectoryData_v1_pickle'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Particle trajectory data saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ParticleTrajectoryData':
        """
        Load data from pickle format.
        
        Parameters:
        -----------
        filepath : str
            Path to the pickle file
            
        Returns:
        --------
        ParticleTrajectoryData
            Loaded particle trajectory data
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check format
        if '_format' not in data or data['_format'] != 'ParticleTrajectoryData_v1_pickle':
            raise ValueError("Invalid file format or version")
        
        # Reconstruct the object
        return ParticleTrajectoryData(
            particle_count=data['particle_count'],
            sfps=data['sfps'],
            sample_rate=data['sample_rate'],
            positions=data['positions'],
            rotations=data['rotations'],
            static_pos_positions=data['static_positions'],
            static_rotations=data['static_rotations'],
            states=data['states'],
            is_static=data['is_static'],
            frames=data['frames']
        )
