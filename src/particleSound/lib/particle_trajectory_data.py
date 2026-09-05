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
    frames: np.ndarray  # interpolated frame number
    particle_idx: int
    is_static: bool = False
    sfps: float 
    sample_rate: int
    particle_count: int
    positions: np.ndarray = None  # dtype=object, shape (particle_count, 3)  where each element is a CubicSpline
    rotations: np.ndarray = None  # dtype=object, shape (particle_count, 3)  where each element is a CubicSpline
    states: np.ndarray = None  # Shape: (particle_count,) dtype=int8  0=dead, 1=alive, 2=unborn

    def get_states(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
        if particle_idx is None:
            if self.is_static:
                return self.states.copy()
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                if self.frames.shape[0] > 1 and sample_idx < self.frames[-1]:
                    idx = np.where(self.frames == np.min(self.frames[0 < self.frames - sample_idx]))
                elif self.frames.shape[0] > 1:
                    idx = np.where(self.frames == self.frames[-1])
                particle_cloud.append(self.states[particle_idx][idx])
            return np.array(particle_cloud)

        elif self.is_static and sample_idx == self.frames[0]:
            return self.states[particle_idx]
        elif self.frames.shape[0] > 1 and sample_idx < self.frames[-1]:
            idx = np.where(self.frames == np.min(self.frames[0 < self.frames - sample_idx]))
        elif self.frames.shape[0] > 1:
            idx = np.where(self.frames == self.frames[-1])
        return self.states[particle_idx][idx]

    def get_position(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
        """
        Get interpolated position for a specific particle at a sample index.
        
        Parameters:
        -----------
        sample_idx : float
            Sample index to evaluate
        particle_idx : int
            Index of the particle
            
        Returns:
        --------
        np.ndarray
            Position (3,) at the given sample index
        """
        if particle_idx is None:
            if self.is_static:
                return self.positions.copy()
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                x = self.positions[particle_idx, 0](sample_idx)
                y = self.positions[particle_idx, 1](sample_idx)
                z = self.positions[particle_idx, 2](sample_idx)
                particle_cloud.append([x,y,z])
            return np.array(particle_cloud)

        elif self.is_static:
            return self.positions[particle_idx].copy()
        else:
            x = self.positions[particle_idx, 0](sample_idx)
            y = self.positions[particle_idx, 1](sample_idx)
            z = self.positions[particle_idx, 2](sample_idx)
            return np.array([x, y, z])
    
    def get_rotation(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
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
        if particle_idx is None:
            if self.is_static:
                return self.rotations.copy()
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                x = self.rotations[particle_idx, 0](sample_idx)
                y = self.rotations[particle_idx, 1](sample_idx)
                z = self.rotations[particle_idx, 2](sample_idx)
                particle_cloud.append([x,y,z])
            return np.array(particle_cloud)

        elif self.is_static:
            return self.rotations[particle_idx].copy()
        else:
            x = self.rotations[particle_idx, 0](sample_idx)
            y = self.rotations[particle_idx, 1](sample_idx)
            z = self.rotations[particle_idx, 2](sample_idx)
            return np.array([x, y, z])
    
    def get_velocity(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
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
        if particle_idx is None:
            if self.is_static:
                return np.zeros((self.particle_count, 3))
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                x = self.positions[particle_idx, 0](sample_idx, 1) * self.sample_rate
                y = self.positions[particle_idx, 1](sample_idx, 1) * self.sample_rate
                z = self.positions[particle_idx, 2](sample_idx, 1) * self.sample_rate
                particle_cloud.append([x,y,z])
            return np.array(particle_cloud)

        if self.is_static:
            return np.zeros(3)
        else:
            x = self.positions[particle_idx, 0](sample_idx, 1) * self.sample_rate
            y = self.positions[particle_idx, 1](sample_idx, 1) * self.sample_rate
            z = self.positions[particle_idx, 2](sample_idx, 1) * self.sample_rate
            return np.array([x, y, z])
    
    def get_acceleration(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
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
        if particle_idx is None:
            if self.is_static:
                return np.zeros((self.particle_count, 3))
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                x = self.positions[particle_idx, 0](sample_idx, 2) * self.sample_rate**2
                y = self.positions[particle_idx, 1](sample_idx, 2) * self.sample_rate**2
                z = self.positions[particle_idx, 2](sample_idx, 2) * self.sample_rate**2
                particle_cloud.append([x,y,z])
            return np.array(particle_cloud)

        if self.is_static:
            return np.zeros(3)
        else:
            x = self.positions[particle_idx, 0](sample_idx, 2) * self.sample_rate**2
            y = self.positions[particle_idx, 1](sample_idx, 2) * self.sample_rate**2
            z = self.positions[particle_idx, 2](sample_idx, 2) * self.sample_rate**2
            return np.array([x, y, z])
    
    def get_angular_velocity(self, sample_idx: float, particle_idx: int = None) -> np.ndarray:
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
        if particle_idx is None:
            if self.is_static:
                return np.zeros((self.particle_count, 3))
            particle_cloud = []
            for particle_idx in range(self.particle_count)
                x = self.rotations[particle_idx, 0](sample_idx, 1) * self.sample_rate
                y = self.rotations[particle_idx, 1](sample_idx, 1) * self.sample_rate
                z = self.rotations[particle_idx, 2](sample_idx, 1) * self.sample_rate
                particle_cloud.append([x,y,z])
            return np.array(particle_cloud)

        if self.is_static:
            return np.zeros(3)
        else:
            x = self.rotations[particle_idx, 0](sample_idx, 1) * self.sample_rate
            y = self.rotations[particle_idx, 1](sample_idx, 1) * self.sample_rate
            z = self.rotations[particle_idx, 2](sample_idx, 1) * self.sample_rate
            return np.array([x, y, z])
    
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
            'positions': self.positions,
            'rotations': self.rotations,
            'states': self.states,
            'is_static': self.is_static,
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
            positions=data['positions'],
            rotations=data['rotations'],
            states=data['states'],
            is_static=data['is_static'],
        )
