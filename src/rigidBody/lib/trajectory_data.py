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
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from ..lib.functions import _euler_to_rotation_matrix

@dataclass
class tmpTrajectoryData:
    """Temporary container for solved trajectory data."""
    obj_idx: int
    sfps: float
    frame: float
    position: np.ndarray
    rotation: np.ndarray = None
    vertices: np.ndarray = None
    normals: np.ndarray = None

    def add_data(self, component: str, data: np.ndarray):
        """Add a data component if not exist"""
        if getattr(self, component) is None:
            setattr(self, component, data)

@dataclass
class TrajectoryData:
    """Container for trajectory and orientation data."""
    obj_idx: int
    static: bool
    sfps: float
    sample_rate: int
    positions: Union[np.ndarray, Tuple[CubicSpline, CubicSpline, CubicSpline]] # static or interpolated world coordinates in meters
    rotations: Union[np.ndarray, RotationSpline] # static or interpolated eurler rotations in rad (x,y,z)
    vertices: np.ndarray # static or interpolated vertices for each sample [[vertex0_x,vertex0_y,vertex0_z],[vertex1_x,vertex1_y,vertex1_z],[vertex2_x,vertex2_y,vertex2_z]...]
    normals: np.ndarray # static or interpolated normals for each vertex [[normal0_x,normal0_y,normal0_z],[normal1_x,normal1_y,normal1_z],[normal2_x,normal2_y,normal2_z]...]
    faces: np.ndarray # rigid body mesh faces

    def get_x(self) -> np.ndarray:
        """Get independent variables of interp1d interpolation"""
        if not self.static:
            return self.positions[0].x
        return np.array([])

    def get_rx(self) -> np.ndarray:
        """Get independent variables of slerp interpolation"""
        if not self.static:
            return self.rotations.times
        return np.array([])

    def get_position(self, sample_idx: float) -> np.ndarray:
        """Get interpolated position at specific sample_idx."""
        if self.static:
            # Static object: positions is a single (3,) array
            return self.positions.copy()
        else:
            # Moving object: positions is a tuple of CubicSpline functions
            x = self.positions[0](sample_idx)
            y = self.positions[1](sample_idx)
            z = self.positions[2](sample_idx)
            return np.array([x, y, z])

    def get_velocity(self, sample_idx: float) -> np.ndarray:
        if self.static:
            # Static object: positions is a single (3,) array
            return np.array([0,0,0])
        else:
            # Moving object: positions is a tuple of CubicSpline functions
            x = self.positions[0](sample_idx, 1) * self.sample_rate
            y = self.positions[1](sample_idx, 1) * self.sample_rate
            z = self.positions[2](sample_idx, 1) * self.sample_rate
            return np.array([x, y, z])

    def get_acceleration(self, sample_idx: float) -> np.ndarray:
        if self.static:
            # Static object: positions is a single (3,) array
            return np.array([0,0,0])
        else:
            # Moving object: positions is a tuple of CubicSpline functions
            x = self.positions[0](sample_idx, 2) * self.sample_rate**2
            y = self.positions[1](sample_idx, 2) * self.sample_rate**2
            z = self.positions[2](sample_idx, 2) * self.sample_rate**2
            return np.array([x, y, z])

    def get_rotation(self, sample_idx: float) -> np.ndarray:
        """Get interpolated rotation at specific sample_idx."""
        if self.static:
            # Static object: rotations is a single (3,) array of Euler angles
            return self.rotations.copy()
        else:
            # Moving object: rotations is a tuple of CubicSpline functions
           return self.rotations(sample_idx).as_euler('XYZ')

    def get_angular_velocity(self, sample_idx: float) -> np.ndarray:
        """Get interpolated angular velocity at specific sample_idx."""
        if self.static:
            # Static object: rotations is a single (3,) array of Euler angles
            return np.array([0,0,0])
        else:
            # Moving object: rotations is a tuple of RotationSpline functions
            return self.rotations(sample_idx, 1) * self.sample_rate

    def get_angular_acceleration(self, sample_idx: float) -> np.ndarray:
        """Get interpolated angular acceleration at specific sample_idx."""
        if self.static:
            # Static object: rotations is a single (3,) array of Euler angles
            return np.array([0,0,0])
        else:
            # Moving object: rotations is a tuple of CubicSpline functions
            return self.rotations(sample_idx, 1) * self.sample_rate**2

    def get_vertices(self, sample_idx: float) -> np.ndarray:
        """Get interpolated position of vertices at specific sample_idx."""
        if self.static:
            return self.vertices.copy()
        else:
            vertices = np.zeros((self.vertices.shape))
            for vertex_idx in range(len(self.vertices)):
                for coord_idx in range(3):
                    # Each vertices coordinate has its own interpolation function
                    vertices[vertex_idx, coord_idx] = self.vertices[vertex_idx, coord_idx](sample_idx)
            return vertices

    def get_normals(self, sample_idx: float) -> np.ndarray:
        """Get interpolated vertex normals at specific sample_idx."""
        if self.static:
            return self.normals.copy()
        else:
            normals = np.zeros((self.normals.shape))
            for normal_idx in range(len(self.normals)):
                for coord_idx in range(3):
                    # Each vertex normals has its own interpolation function
                    normals[normal_idx, coord_idx] = self.normals[normal_idx, coord_idx](sample_idx)
            return normals

    def get_faces(self, sample_idx: float = None) -> np.ndarray:
        """Get faces"""
        return self.faces

    def save(self, filepath: str) -> None:
        """Save data in pickle format (preserves interpolation objects)."""
        # Create a serializable version of the object
        save_dict = {
            'obj_idx': self.obj_idx,
            'static': self.static,
            'sfps': self.sfps,
            'sample_rate': self.sample_rate,
            'positions': self.positions,
            'rotations': self.rotations,
            'vertices': self.vertices,
            'normals': self.normals,
            'faces': self.faces,
            '_format': 'TrajectoryData_v1_pickle'
        }
    
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Trajectory data saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TrajectoryData':
        """Load data from pickle format."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
        # Check format
        if '_format' not in data or data['_format'] != 'TrajectoryData_v1_pickle':
            raise ValueError("Invalid file format or version")
    
        # Reconstruct the object
        return TrajectoryData(
            obj_idx=data['obj_idx'],
            static=data['static'],
            sfps=data['sfps'],
            sample_rate=data['sample_rate'],
            positions=data['positions'],
            rotations=data['rotations'],
            vertices=data['vertices'],
            normals=data['normals'],
            faces=data['faces']
        )
