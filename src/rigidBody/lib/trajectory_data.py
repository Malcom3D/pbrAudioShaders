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
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from ..lib.functions import _euler_to_rotation_matrix

@dataclass
class tmpTrajectoryData:
    """Temporary container for solved trajectory data."""
    obj_idx: int
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
    sample_rate: int
    positions: Union[np.ndarray, Tuple[interp1d, interp1d, interp1d]] # static or interpolated world coordinates in meters
    rotations: Union[np.ndarray, Slerp] # static or interpolated eurler rotations in rad (x,y,z)
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
            # Moving object: positions is a tuple of interp1d functions
            x = self.positions[0](sample_idx)
            y = self.positions[1](sample_idx)
            z = self.positions[2](sample_idx)
            return np.array([x, y, z])

    def get_rotation(self, sample_idx: float) -> np.ndarray:
        """Get interpolated rotation at specific sample_idx."""
        if self.static:
            # Static object: rotations is a single (3,) array of Euler angles
            return self.rotations.copy()
        else:
            # Moving object: rotations is a Slerp object
            rot = self.rotations(sample_idx)
            # Convert to Euler angles (xyz convention)
            return rot.as_euler('xyz')

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

    def get_relative_transformation(self, from_sample: float, to_sample: float) -> np.ndarray:
        """Get relative rigid transformation from one sample_idx to another."""
        if not self.static:
            if from_sample < 0:
               from_sample = 0
            vertices1 = self.get_vertices(from_sample)
            vertices2 = self.get_vertices(to_sample)
            centroid1 = np.mean(vertices1, axis=0)
            centroid2 = np.mean(vertices2, axis=0)
            centered1 = vertices1 - centroid1
            centered2 = vertices2 - centroid2
            # Compute rotation using Kabsch algorithm
            H = centered1.T @ centered2
            U, S, Vt = np.linalg.svd(H)
            # Ensure right-handed coordinate system
            d = np.linalg.det(Vt.T @ U.T)
            if d < 0:
                Vt[-1, :] *= -1
            R_vertices = Vt.T @ U.T
            t_vertices = centroid2 - R_vertices @ centroid1
            # Create 4x4 transformation matrix from vertices
            T_from_vertices = np.eye(4, dtype=np.float32) 
            T_from_vertices[:3, :3] = R_vertices
            T_from_vertices[:3, 3] = t_vertices
            return T_from_vertices
        return np.eye(4, dtype=np.float32) 

#    def get_transformation(self, sample_idx: float) -> np.ndarray:
#        """Get transformation matrix for a specific sample_idx."""
#        # Get position and rotation at the sample
#        position = self.get_position(sample_idx)
#        euler_angles = self.get_rotation(sample_idx)
#        
#        # Convert Euler angles to rotation matrix
#        rotation_matrix = _euler_to_rotation_matrix(euler_angles)
#        
#        # Create 4x4 transformation matrix
#        transformation = np.eye(4)
#        transformation[:3, :3] = rotation_matrix
#        transformation[:3, 3] = position
#        
#        return transformation
#
#    def get_relative_transformation(self, from_sample: float, to_sample: float) -> np.ndarray:
#        """Get relative rigid transformation from one sample_idx to another."""
#        # Get transformation matrices for both samples
#        T_from = self.get_transformation(from_sample)
#        T_to = self.get_transformation(to_sample)
#        
#        # Calculate relative transformation: T_relative = T_to * inv(T_from)
#        T_from_inv = np.linalg.inv(T_from)
#        T_relative = T_to @ T_from_inv
#        
#        return T_relative
