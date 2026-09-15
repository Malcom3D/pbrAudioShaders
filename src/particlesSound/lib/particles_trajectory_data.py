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
from typing import Union, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

@dataclass
class ParticlesTrajectoryData:
    """
    Container for interpolated particle trajectory data.
    Manages merged frames (original + solved) for positions, rotations, and sizes.
    """
    obj_idx: int
    static: bool
    sfps: float
    sample_rate: int
    # Interpolators for each particle's position, rotation, and size
    # Stored as lists of CubicSpline/RotationSpline objects, one per particle
    positions: List[Tuple[CubicSpline, CubicSpline, CubicSpline]]
    rotations: List[RotationSpline]
    sizes: List[Tuple[CubicSpline, CubicSpline, CubicSpline]]
    states: np.ndarray  # Per-particle state over time (0=dead, 1=alive, 2=unborn)
    original_frames: np.ndarray
    solved_frames: np.ndarray

    def get_x(self) -> np.ndarray:
        """Get all unique sample times (merged frames)."""
        if not self.static:
            return np.unique(np.concatenate((self.original_frames, self.solved_frames)))
        return self.original_f_frames

    def get_position(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """Get interpolated position for a specific particle at a sample index."""
        if self.static or not self.positions:
            return np.zeros(3)
        
        if particle_idx >= len(self.positions):
            return np.zeros(3)

        pos_interp = self.positions[particle_idx]
        if pos_interp is None:
            return np.zeros(3)

        x = pos_interp[0](sample_idx)
        y = pos_interp[1](sample_idx)
        z = pos_interp[2](sample_idx)
        return np.array([x, y, z])

    def get_rotation(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """Get interpolated rotation for a specific particle at a sample index."""
        if self.static or not self.rotations:
            return np.array([0., 0., 0.])

        if particle_idx >= len(self.rotations):
            return np.array([0., 0., 0.])

        rot_interp = self.rotations[particle_idx]
        if rot_interp is None:
            return np.array([0., 0., 0.])
            
        return rot_interp(sample_idx).as_euler('XYZ')

    def get_size(self, particle_idx: int, sample_idx: float) -> np.ndarray:
        """Get interpolated size for a specific particle at a sample index."""
        if self.static or not self.sizes:
            return np.zeros(3)
        
        if particle_idx >= len(self.sizes):
            return np.zeros(3)

        size_interp = self.sizes[particle_idx]
        if size_interp is None:
            return np.zeros(3)

        x = size_interp[0](sample_idx)
        y = size_interp[1](sample_idx)
        z = size_interp[2](sample_idx)
        return np.array([x, y, z])

    def get_state(self, particle_idx: int, sample_idx: float) -> int:
        """Get particle state at a sample index (nearest neighbor)."""
        if self.static or self.states.size == 0:
            return 0
        
        if particle_idx >= self.states.shape[1]:
            return 0

        # Find the closest frame index
        all_frames = self.get_x()
        frame_idx = np.argmin(np.abs(all_frames - sample_idx))
        
        if frame_idx < self.states.shape[0]:
            return self.states[frame_idx, particle_idx]
        return 0

    def save(self, filepath: str) -> None:
        """Save data in pickle format (preserves interpolation objects)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_dict = {
            'obj_idx': self.obj_idx, 'static': self.static, 'sfps': self.sfps,
            'sample_rate': self.sample_rate, 'positions': self.positions,
            'rotations': self.rotations, 'sizes': self.sizes,
            'states': self.states, 'original_frames': self.original_frames,
            'solved_frames': self.solved_frames,
            '_format': 'ParticlesTrajectoryData_v1_pickle'
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Particles trajectory data saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ParticlesTrajectoryData':
        """Load data from pickle format."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if '_format' not in data or data['_format'] != 'ParticlesTrajectoryData_v1_pickle':
            raise ValueError("Invalid file format or version")
        return ParticlesTrajectoryData(**{k: v for k, v in data.items() if k != '_format'})

