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
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

@dataclass
class ParticlesCollisionsVoxels:
    """
    Stores collision data for massive particles objects,, remapped to surface voxel indices.
    """
    particles_obj_idx: int
    # Frame -> {voxel_idx_tuple -> {'obj_idx': int, 'force': float, 'count': int}}
    # The voxel_idx_tuple is (obj_idx, *voxel_index) to be globally unique
    collisions: Dict[float, Dict[Tuple[int, ...], Dict[str, Any]]] = field(default_factory=dict)

    def add_collision(self, sample_idx: float, obj_idx: int, voxel_ids: np.ndarray, velocity: np.ndarray, forces: np.ndarray):
        """Adds a collision event to a specific voxel."""
        if sample_idx not in self.collisions:
            self.collisions[sample_idx] = {}
        self.collisions[sample_idx][obj_idx] = {
            'obj_idx': obj_idx,
            'voxel_ids': voxel_ids,
            'velocity': velocity,
            'forces': forces
        }

    def save(self, filepath: str):
        """Saves the collision data to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"ParticlesCollisionsVoxels data saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ParticlesCollisionsVoxels':
        """Loads collision data from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

