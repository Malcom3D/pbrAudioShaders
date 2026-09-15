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
from typing import Dict, List, Any

@dataclass
class ParticlesCollisionsPoints:
    """
    Stores collision data for hero particles objects, keeping track of precise contact points.
    """
    particles_obj_idx: int
    # Frame -> {particle_idx -> {'obj_idx': int, 'position': np.ndarray, 'velocity': np.ndarray, 'force': float}}
    collisions: Dict[float, Dict[int, Dict[str, Any]]] = field(default_factory=dict)

    def add_collision(self, frame: float, particle_idx: int, obj_idx: int, position: np.ndarray, velocity: np.ndarray, force: float):
        """Adds a single collision event."""
        if frame not in self.collisions:
            self.collisions[frame] = {}
        self.collisions[frame][particle_idx] = {
            'obj_idx': obj_idx,
            'position': position,
            'velocity': velocity,
            'force': force
        }

    def save(self, filepath: str):
        """Saves the collision data to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"ParticlesCollisionsPoints data saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ParticlesCollisionsPoints':
        """Loads collision data from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

