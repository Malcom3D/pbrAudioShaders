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
from enum import Enum
import numpy as np
from typing import Tuple, List, Union, Dict, Any
from dataclasses import dataclass, field, asdict

class CollisionType(Enum):
    """Enum for different Type of collisions"""
    IMPACT = "impact"
    CONTACT = "contact"
    CONNECTED = "connected"

@dataclass
class CollisionData:
    """Container for collision event data."""
    type: CollisionType
    obj1_idx: int
    obj2_idx: int
    frame: float = None # interpolated frame number
    frame_range: int = 1
    impulse_range: int = 0
    avg_distance: float = None
    threshold: float = None
    distances: Union[float, np.ndarray] = None
    samples: np.ndarray = None
    valid: bool = True

    def update_type(self, collision_type: CollisionType):
        setattr(self, 'type', collision_type)
  
    def update_impulse_range(self, impulse_range: int):
        setattr(self, 'impulse_range', impulse_range)

    def update_frame_range(self, frame_range: int):
        setattr(self, 'frame_range', frame_range)
    
    def save(self, filepath: str) -> None:
        """
        Save collision data to a pickle file.
        
        Args:
            filepath: Path to save the pickle file
        """
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        save_dict = {
            'type': self.type,
            'obj1_idx': self.obj1_idx,
            'obj2_idx': self.obj2_idx,
            'frame': self.frame,
            'frame_range': self.frame_range,
            'impulse_range': self.impulse_range,
            'avg_distance': self.avg_distance,
            'threshold': self.threshold,
            'distances': self.distances,
            'samples': self.samples
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"CollisionData data saved to {filepath}")        

    @staticmethod
    def load(filepath: str) -> 'CollisionData':
        """
        Load collision data from a pickle file.
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            CollisionData instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct the object
        return CollisionData(
            type=data['type'],
            obj1_idx=data['obj1_idx'],
            obj2_idx=data['obj2_idx'],
            frame=data['frame'],
            frame_range=data['frame_range'],
            impulse_range=data['impulse_range'],
            avg_distance=data['avg_distance'],
            threshold=data['threshold'],
            distances=data['distances'],
            samples=data['samples']
        )
