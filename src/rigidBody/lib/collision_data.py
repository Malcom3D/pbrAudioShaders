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
import json
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

    def update_type(self, collision_type: CollisionType):
        setattr(self, 'type', collision_type)
  
    def update_impulse_range(self, impulse_range: int):
        setattr(self, 'impulse_range', impulse_range)

    def update_frame_range(self, frame_range: int):
        setattr(self, 'frame_range', frame_range)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert CollisionData to a serializable dictionary.
        
        Returns:
            Dictionary representation of the collision data.
        """
        data_dict = asdict(self)
        
        # Handle Enum serialization
        data_dict['type'] = self.type.value
        
        # Handle numpy array serialization
        if isinstance(self.distances, np.ndarray):
            data_dict['distances'] = self.distances.tolist()
        elif isinstance(self.distances, (int, float)):
            data_dict['distances'] = float(self.distances)
        
        # Handle other numpy types
        for key, value in data_dict.items():
            if isinstance(value, np.generic):
                data_dict[key] = value.item()
        
        return data_dict
    
    def save(self, filepath: str, indent: int = 2) -> None:
        """
        Save collision data to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            indent: JSON indentation level
        """
        data_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=indent)

    @classmethod
    def load(cls, filepath: str) -> 'CollisionData':
        """
        Load collision data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            CollisionData instance
        """
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
        
        # Convert string back to Enum
        data_dict['type'] = CollisionType(data_dict['type'])
        
        # Convert list back to numpy array if it was a distances array
        if 'distances' in data_dict and isinstance(data_dict['distances'], list):
            data_dict['distances'] = np.array(data_dict['distances'])
        
        return cls(**data_dict)
