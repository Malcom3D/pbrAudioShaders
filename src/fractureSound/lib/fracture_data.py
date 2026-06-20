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
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from physicsSolver import CollisionData, ForceDataSequence

class FractureType(Enum):
    """Types of fracture events."""
    SHATTER = "shatter"      # Object breaks into multiple pieces
    CRACK = "crack"          # Single crack forms
    SNAP = "snap"            # Object snaps in two
    TEAR = "tear"            # Tearing fracture

@dataclass
class FragmentData:
    """Data about a fracture fragment."""
    obj_idx: int
    vertices: np.ndarray      # Vertex positions
    normals: np.ndarray       # Vertex normals
    faces: np.ndarray         # Face indices
    mass: float               # Mass of fragment (kg)
    volume: float             # Volume of fragment (m³)
    center_of_mass: np.ndarray # Center of mass (m)
    inertia_tensor: np.ndarray # Moment of inertia tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'obj_idx': self.obj_idx,
            'vertices': self.vertices.tolist(),
            'normals': self.normals.tolist(),
            'faces': self.faces.tolist(),
            'mass': float(self.mass),
            'volume': float(self.volume),
            'center_of_mass': self.center_of_mass.tolist(),
            'inertia_tensor': self.inertia_tensor.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FragmentData':
        """Create from dictionary."""
        data['vertices'] = np.array(data['vertices'])
        data['normals'] = np.array(data['normals'])
        data['faces'] = np.array(data['faces'])
        data['center_of_mass'] = np.array(data['center_of_mass'])
        data['inertia_tensor'] = np.array(data['inertia_tensor'])
        return cls(**data)

@dataclass
class FractureEvent:
    """Complete data for a fracture event."""
    
    fracture_type: FractureType
    frame: float                     # Fracture frame
    original_obj_idx: int            # Original object before fracture
    fragment1_idx: int               # First fragment after fracture
    fragment2_idx: int               # Second fragment after fracture
    collisions: List[CollisionData]  # Collisions at fracture time
    forces: List[ForceDataSequence]  # Forces at fracture time
    
    # Fracture parameters
    crack_velocity: float = 500.0    # Crack propagation velocity (m/s)
    crack_duration: float = None     # Crack propagation duration (s)
    fracture_energy: float = None     # Energy released in fracture (J)
    fragment_velocities: List[np.ndarray] = field(default_factory=list)  # Initial fragment velocities
    
    # Modal modifications
    frequency_shift: float = 0.0      # Frequency shift due to fracture
    damping_change: float = 0.0       # Damping change due to fracture
    
    # Pre-computed data
    fragment1_data: FragmentData = None
    fragment2_data: FragmentData = None
    
    def __post_init__(self):
        if self.crack_duration is None:
            # Estimate crack duration (simplified)
            # Assuming typical crack velocity ~500 m/s and fragment size ~0.1 m
            self.crack_duration = 0.1 / self.crack_velocity
    
    def save(self, filepath: str) -> None:
        """Save fracture event to pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        save_dict = {
            'fracture_type': self.fracture_type,
            'frame': self.frame,
            'original_obj_idx': self.original_obj_idx,
            'fragment1_idx': self.fragment1_idx,
            'fragment2_idx': self.fragment2_idx,
            'collisions': self.collisions,
            'forces': self.forces,
            'crack_velocity': self.crack_velocity,
            'crack_duration': self.crack_duration,
            'fracture_energy': self.fracture_energy,
            'fragment_velocities': [v.tolist() for v in self.fragment_velocities],
            'frequency_shift': self.frequency_shift,
            'damping_change': self.damping_change,
            'fragment1_data': self.fragment1_data.to_dict() if self.fragment1_data else None,
            'fragment2_data': self.fragment2_data.to_dict() if self.fragment2_data else None,
            '_format': 'FractureEvent_v1_pickle'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FractureEvent':
        """Load fracture event from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if '_format' not in data or data['_format'] != 'FractureEvent_v1_pickle':
            raise ValueError("Invalid file format")
        
        # Reconstruct fragment velocities
        data['fragment_velocities'] = [
            np.array(v) for v in data['fragment_velocities']
        ]
        
        # Reconstruct fragment data
        if data['fragment1_data']:
            data['fragment1_data'] = FragmentData.from_dict(data['fragment1_data'])
        if data['fragment2_data']:
            data['fragment2_data'] = FragmentData.from_dict(data['fragment2_data'])
        
        return cls(**{k: v for k, v in data.items() if k != '_format'})
