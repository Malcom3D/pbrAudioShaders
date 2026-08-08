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

# pbrAudioShaders/src/fractureSound/lib/fracture_data.py
"""
Fracture event data structures.
"""

import os
import pickle
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from physicsSolver import CollisionData, ForceDataSequence, TrajectoryData


class FractureType(Enum):
    """Types of fracture events."""
    SHATTER = "shatter"      # Object breaks into multiple pieces
    CRACK = "crack"          # Single crack forms
    SNAP = "snap"            # Object snaps in two


@dataclass
class FragmentData:
    """Data about a fracture fragment."""
    obj_idx: int
    obj_name: str
    vertices: np.ndarray      # Vertex positions at fracture frame
    normals: np.ndarray       # Vertex normals at fracture frame
    faces: np.ndarray         # Face indices
    mass: float               # Mass of fragment (kg)
    volume: float             # Volume of fragment (m³)
    center_of_mass: np.ndarray  # Center of mass (m)
    inertia_tensor: np.ndarray  # Moment of inertia tensor
    parent_obj_idx: int       # Original object index before fracture
    is_shard: bool = False    # Whether this is a shard (created from fracture)
    fracture_frame: float = 0.0  # Frame at which this fragment was created
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'obj_idx': self.obj_idx,
            'obj_name': self.obj_name,
            'vertices': self.vertices.tolist(),
            'normals': self.normals.tolist(),
            'faces': self.faces.tolist(),
            'mass': float(self.mass),
            'volume': float(self.volume),
            'center_of_mass': self.center_of_mass.tolist(),
            'inertia_tensor': self.inertia_tensor.tolist(),
            'parent_obj_idx': self.parent_obj_idx,
            'is_shard': self.is_shard,
            'fracture_frame': float(self.fracture_frame)
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
    frame: float                     # Fracture frame (interpolated)
    original_obj_idx: int            # Original object before fracture
    original_obj_name: str           # Original object name
    fragment_indices: List[int]      # Indices of fragments after fracture
    
    # Pre-fracture data
    pre_fracture_velocity: np.ndarray  # Velocity of original object (m/s)
    pre_fracture_angular_velocity: np.ndarray  # Angular velocity (rad/s)
    pre_fracture_force: np.ndarray     # Force at fracture (N)
    pre_fracture_stress: np.ndarray    # Stress tensor at fracture (Pa)
    
    # Post-fracture data
    fragment_velocities: List[np.ndarray]  # Velocity of each fragment (m/s)
    fragment_angular_velocities: List[np.ndarray]  # Angular velocity of each fragment (rad/s)
    
    # Fracture parameters
    fracture_energy: float = 0.0       # Energy released in fracture (J)
    crack_velocity: float = 500.0      # Crack propagation velocity (m/s)
    crack_duration: float = 0.01       # Crack propagation duration (s)
    crack_length: float = 0.0          # Total crack length (m)
    
    # Material properties at fracture
    young_modulus: float = 1e9         # Young's modulus (Pa)
    density: float = 1000.0            # Density (kg/m³)
    damping: float = 0.02              # Damping ratio
    failure_stress: float = 1e6        # Failure stress (Pa)
    
    # Pre-computed data
    fragment_data: List[FragmentData] = field(default_factory=list)
    collision_data: List[CollisionData] = field(default_factory=list)
    force_data: List[ForceDataSequence] = field(default_factory=list)
    
    # Modal modifications
    frequency_shift: float = 0.0       # Frequency shift due to fracture
    damping_change: float = 0.0        # Damping change due to fracture
    
    def __post_init__(self):
        if self.crack_duration <= 0:
            # Estimate crack duration from crack length and velocity
            if self.crack_length > 0 and self.crack_velocity > 0:
                self.crack_duration = self.crack_length / self.crack_velocity
            else:
                self.crack_duration = 0.01  # Default 10ms
    
    def get_fragment_by_idx(self, obj_idx: int) -> Optional[FragmentData]:
        """Get fragment data by object index."""
        for frag in self.fragment_data:
            if frag.obj_idx == obj_idx:
                return frag
        return None
    
    def get_fragment_velocity(self, obj_idx: int) -> Optional[np.ndarray]:
        """Get fragment velocity by object index."""
        for i, frag in enumerate(self.fragment_data):
            if frag.obj_idx == obj_idx and i < len(self.fragment_velocities):
                return self.fragment_velocities[i]
        return None
    
    def save(self, filepath: str) -> None:
        """Save fracture event to pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        save_dict = {
            'fracture_type': self.fracture_type,
            'frame': self.frame,
            'original_obj_idx': self.original_obj_idx,
            'original_obj_name': self.original_obj_name,
            'fragment_indices': self.fragment_indices,
            'pre_fracture_velocity': self.pre_fracture_velocity.tolist() if self.pre_fracture_velocity is not None else None,
            'pre_fracture_angular_velocity': self.pre_fracture_angular_velocity.tolist() if self.pre_fracture_angular_velocity is not None else None,
            'pre_fracture_force': self.pre_fracture_force.tolist() if self.pre_fracture_force is not None else None,
            'pre_fracture_stress': self.pre_fracture_stress.tolist() if self.pre_fracture_stress is not None else None,
            'fragment_velocities': [v.tolist() for v in self.fragment_velocities] if self.fragment_velocities else [],
            'fragment_angular_velocities': [v.tolist() for v in self.fragment_angular_velocities] if self.fragment_angular_velocities else [],
            'fracture_energy': self.fracture_energy,
            'crack_velocity': self.crack_velocity,
            'crack_duration': self.crack_duration,
            'crack_length': self.crack_length,
            'young_modulus': self.young_modulus,
            'density': self.density,
            'damping': self.damping,
            'failure_stress': self.failure_stress,
            'frequency_shift': self.frequency_shift,
            'damping_change': self.damping_change,
            'fragment_data': [f.to_dict() for f in self.fragment_data],
            '_format': 'FractureEvent_v2_pickle'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FractureEvent':
        """Load fracture event from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if '_format' not in data or data['_format'] != 'FractureEvent_v2_pickle':
            raise ValueError("Invalid file format")
        
        # Reconstruct arrays
        if data['pre_fracture_velocity'] is not None:
            data['pre_fracture_velocity'] = np.array(data['pre_fracture_velocity'])
        if data['pre_fracture_angular_velocity'] is not None:
            data['pre_fracture_angular_velocity'] = np.array(data['pre_fracture_angular_velocity'])
        if data['pre_fracture_force'] is not None:
            data['pre_fracture_force'] = np.array(data['pre_fracture_force'])
        if data['pre_fracture_stress'] is not None:
            data['pre_fracture_stress'] = np.array(data['pre_fracture_stress'])
        
        data['fragment_velocities'] = [np.array(v) for v in data['fragment_velocities']]
        data['fragment_angular_velocities'] = [np.array(v) for v in data['fragment_angular_velocities']]
        data['fragment_data'] = [FragmentData.from_dict(f) for f in data['fragment_data']]
        
        return cls(**{k: v for k, v in data.items() if k != '_format'})
