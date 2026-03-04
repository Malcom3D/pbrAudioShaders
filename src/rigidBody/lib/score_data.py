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

import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class ScoreEvent:
    """Represents a single score event at a specific sample."""
    type: int
    sample_idx: int
    vertex_ids: np.ndarray  # Variable length array of vertex indices
    contact_area: float = None # 
    force: float = None # Excitation force magnitude
    coupling_data: np.ndarray = None # Array of [other_obj_idx, coupling_strength] pairs
    
    def __post_init__(self):
        # Ensure arrays are proper numpy arrays
        self.vertex_ids = np.asarray(self.vertex_ids, dtype=np.int32)
        self.coupling_data = np.asarray(self.coupling_data, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': int(self.type),
            'sample_idx': int(self.sample_idx),
            'vertex_ids': self.vertex_ids.tolist(),
            'contact_area': float(self.contact_area),
            'force': float(self.force),
            'coupling_data': self.coupling_data.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreEvent':
        """Create from dictionary."""
        return cls(
            sample_idx=data['sample_idx'],
            vertex_ids=np.array(data['vertex_ids'], dtype=np.int32),
            force=data['force'],
            coupling_data=np.array(data['coupling_data'], dtype=np.float32)
        )

@dataclass
class ScoreTrack:
    """Represents a score track for a single object."""
    obj_idx: int
    obj_name: str
    events: List[ScoreEvent] = field(default_factory=list)
    
    def add_event(self, event: ScoreEvent) -> None:
        """Add an event to the track."""
        self.events.append(event)
    
    def get_events_at_sample(self, sample_idx: int) -> List[ScoreEvent]:
        """Get all events at a specific sample index."""
        return [event for event in self.events if event.sample_idx == sample_idx]
    
    def save(self, filepath: str) -> None:
        """
        Save the ScoreTrack to a pickle file.
        
        Args:
            filepath: Path to save the JSON file
            indent: JSON indentation level (None for compact format)
        """
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ScoreTrack':
        """
        Load a ScoreTrack from a pickle file.
        
        Args:
            filepath: Path to the JSON file to load
            
        Returns:
            Loaded ScoreTrack instance
        """
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        return obj
