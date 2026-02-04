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
from enum import Enum
import numpy as np
from typing import Tuple, List, Union
from dataclasses import dataclass, field

class CollisionType(Enum):
    """Enum for different Type of collisions"""
    IMPACT = "impact"
    CONTACT = "contact"

@dataclass
class CollisionData:
    """Container for collision event data."""
    type: CollisionType
    obj1_idx: int
    obj2_idx: int
    frame: float # interpolated frame number
    frame_range: int = 1
    impulse_range: int = 0
    avg_distance: float = None
    threshold: float = None
    distances: Union[float, np.ndarray] = None
    collision_area: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None # Tuple[samples, obj1_triangles_idx, obj1_triangles_num, obj1_vertex_list, obj2_trangles_idx, obj2_triangles_num, obj2_vertex_list]

    def add_area(self, component: str, data: List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]):
        """Add a data component if not exist"""
        if getattr(self, component) is None:
            setattr(self, component, data)

    def update_type(self, collision_type: CollisionType):
        setattr(self, 'type', collision_type)
  
    def update_impulse_range(self, impulse_range: int):
        setattr(self, 'impulse_range', impulse_range)

    def update_frame_range(self, frame_range: int):
        setattr(self, 'frame_range', frame_range)

@dataclass
class tmpCollisionData:
    """Temporary container for solved collision event data."""
    obj1_idx: int
    obj1_idx: int
    restitution: float = None
    distances: np.ndarray = None
    consec_idx: np.ndarray = None
