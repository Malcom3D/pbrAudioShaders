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
from enum import Enum
from typing import Tuple, List, Union
from dataclasses import dataclass, field

class CollisionType(Enum):
    """Enum for different Type of collisions"""
    IMPACT = "impact"
    CONTACT = "contact"

@dataclass
class CollisionArea:
    """Represents a collision area for a objects."""
    obj_idx: int
    faces_idx: np.ndarray # Shape: (n_id, 3) - colliding faces

@dataclass
class CollisionData:
    """Container for collision event data."""
    type: CollisionType
    obj1_idx: int
    obj2_idx: int
    frame: float # interpolated frame number
    frame_range: int = 1
    distances: Union[float, np.ndarray] = None
    collision_area: List[Tuple[int, Tuple[CollisionArea, CollisionArea]]] = None # List[samples, [obj1_triangles_idx, obj2_trangles_idx]]

    def add_area(self, component: str, data: List[Tuple[int, Tuple[CollisionArea, CollisionArea]]]):
        """Add a data component if not exist"""
        if getattr(self, component) is None:
            setattr(self, component, data)

@dataclass
class tmpCollisionData:
    """Temporary container for solved collision event data."""
    obj1_idx: int
    obj1_idx: int
    restitution: float = None
    distances: np.ndarray = None
    consec_idx: np.ndarray = None
