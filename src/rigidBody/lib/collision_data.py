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
from typing import Tuple
from dataclasses import dataclass, field

from ..lib.collision_area import CollisionArea

@dataclass
class CollisionData:
    """Container for collision event data."""
    frame: int # interpolated frame number
    obj_idx1: int
    obj_idx2: int
    distance: float
    collision_area: Tuple[CollisionArea, CollisionArea] = None

@dataclass
class tmpCollisionData:
    """Temporary container for solved collision event data."""
    name: str
    obj_idx1: int
    obj_idx2: int
    frame: float
    delta_time: float
    value: np.ndarray
    frames_idx: np.ndarray = None
    idx_smallest: np.ndarray = None
    dists: np.ndarray = None
