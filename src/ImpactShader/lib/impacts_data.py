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

import os, sys
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Collision:
    id_vertex: int = None
    force_vector = None

@dataclass
class ObjCollision:
    obj_idx: int # idx of the objct
    collision: Collision = field(default_factory=lambda: Collision())

@dataclass
class ImpactData:
    idx: int = None # idx of impact
    time: float = None # absolute time of impact from frame 0
    coord: Tuple[float, float, float] = None # impact coordinate
    collisions: List[ObjCollision] = = field(default_factory=list) # list of all collisions for all objects

    def __post_init__(self):
        pass

    def add_collision(self, obj_idx: int, collision: ObjImpacts):
        pass


