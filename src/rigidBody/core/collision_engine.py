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
from typing import List, Tuple
from dataclasses import dataclass, field
from dask import delayed, compute

from ..core.entity_manager import EntityManager
from ..core.position_solver import PositionSolver
from ..core.rotation_solver import RotationSolver
from ..core.landmark_solver import LandmarkSolver
from ..core.collision_solver import CollisionSolver
from ..core.flight_path import FlightPath

@dataclass
class CollisionEngine:
    entity_manager: EntityManager
    obj_done: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.cs = CollisionSolver(self.entity_manager)
        self.ps = PositionSolver(self.entity_manager)
        self.rs = RotationSolver(self.entity_manager)
        self.ls = LandmarkSolver(self.entity_manager)
#        self.fp = FlightPath(self.entity_manager)

    def compute(self):
        config = self.entity_manager.get('config')
        obj_pairs = []
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                obj_pairs.append([config.objects[i].idx, config.objects[j].idx])
        tasks = [self.prebake(objs_idx) for objs_idx in obj_pairs]
        results = compute(*tasks)

    @delayed
    def prebake(self, objs_idx: Tuple[int, int]):
        config = self.entity_manager.get('config')
        for config_obj in config.objects:
            if config_obj.idx == objs_idx[0] or config_obj.idx == objs_idx[1]:
                if not config_obj.static and not config_obj.idx in self.obj_done:
                    self.obj_done.append(config_obj.idx)
                    self.ps.compute(config_obj.idx)
                    self.rs.compute(config_obj.idx)
                    self.ls.compute(config_obj.idx)
#        self.cs.compute(objs_idx)
