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
from ..core.vertex_solver import VertexSolver
from ..core.normal_solver import NormalSolver
from ..core.flight_path import FlightPath
from ..core.distance_solver import DistanceSolver
from ..core.force_solver import ForceSolver
from ..core.collision_solver import CollisionSolver

@dataclass
class CollisionEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.ps = PositionSolver(self.entity_manager)
        self.rs = RotationSolver(self.entity_manager)
        self.vs = VertexSolver(self.entity_manager)
        self.ns = NormalSolver(self.entity_manager)
        self.fp = FlightPath(self.entity_manager)
        self.ds = DistanceSolver(self.entity_manager)
        self.fs = ForceSolver(self.entity_manager)
        self.cs = CollisionSolver(self.entity_manager)

    def compute(self):
        config = self.entity_manager.get('config')
        obj_statics, obj_dyn, obj_pairs = ([] for _ in range(3))
        for config_obj in config.objects:
            if not config_obj.static and not config_obj.idx in obj_dyn:
                obj_dyn.append(config_obj.idx)
            if config_obj.static and not config_obj.idx in obj_statics:
                obj_statics.append(config_obj.idx)
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                obj_pairs.append([config.objects[i].idx, config.objects[j].idx])
        tasks_static = [self.fp.compute(obj_idx) for obj_idx in obj_statics]
        results_static = compute(*tasks_static)
        tasks_obj = [self.prebake_object(obj_idx) for obj_idx in obj_dyn]
        results_obj = compute(*tasks_obj)
        tasks_coll = [self.prebake_collision(objs_idx) for objs_idx in obj_pairs]
        results_coll = compute(*tasks_coll)
        tasks_force = [self.prebake_force(obj_idx) for obj_idx in obj_dyn]
        results_force = compute(*tasks_force)

    @delayed
    def prebake_object(self, obj_idx: int):
        config = self.entity_manager.get('config')
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                self.ps.compute(config_obj.idx)
                self.rs.compute(config_obj.idx)
                self.vs.compute(config_obj.idx)
                self.ns.compute(config_obj.idx)
                self.fp.compute(config_obj.idx)

    @delayed
    def prebake_collision(self, objs_idx: Tuple[int, int]):
        self.ds.compute(objs_idx)

    @delayed
    def prebake_force(self, obj_idx: int):
        self.fs.compute(obj_idx)

    @delayed
    def render(self, objs_idx: Tuple[int, int]):
        self.cs.compute(objs_idx)
