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
from ..core.distance_detector import DistanceDetector
from ..core.collision_solver import CollisionSolver
#from ..core.flight_path import FlightPath

@dataclass
class CollisionEngine:
    entity_manager: EntityManager

    def __post_init__(self):
        self.dd = DistanceDetector(self.entity_manager)
        self.cs = CollisionSolver(self.entity_manager)
#        self.fp = FlightPath(self.entity_manager)

    def compute(self):
        config = self.entity_manager.get('config')
        obj_pairs = []
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                obj_pairs.append([config.objects[i].idx, config.objects[j].idx])
        tasks = [self.prebake(objs_idx) for objs_idx in obj_pairs]
        results = compute(*tasks)

        # to dask
#        for frames_idx, obj1, obj2, distance, points in results:
#            self.fp.compute(frames_idx, obj1, distance, points[0])
#            self.fp.compute(frames_idx, obj2, distance, points[1])

    @delayed
    def prebake(self, objs_idx: Tuple[int, int]):
        detected_distances = self.dd.compute(objs_idx)
        if not detected_distances == None:
            print('CollisionSolver: ', objs_idx)
            self.cs.compute(objs_idx, detected_distances)
