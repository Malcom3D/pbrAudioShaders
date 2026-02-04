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
from typing import List, Tuple, Any
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
from ..core.force_synth import ForceSynth
from ..core.mesh2modal import Mesh2Modal
from ..core.collision_solver import CollisionSolver
from ..core.modal_composer import ModalComposer
from ..core.modal_luthier import ModalLuthier
from ..core.modal_player import ModalPlayer

@dataclass
class CollisionEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)
    obj_static: List[int] = field(default_factory=list)

    def prebake(self):
        self.ps = PositionSolver(self.entity_manager)
        self.rs = RotationSolver(self.entity_manager)
        self.vs = VertexSolver(self.entity_manager)
        self.ns = NormalSolver(self.entity_manager)
        self.fp = FlightPath(self.entity_manager)
        self.ds = DistanceSolver(self.entity_manager)
        self.fs = ForceSolver(self.entity_manager)
        self.force_synth = ForceSynth(self.entity_manager)
        self.mm = Mesh2Modal(self.entity_manager)

        config = self.entity_manager.get('config')
        obj_static, obj_dyn, obj_pairs = ([] for _ in range(3))
        for config_obj in config.objects:
            if not config_obj.static and not config_obj.idx in obj_dyn:
                obj_dyn.append(config_obj.idx)
            if config_obj.static and not config_obj.idx in obj_static:
                obj_static.append(config_obj.idx)
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                obj_pairs.append([config.objects[i].idx, config.objects[j].idx])
        self.obj_dyn = obj_dyn
        self.obj_static = obj_static
        tasks_static = [self.fp.compute(obj_idx) for obj_idx in obj_static]
        results_static = compute(*tasks_static)
        tasks_obj = [self.prebake_object(obj_idx) for obj_idx in obj_dyn]
        results_obj = compute(*tasks_obj)
        tasks_dists = [self.prebake_distances(objs_idx) for objs_idx in obj_pairs]
        results_dists = compute(*tasks_dists)
        tasks_force = [self.prebake_force(obj_idx) for obj_idx in obj_dyn]
        results_force = compute(*tasks_force)
#        tasks_modal = [self.prebake_modal(obj_idx) for obj_idx in obj_dyn + obj_static]
#        results_modal = compute(*tasks_modal)
        import shutil
        s = '/home/malcom3d/Documents/dsp'
        t = 'pbrAudioCache/dsp'
        files = os.listdir(s)
        for fname in files:
            shutil.copy(os.path.join(s, fname), t)

    def bake(self):
        self.collision_data = self.entity_manager.get('collisions')
        self.ml = ModalLuthier(self.entity_manager)

        tasks_collision = [self.bake_collision(self.collision_data[collision_idx]) for collision_idx in self.collision_data.keys()]
        results_collision = compute(*tasks_collision)
        tasks_composer = [self.bake_composer(self.collision_data[collision_idx]) for collision_idx in self.collision_data.keys()]
        results_composer = compute(*tasks_composer)
        tasks_luthier = [self.bake_luthier(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_luthier = compute(*tasks_luthier)
        players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        tasks_player = [self.bake_player(player) for player in players]
        results_player = compute(*tasks_player)
        tasks_save = [self.bake_save(player) for player in players]
        results_save = compute(*tasks_save)

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
    def prebake_distances(self, objs_idx: Tuple[int, int]):
        self.ds.compute(objs_idx)

    @delayed
    def prebake_force(self, obj_idx: int):
        self.fs.compute(obj_idx)
        self.force_synth.compute(obj_idx)

    @delayed
    def prebake_modal(self, obj_idx: int):
        self.mm.compute(obj_idx)

    @delayed
    def bake_collision(self, collision: Any):
        cs = CollisionSolver(self.entity_manager)
        cs.compute(collision)

    @delayed
    def bake_composer(self, collision: Any):
        mc = ModalComposer(self.entity_manager)
        mc.compute(collision)

    @delayed
    def bake_luthier(self, obj_idx: int):
        self.ml.compute(obj_idx)

    @delayed
    def bake_player(self, player: Any):
        player.compute()

    @delayed
    def bake_save(self, player: Any):
        player.save_synth_track()
