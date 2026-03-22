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
from typing import List, Tuple, Any, Dict
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
from ..core.force_synth import ForceSynth

from ..lib.collision_data import CollisionData
from ..lib.trajectory_data import TrajectoryData, tmpTrajectoryData
from ..lib.force_data import ForceDataSequence
from ..lib.modal_vertices import ModalVertices
from ..lib.score_data import ScoreTrack

@dataclass
class physicsEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)
    obj_static: List[int] = field(default_factory=list)
    obj_pairs: List[int] = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        self.trajectories_dir = f"{config.system.cache_path}/trajectories"
        self.forces_dir = f"{config.system.cache_path}/forces_data"
        self.modalvertices_dir = f"{config.system.cache_path}/modalvertices"
        self.scoretracks_dir = f"{config.system.cache_path}/scoretracks"
        obj_static, obj_dyn, obj_pairs = ([] for _ in range(3))
        for config_obj in config.objects:
            if not config_obj.static and not config_obj.idx in obj_dyn:
                self.obj_dyn.append(config_obj.idx)
            if config_obj.static and not config_obj.idx in obj_static:
                self.obj_static.append(config_obj.idx)
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                self.obj_pairs.append([config.objects[i].idx, config.objects[j].idx])

    def bake(self):
#        self.ps = PositionSolver(self.entity_manager)
#        self.rs = RotationSolver(self.entity_manager)
#        self.vs = VertexSolver(self.entity_manager)
#        self.ns = NormalSolver(self.entity_manager)
#        self.fp = FlightPath(self.entity_manager)
#        self.ds = DistanceSolver(self.entity_manager)
#        self.fs = ForceSolver(self.entity_manager)

#        tasks_static = [self.fp.compute(obj_idx) for obj_idx in self.obj_static]
#        results_static = compute(*tasks_static)

        tasks_static = [self.static(obj_idx) for obj_idx in self.obj_static]
        results_static = compute(*tasks_static)

        tasks_pos = [self.position(obj_idx) for obj_idx in self.obj_dyn]
        results_pos = compute(*tasks_pos)
        tasks_rot = [self.rotation(obj_idx) for obj_idx in self.obj_dyn]
        results_rot = compute(*tasks_rot)
        tasks_vertex = [self.vertex(obj_idx) for obj_idx in self.obj_dyn]
        results_vertex = compute(*tasks_vertex)
        tasks_norm = [self.normal(obj_idx) for obj_idx in self.obj_dyn]
        results_norm = compute(*tasks_norm)
        tasks_traj = [self.trajectory(obj_idx) for obj_idx in self.obj_dyn]
        results_traj = compute(*tasks_traj)

        # Remove temporary trajectory data for this object
        for obj_idx in self.obj_dyn + self.obj_static:
            self._cleanup_tmp_trajectories(obj_idx)

        tasks_dists = [self.distances(objs_idx) for objs_idx in self.obj_pairs]
        results_dists = compute(*tasks_dists)

        tasks_force = [self.force(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_force = compute(*tasks_force)

        collisions = self.entity_manager.get('collisions')
        tasks_collision = [self.collision(collisions[collision_idx]) for collision_idx in collisions.keys()]
        results_collision = compute(*tasks_collision)

        tasks_force_synth = [self.force_synth(obj_idx) for obj_idx in self.obj_dyn]
        results_force_synth = compute(*tasks_force_synth)

        # Ensure directory exists
        os.makedirs(self.collisions_dir, exist_ok=True)
        os.makedirs(self.modalvertices_dir, exist_ok=True)
        os.makedirs(self.scoretracks_dir, exist_ok=True)

        # Save collision data
        collision_data = self.entity_manager.get('collisions')
        print('Save collisions: ', len(collision_data))
        for c_idx in collision_data.keys():
            collision_data[c_idx].save(f"{self.collisions_dir}/{c_idx:05d}.pkl")

        # Save modal vertices and score tracks data
        modal_vertices = self.entity_manager.get('modal_vertices')
        print('Save modal_vertices: ', len(modal_vertices))
        for m_idx in modal_vertices.keys():
            modal_vertices[m_idx].save(f"{self.modalvertices_dir}/{m_idx:05d}.json")

        score_tracks = self.entity_manager.get('score_tracks')
        print('Save score_tracks: ', len(score_tracks))
        for s_idx in score_tracks.keys():
            score_tracks[s_idx].save(f"{self.scoretracks_dir}/{s_idx:05d}.pkl")

    def _cleanup_tmp_trajectories(self, obj_idx: int):
        """Remove temporary trajectory data for the given object."""
        import copy
        trajectories = self.entity_manager.get('trajectories')
        tmp_trajectories = copy.deepcopy(trajectories)
        for key in trajectories.keys():
            if isinstance(tmp_trajectories[key], tmpTrajectoryData) and tmp_trajectories[key].obj_idx == obj_idx:
                del tmp_trajectories[key]
        self.entity_manager._trajectories = tmp_trajectories

    @delayed
    def position(self, obj_idx: int):
        ps = PositionSolver(self.entity_manager)
        ps.compute(obj_idx)

    @delayed
    def rotation(self, obj_idx: int):
        rs = RotationSolver(self.entity_manager)
        rs.compute(obj_idx)

    @delayed
    def vertex(self, obj_idx: int):
        vs = VertexSolver(self.entity_manager)
        vs.compute(obj_idx)

    @delayed
    def normal(self, obj_idx: int):
        ns = NormalSolver(self.entity_manager)
        ns.compute(obj_idx)

    @delayed
    def trajectory(self, obj_idx: int):
        fp = FlightPath(self.entity_manager)
        fp.compute(obj_idx)

    @delayed
    def static(self, obj_idx: int):
        fp = FlightPath(self.entity_manager)
        fp.compute(obj_idx)

    @delayed
    def distances(self, objs_idx: Tuple[int, int]):
        ds = DistanceSolver(self.entity_manager)
        ds.compute(objs_idx)

    @delayed
    def force(self, obj_idx: int):
        fs = ForceSolver(self.entity_manager)
        fs.compute(obj_idx)

    @delayed
    def force_synth(self, obj_idx: int):
        fsy = ForceSynth(self.entity_manager)
        fsy.compute(obj_idx)

    @delayed
    def collision(self, collision: CollisionData):
        cs = CollisionSolver(self.entity_manager)
        cs.compute(collision)
