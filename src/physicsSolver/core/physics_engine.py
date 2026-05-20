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
from multiprocessing import Pool, cpu_count
from functools import partial
import copy

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

from ..lib.functions import _update_status


def _process_position(entity_manager, obj_idx):
    """Wrapper function for parallel execution of position solving."""
    ps = PositionSolver(entity_manager)
    ps.compute(obj_idx)


def _process_rotation(entity_manager, obj_idx):
    """Wrapper function for parallel execution of rotation solving."""
    rs = RotationSolver(entity_manager)
    rs.compute(obj_idx)


def _process_vertex(entity_manager, obj_idx):
    """Wrapper function for parallel execution of vertex solving."""
    vs = VertexSolver(entity_manager)
    vs.compute(obj_idx)


def _process_normal(entity_manager, obj_idx):
    """Wrapper function for parallel execution of normal solving."""
    ns = NormalSolver(entity_manager)
    ns.compute(obj_idx)


def _process_trajectory(entity_manager, obj_idx):
    """Wrapper function for parallel execution of trajectory computation."""
    fp = FlightPath(entity_manager)
    fp.compute(obj_idx)


def _process_static(entity_manager, obj_idx):
    """Wrapper function for parallel execution of static object computation."""
    fp = FlightPath(entity_manager)
    fp.compute(obj_idx)


def _process_distances(entity_manager, objs_idx):
    """Wrapper function for parallel execution of distance computation."""
    ds = DistanceSolver(entity_manager)
    ds.compute(objs_idx)


def _process_force(entity_manager, obj_idx):
    """Wrapper function for parallel execution of force computation."""
    fs = ForceSolver(entity_manager)
    fs.compute(obj_idx)


def _process_force_synth(entity_manager, obj_idx):
    """Wrapper function for parallel execution of force synthesis."""
    fsy = ForceSynth(entity_manager)
    fsy.compute(obj_idx)


def _process_collision(entity_manager, collision):
    """Wrapper function for parallel execution of collision solving."""
    cs = CollisionSolver(entity_manager)
    cs.compute(collision)


@dataclass
class physicsEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)
    obj_static: List[int] = field(default_factory=list)
    obj_pairs: List[int] = field(default_factory=list)
    num_workers: int = field(default=None)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        self.trajectories_dir = f"{config.system.cache_path}/trajectories"
        self.forces_dir = f"{config.system.cache_path}/forces_data"
        self.modalvertices_dir = f"{config.system.cache_path}/modalvertices"
        self.scoretracks_dir = f"{config.system.cache_path}/scoretracks"

        # Set number of workers (default to CPU count)
        if self.num_workers is None:
            self.num_workers = cpu_count()

        # Ensure status directory exists
        os.makedirs(self.status_dir, exist_ok=True)

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
        _update_status(f"{self.status_dir}/bake", 0)

        # Phase 1: Static objects
        with Pool(processes=self.num_workers) as pool:
            # Process static objects
            static_func = partial(_process_static, self.entity_manager)
            pool.map(static_func, self.obj_static)
        _update_status(f"{self.status_dir}/bake", 9)

        # Phase 2: Position solving
        with Pool(processes=self.num_workers) as pool:
            pos_func = partial(_process_position, self.entity_manager)
            pool.map(pos_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 18)

        # Phase 3: Rotation solving
        with Pool(processes=self.num_workers) as pool:
            rot_func = partial(_process_rotation, self.entity_manager)
            pool.map(rot_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 27)

        # Phase 4: Vertex solving
        with Pool(processes=self.num_workers) as pool:
            vertex_func = partial(_process_vertex, self.entity_manager)
            pool.map(vertex_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 36)

        # Phase 5: Normal solving
        with Pool(processes=self.num_workers) as pool:
            normal_func = partial(_process_normal, self.entity_manager)
            pool.map(normal_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 45)

        # Phase 6: Trajectory computation
        with Pool(processes=self.num_workers) as pool:
            traj_func = partial(_process_trajectory, self.entity_manager)
            pool.map(traj_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 54)

        # Cleanup temporary trajectory data
        for obj_idx in self.obj_dyn + self.obj_static:
            self._cleanup_tmp_trajectories(obj_idx)
        _update_status(f"{self.status_dir}/bake", 55)

        # Phase 7: Distance computation
        with Pool(processes=self.num_workers) as pool:
            dist_func = partial(_process_distances, self.entity_manager)
            pool.map(dist_func, self.obj_pairs)
        _update_status(f"{self.status_dir}/bake", 72)

        # Phase 8: Force computation
        force_objects = self.obj_dyn + self.obj_static
        with Pool(processes=self.num_workers) as pool:
            force_func = partial(_process_force, self.entity_manager)
            pool.map(force_func, force_objects)
        _update_status(f"{self.status_dir}/bake", 80)

        # Phase 9: Collision solving
        collisions = self.entity_manager.get('collisions')
        collision_list = list(collisions.values())
        with Pool(processes=self.num_workers) as pool:
            coll_func = partial(_process_collision, self.entity_manager)
            pool.map(coll_func, collision_list)
        _update_status(f"{self.status_dir}/bake", 90)

        # Phase 10: Force synthesis
        with Pool(processes=self.num_workers) as pool:
            force_synth_func = partial(_process_force_synth, self.entity_manager)
            pool.map(force_synth_func, self.obj_dyn)
        _update_status(f"{self.status_dir}/bake", 99)

        # Ensure directories exist
        os.makedirs(self.collisions_dir, exist_ok=True)
        os.makedirs(self.modalvertices_dir, exist_ok=True)
        os.makedirs(self.scoretracks_dir, exist_ok=True)

        # Save collision data
        collision_data = self.entity_manager.get('collisions')
        print('Saved collisions: ', len(collision_data))
        for c_idx in collision_data.keys():
            collision_data[c_idx].save(f"{self.collisions_dir}/{c_idx:05d}.pkl")

        # Save modal vertices and score tracks data
        modal_vertices = self.entity_manager.get('modal_vertices')
        print('Saved modal_vertices: ', len(modal_vertices))
        for m_idx in modal_vertices.keys():
            modal_vertices[m_idx].save(f"{self.modalvertices_dir}/{m_idx:05d}.json")

        score_tracks = self.entity_manager.get('score_tracks')
        print('Saved score_tracks: ', len(score_tracks))
        for s_idx in score_tracks.keys():
            score_tracks[s_idx].save(f"{self.scoretracks_dir}/{s_idx:05d}.pkl")

        _update_status(f"{self.status_dir}/bake", 100)

    def _cleanup_tmp_trajectories(self, obj_idx: int):
        """Remove temporary trajectory data for the given object."""
        trajectories = self.entity_manager.get('trajectories')
        tmp_trajectories = copy.deepcopy(trajectories)
        for key in list(tmp_trajectories.keys()):
            if isinstance(tmp_trajectories[key], tmpTrajectoryData) and tmp_trajectories[key].obj_idx == obj_idx:
                del tmp_trajectories[key]
        self.entity_manager._trajectories = tmp_trajectories

