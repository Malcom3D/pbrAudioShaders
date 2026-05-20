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

from physicsSolver import EntityManager, ForceDataSequence, ModalVertices, ScoreTrack, CollisionData, TrajectoryData
from ..core.mesh2modal import Mesh2Modal
from ..core.modal_composer import ModalComposer
from ..core.modal_luthier import ModalLuthier
from ..core.modal_player import ModalPlayer

from ..lib.sample_counter import SampleCounter
from ..lib.connected_buffer import ConnectedBuffer

from ..lib.functions import _update_status


def _prebake_modal(entity_manager, obj_idx):
    """Wrapper function for parallel execution of modal prebaking."""
    mm = Mesh2Modal(entity_manager)
    mm.compute(obj_idx)


def _prebake_composer(entity_manager, collision):
    """Wrapper function for parallel execution of composer prebaking."""
    mc = ModalComposer(entity_manager)
    mc.compute(collision)


def _bake_luthier(entity_manager, obj_idx):
    """Wrapper function for parallel execution of luthier baking."""
    ml = ModalLuthier(entity_manager)
    ml.compute(obj_idx)


def _bakeake_player(entity_manager, player):
    """Wrapper function for parallel execution of player baking."""
    player.compute()


def _bake_save(entity_manager, player):
    """Wrapper function for parallel execution of player saving."""
    player.save_synth_tracks()


@dataclass
class rigidBodyEngine:
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

        trajectories = self.entity_manager.get('trajectories')
        if len(trajectories) == 0:
            if os.path.exists(f"{self.trajectories_dir}") and not len(os.listdir(f"{self.trajectories_dir}")) == 0:
                for filename in os.listdir(f"{self.trajectories_dir}"):
                    if filename.endswith('.pkl'):
                        trajectories = TrajectoryData.load(f"{self.trajectories_dir}/{filename}")
                        self.entity_manager.register('trajectories', trajectories)

        collisions = self.entity_manager.get('collisions')
        if len(collisions) == 0:
            if os.path.exists(f"{self.collisions_dir}") and not len(os.listdir(f"{self.collisions_dir}")) == 0:
                for filename in os.listdir(f"{self.collisions_dir}"):
                    if filename.endswith('.pkl'):
                        collisions = CollisionData.load(f"{self.collisions_dir}/{filename}")
                        self.entity_manager.register('collisions', collisions)

        forces = self.entity_manager.get('forces')
        if len(forces) == 0:
            if os.path.exists(f"{self.forces_dir}") and not len(os.listdir(f"{self.forces_dir}")) == 0:
                for filename in os.listdir(f"{self.forces_dir}"):
                    if filename.endswith('.pkl'):
                        forces = ForceDataSequence.load(f"{self.forces_dir}/{filename}")
                        self.entity_manager.register('forces', forces)
            forces = self.entity_manager.get('forces')

        modal_vertices = self.entity_manager.get('modal_vertices')
        if len(modal_vertices) == 0:
            if os.path.exists(self.modalvertices_dir):
                filenames = os.listdir(self.modalvertices_dir)
                for filename in filenames:
                    modal_vertices = ModalVertices.load(f"{self.modalvertices_dir}/{filename}")
                    self.entity_manager.register('modal_vertices', modal_vertices)

        score_tracks = self.entity_manager.get('score_tracks')
        if len(score_tracks) == 0:
            if os.path.exists(self.scoretracks_dir):
                filenames = os.listdir(self.scoretracks_dir)
                for filename in filenames:
                    score_tracks = ScoreTrack.load(f"{self.scoretracks_dir_dir}/{filename}")
                    self.entity_manager.register('score_tracks', score_tracks)

    def prebake(self):
        _update_status(f"{self.status_dir}/prebake", 0)

        # Phase 1: Modal computation (parallel)
        all_objs = self.obj_dyn + self.obj_static
        with Pool(processes=self.num_workers) as pool:
            modal_func = partial(_prebake_modal, self.entity_manager)
            pool.map(modal_func, all_objs)
        _update_status(f"{self.status_dir}/prebake", 45)

        # Phase 2: Composer computation (parallel)
        collisions = self.entity_manager.get('collisions')
        collision_list = list(collisions.values())
        with Pool(processes=self.num_workers) as pool:
            composer_func = partial(_prebake_composer, self.entity_manager)
            pool.map(composer_func, collision_list)
        _update_status(f"{self.status_dir}/prebake", 90)

        # Save modal vertices and score tracks data
        modal_vertices = self.entity_manager.get('modal_vertices')
        print('Save modal_vertices: ', len(modal_vertices))
        for m_idx in modal_vertices.keys():
            modal_vertices[m_idx].save(f"{self.modalvertices_dir}/{m_idx:05d}.json")

        score_tracks = self.entity_manager.get('score_tracks')
        print('Save score_tracks: ', len(score_tracks))
        for s_idx in score_tracks.keys():
            score_tracks[s_idx].save(f"{self.scoretracks_dir}/{s_idx:05d}.pkl")

        _update_status(f"{self.status_dir}/prebake", 99)

    def bake(self):
        _update_status(f"{self.status_dir}/bake", 0)

        connected_buffer = ConnectedBuffer()
        self.entity_manager.register('connected_buffer', connected_buffer)
        sample_counter = SampleCounter(status_file=f"{self.status_dir}/bake")
        trajectories = self.entity_manager.get('trajectories')
        if len(trajectories) == 0: 
            if os.path.exists(self.trajectories_dir):
                filenames = os.listdir(self.trajectories_dir)
                for filename in filenames:
                    trajectory = TrajectoryData.load(f"{self.trajectories_dir}/{filename}")
                    if not trajectory.static:
                        break
        else:
            for t_idx in trajectories.keys():
                if not trajectories[t_idx].static:
                    trajectory = trajectories[t_idx]
                    break

        sample_counter.set_total_samples(int(trajectory.get_x()[-1]))
        self.entity_manager.register('sample_counter', sample_counter)

        # Phase 1: Luthier computation (parallel)
        all_objs = self.obj_dyn + self.obj_static
        with Pool(processes=self.num_workers) as pool:
            luthier_func = partial(_bake_luthier, self.entity_manager)
            pool.map(luthier_func, all_objs)
        _update_status(f"{self.status_dir}/bake", 10)

        # Phase 2: Player computation (parallel)
        players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in all_objs]
        with Pool(processes=self.num_workers) as pool:
            player_func = partial(_bake_player, self.entity_manager)
            pool.map(player_func, players)
        _update_status(f"{self.status_dir}/bake", 90)

        # Phase 3: Save players (parallel)
        print('rigidBodyEngine: Save player')
        with Pool(processes=self.num_workers) as pool:
            save_func = partial(_bake_save, self.entity_manager)
            pool.map(save_func, players)

        _update_status(f"{self.status_dir}/bake", 99)

