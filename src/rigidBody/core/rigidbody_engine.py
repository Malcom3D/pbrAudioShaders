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

# Configure Dask to use more threads
from dask import config as dask_config
#dask_config.set(scheduler='threads', num_workers=1024)
dask_config.set(num_workers=1024)

from physicsSolver import EntityManager, ForceDataSequence, ModalVertices, ScoreTrack, CollisionData, TrajectoryData
from ..core.mesh2modal import Mesh2Modal
from ..core.modal_composer import ModalComposer
from ..core.modal_luthier import ModalLuthier
from ..core.modal_player import ModalPlayer

from ..lib.sample_counter import SampleCounter
from ..lib.connected_buffer import ConnectedBuffer

from ..lib.functions import _update_status

@dataclass
class rigidBodyEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)
    obj_static: List[int] = field(default_factory=list)
    obj_pairs: List[int] = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        self.trajectories_dir = f"{config.system.cache_path}/trajectories"
        self.forces_dir = f"{config.system.cache_path}/forces_data"
        self.modalvertices_dir = f"{config.system.cache_path}/modalvertices"
        self.scoretracks_dir = f"{config.system.cache_path}/scoretracks"

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
#                trajectories_idx = 0
                for filename in os.listdir(f"{self.trajectories_dir}"):
                    if filename.endswith('.pkl'):
                        trajectories = TrajectoryData.load(f"{self.trajectories_dir}/{filename}")
                        self.entity_manager.register('trajectories', trajectories)
#                        self.entity_manager.register('trajectories', trajectories, trajectories_idx)
#                        trajectories_idx += 1

        collisions = self.entity_manager.get('collisions')
        if len(collisions) == 0:
            if os.path.exists(f"{self.collisions_dir}") and not len(os.listdir(f"{self.collisions_dir}")) == 0:
                for filename in os.listdir(f"{self.collisions_dir}"):
                    if filename.endswith('.pkl'):
#                        idx = int(filename.removesuffix('.pkl'))
                        collisions = CollisionData.load(f"{self.collisions_dir}/{filename}")
#                        self.entity_manager.register('collisions', collisions, idx)
                        self.entity_manager.register('collisions', collisions)

        forces = self.entity_manager.get('forces')
        if len(forces) == 0:
            if os.path.exists(f"{self.forces_dir}") and not len(os.listdir(f"{self.forces_dir}")) == 0:
#                forces_idx = 0
                for filename in os.listdir(f"{self.forces_dir}"):
                    if filename.endswith('.pkl'):
                        forces = ForceDataSequence.load(f"{self.forces_dir}/{filename}")
                        self.entity_manager.register('forces', forces)
#                        self.entity_manager.register('forces', forces, forces_idx)
#                        forces_idx += 1
            forces = self.entity_manager.get('forces')

        modal_vertices = self.entity_manager.get('modal_vertices')
        if len(modal_vertices) == 0:
            if os.path.exists(self.modalvertices_dir):
                filenames = os.listdir(self.modalvertices_dir)
#                modalvertices_idx = 0
                for filename in filenames:
                    modal_vertices = ModalVertices.load(f"{self.modalvertices_dir}/{filename}")
                    self.entity_manager.register('modal_vertices', modal_vertices)
#                    self.entity_manager.register('modal_vertices', modal_vertices, modalvertices_idx)
#                    modalvertices_idx += 1

        score_tracks = self.entity_manager.get('score_tracks')
        if len(score_tracks) == 0:
            if os.path.exists(self.scoretracks_dir):
                filenames = os.listdir(self.scoretracks_dir)
#                scoretracks_idx = 0
                for filename in filenames:
                    score_tracks = ScoreTrack.load(f"{self.scoretracks_dir}/{filename}")
                    self.entity_manager.register('score_tracks', score_tracks)
#                    self.entity_manager.register('score_tracks', score_tracks, scoretracks_idx)
#                    scoretracks_idx += 1

    def prebake(self):
        _update_status(f"{self.status_dir}/prebake", 0)

        tasks_modal = [self.prebake_modal(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_modal = compute(*tasks_modal)
        _update_status(f"{self.status_dir}/prebake", 45)

        collisions = self.entity_manager.get('collisions')
        tasks_composer = [self.prebake_composer(collisions[collision_idx]) for collision_idx in collisions.keys()]
        results_composer = compute(*tasks_composer)
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

        tasks_luthier = [self.bake_luthier(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_luthier = compute(*tasks_luthier)
        _update_status(f"{self.status_dir}/bake", 10)

#        self.players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
#        tasks_player = [self.bake_player(player) for player in self.players]
        players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        tasks_player = [self.bake_player(player) for player in players]
        results_player = compute(*tasks_player)
        _update_status(f"{self.status_dir}/bake", 90)

        print('rigidBodyEngine: Save player')
#        tasks_save = [self.bake_save(player) for player in self.players]
        tasks_save = [self.bake_save(player) for player in players]
        results_save = compute(*tasks_save)

        _update_status(f"{self.status_dir}/bake", 99)

    @delayed
    def prebake_modal(self, obj_idx: int):
        mm = Mesh2Modal(self.entity_manager)
        mm.compute(obj_idx)

    @delayed
    def prebake_composer(self, collision: CollisionData):
        mc = ModalComposer(self.entity_manager)
        mc.compute(collision)

    @delayed
    def bake_luthier(self, obj_idx: int):
        ml = ModalLuthier(self.entity_manager)
        ml.compute(obj_idx)

    @delayed
    def bake_player(self, player: Any):
        player.compute()

    @delayed
    def bake_save(self, player: Any):
        player.save_synth_tracks()
