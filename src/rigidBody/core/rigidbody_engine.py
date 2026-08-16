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
#dask_config.set(num_workers=1024)
dask_config.set({'num_workers': 1024, 'optimization.fuse.active': True, 'optimization.fuse.max_depth': 10,})

from pbrAudioCommon import EntityManager, ScoreTrack
from pbrAudioCommon import _update_status
from physicsSolver import ForceDataSequence, ModalVertices, CollisionData, TrajectoryData
from ellipsoidalProxy import Modal4Proxy, ProxySynth, ProxyEngine
from postProcess import PostProcessEngine

from ..core.mesh2modal import Mesh2Modal
from ..core.modal_composer import ModalComposer
from ..core.modal_luthier import ModalLuthier
from ..core.modal_player import ModalPlayer
from ..lib.sample_counter import SampleCounter
from ..lib.connected_buffer import ConnectedBuffer

@dataclass
class rigidBodyEngine:
    entity_manager: EntityManager
    obj_dyn: List[int] = field(default_factory=list)
    obj_static: List[int] = field(default_factory=list)
    obj_pairs: List[int] = field(default_factory=list)
    obj_modal: List[int] = field(default_factory=list)
    obj_proxy_synth: List[int] = field(default_factory=list)

    def __post_init__(self):
        from physicsSolver import ForceDataSequence, ModalVertices, CollisionData, TrajectoryData

        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        self.trajectories_dir = f"{config.system.cache_path}/trajectories"
        self.forces_dir = f"{config.system.cache_path}/forces_data"
        self.modalvertices_dir = f"{config.system.cache_path}/modalvertices"
        self.scoretracks_dir = f"{config.system.cache_path}/scoretracks"
        self.progress = 0

        # Ensure status directory exists
        os.makedirs(self.status_dir, exist_ok=True)

        obj_static, obj_dyn, obj_pairs, obj_modal, obj_proxy_synth = ([] for _ in range(5))
        for config_obj in config.objects:
            if not config_obj.static and not config_obj.idx in obj_dyn:
                self.obj_dyn.append(config_obj.idx)
            if config_obj.static and not config_obj.idx in obj_static:
                self.obj_static.append(config_obj.idx)
            if config_obj.proxy_type is False or config_obj.proxy_type in [3,4,5,6]:
                self.obj_modal.append(config_obj.idx)
            elif config.system.enable_proxy_synth and config_obj.proxy_type in [0,1,2]:
                self.obj_proxy_synth.append(config_obj.idx)
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                self.obj_pairs.append([config.objects[i].idx, config.objects[j].idx])

        trajectories = self.entity_manager.get('trajectories')
        if len(trajectories) == 0:
            if os.path.exists(f"{self.trajectories_dir}") and not len(os.listdir(f"{self.trajectories_dir}")) == 0:
                for filename in os.listdir(f"{self.trajectories_dir}"):
                    trajectory = None
                    if filename.endswith('.pkl') and os.path.isfile(f"{self.trajectories_dir}/corrected/{filename}"):
                        trajectory = TrajectoryData.load(f"{self.trajectories_dir}/corrected/{filename}")
                    elif filename.endswith('.pkl') and not os.path.isfile(f"{self.trajectories_dir}/corrected/{filename}"):
                        trajectory = TrajectoryData.load(f"{self.trajectories_dir}/{filename}")
                    if trajectory is not None:
                        _ = self.entity_manager.register('trajectories', trajectory)
#                        self.entity_manager.register('trajectories', trajectories, trajectories_idx)
#                        trajectories_idx += 1

        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if not trajectories[t_idx].static:
                trajectory = trajectories[t_idx]
                self.total_samples = int(trajectory.get_x()[-1])
                break

        collisions = self.entity_manager.get('collisions')
        if len(collisions) == 0:
            if os.path.exists(f"{self.collisions_dir}") and not len(os.listdir(f"{self.collisions_dir}")) == 0:
                for filename in os.listdir(f"{self.collisions_dir}"):
                    if filename.endswith('.pkl'):
#                        idx = int(filename.removesuffix('.pkl'))
                        collisions = CollisionData.load(f"{self.collisions_dir}/{filename}")
#                        self.entity_manager.register('collisions', collisions, idx)
                        _ = self.entity_manager.register('collisions', collisions)

        forces = self.entity_manager.get('forces')
        if len(forces) == 0:
            if os.path.exists(f"{self.forces_dir}") and not len(os.listdir(f"{self.forces_dir}")) == 0:
#                forces_idx = 0
                for filename in os.listdir(f"{self.forces_dir}"):
                    if filename.endswith('.pkl'):
                        forces = ForceDataSequence.load(f"{self.forces_dir}/{filename}")
                        _ = self.entity_manager.register('forces', forces)
#                        self.entity_manager.register('forces', forces, forces_idx)
#                        forces_idx += 1
            forces = self.entity_manager.get('forces')

        modal_vertices = self.entity_manager.get('modal_vertices')
        if len(modal_vertices) == 0:
            if os.path.exists(self.modalvertices_dir):
                filenames = os.listdir(self.modalvertices_dir)
#                modalvertices_idx = 0
                for filename in filenames:
                    if os.path.isfile(f"{self.modalvertices_dir}/{filename}"):
                        modal_vertices = ModalVertices.load(f"{self.modalvertices_dir}/{filename}")
                        _ = self.entity_manager.register('modal_vertices', modal_vertices)
#                    self.entity_manager.register('modal_vertices', modal_vertices, modalvertices_idx)
#                    modalvertices_idx += 1

    def prebake(self):
        self.progress = _update_status(f"{self.status_dir}/prebake", 0)

        score_tracks = self.entity_manager.get('score_tracks')
        if len(score_tracks) == 0:
            if os.path.exists(self.scoretracks_dir):
                filenames = os.listdir(self.scoretracks_dir)
#                scoretracks_idx = 0
                for filename in filenames:
                    if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                        score_tracks = ScoreTrack.load(f"{self.scoretracks_dir}/{filename}")
                        _ = self.entity_manager.register('score_tracks', score_tracks)
                for filename in filenames:
                    if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                        os.remove(f"{self.scoretracks_dir}/{filename}")
#                    self.entity_manager.register('score_tracks', score_tracks, scoretracks_idx)
#                    scoretracks_idx += 1

        tasks_modal = [self.prebake_modal(obj_idx) for obj_idx in self.obj_modal]
        results_modal = compute(*tasks_modal)
        self.progress = _update_status(f"{self.status_dir}/prebake", 30)

        tasks_proxy = [self.prebake_proxy(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_proxy = compute(*tasks_proxy)
        self.progress = _update_status(f"{self.status_dir}/prebake", 45)

        # Init per object final score track
        config = self.entity_manager.get('config')
        for config_obj in config.objects:
            score_track_final = ScoreTrack(obj_idx=config_obj.idx, obj_name=config_obj.name, is_final=True, total_samples=self.total_samples)
            _ = self.entity_manager.register('score_tracks', score_track_final)

        collisions = self.entity_manager.get('collisions')
        tasks_composer = [self.prebake_composer(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_composer = compute(*tasks_composer)
        self.progress = _update_status(f"{self.status_dir}/prebake", 90)

        # Save modal vertices and score tracks data
        modal_vertices = self.entity_manager.get('modal_vertices')
        print('Save modal_vertices: ', len(modal_vertices))
#        for m_idx in modal_vertices.keys():
#            modal_vertices[m_idx].save(f"{self.modalvertices_dir}/{m_idx:05d}.json")
        tasks_save_modal_vertices = [self.save_modal_vertices(modal_vertices[m_idx], f"{m_idx:05d}.json") for m_idx in modal_vertices.keys()]
        results_save_modal_vertices = compute(*tasks_save_modal_vertices)

        self.progress = _update_status(f"{self.status_dir}/prebake", 95)

        score_tracks = self.entity_manager.get('score_tracks')
        n_score = 0
        for s_idx in score_tracks.keys():
            if score_tracks[s_idx].is_final:
                score_tracks[s_idx].save(f"{self.scoretracks_dir}/{s_idx:05d}.tar.gz")
                n_score += 1
        print('Saved final score_tracks: ', n_score)
#        tasks_save_score_tracks = [self.save_score_tracks(score_tracks[s_idx], f"{s_idx:05d}.tar.gz") for s_idx in score_tracks.keys()]
#        results_save_score_tracks = compute(*tasks_save_score_tracks)

        self.progress = _update_status(f"{self.status_dir}/prebake", 99)

    def bake(self):
        self.progress = _update_status(f"{self.status_dir}/bake", 0)

        score_tracks = self.entity_manager.get('score_tracks')
        if len(score_tracks) == 0:
            if os.path.exists(self.scoretracks_dir):
                filenames = os.listdir(self.scoretracks_dir)
#                scoretracks_idx = 0
                for filename in filenames:
                    if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                        score_tracks = ScoreTrack.load(f"{self.scoretracks_dir}/{filename}", final=True)
                        _ = self.entity_manager.register('score_tracks', score_tracks)
#                    self.entity_manager.register('score_tracks', score_tracks, scoretracks_idx)
#                    scoretracks_idx += 1

        connected_buffer = ConnectedBuffer()
        _ = self.entity_manager.register('connected_buffer', connected_buffer)
        sample_counter = SampleCounter(status_file=f"{self.status_dir}/bake")
        sample_counter.set_total_samples(self.total_samples)
        _ = self.entity_manager.register('sample_counter', sample_counter)

        tasks_luthier = [self.bake_luthier(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_luthier = compute(*tasks_luthier)
        self.progress = _update_status(f"{self.status_dir}/bake", 10)

#        self.players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
#        tasks_player = [self.bake_player(player) for player in self.players]
#        tasks_save = [self.bake_save(player) for player in self.players]
        modal_obj_idx = list(set(self.obj_dyn + self.obj_static) - set(self.obj_proxy_synth))
        players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in modal_obj_idx]
        tasks_player = [self.bake_player(player) for player in players]
        results_player = compute(*tasks_player)

        self.progress = _update_status(f"{self.status_dir}/bake", 60)

#        # ProxySynth
#        if not len(self.obj_proxy_synth) == 0:
#            tasks_proxy_synth = [self.bake_proxy_synth(obj_idx) for obj_idx in self.obj_proxy_synth]
#            results_proxy_synth = compute(*tasks_proxy_synth)

        # ProxySynth
        if not len(self.obj_proxy_synth) == 0:
            proxy_engine = ProxyEngine(self.entity_manager)
            tasks_proxy_synth = [proxy_engine.compute(obj_idx, self.total_samples) for obj_idx in self.obj_proxy_synth]
            results_proxy_synth = compute(*tasks_proxy_synth)

        self.progress = _update_status(f"{self.status_dir}/bake", 90)

        print('rigidBodyEngine: Save player')
        tasks_save = [self.bake_save(player) for player in players]
        results_save = compute(*tasks_save)
        self.progress = _update_status(f"{self.status_dir}/bake", 92)

        post_engine = PostProcessEngine(self.entity_manager)
        post_engine.process_with_modal_player()

        self.progress = _update_status(f"{self.status_dir}/bake", 99)

    @delayed
    def prebake_modal(self, obj_idx: int):
        mm = Mesh2Modal(self.entity_manager)
        mm.compute(obj_idx)

    @delayed
    def prebake_proxy(self, obj_idx: int):
        mp = Modal4Proxy(self.entity_manager)
        mp.compute(obj_idx)

    @delayed
    def prebake_composer(self, collision: CollisionData):
        mc = ModalComposer(self.entity_manager)
        mc.compute(collision)

    @delayed
    def save_modal_vertices(self, modal_vertices: Any, filename: str):
        modal_vertices.save(f"{self.modalvertices_dir}/{filename}")

    @delayed
    def save_score_tracks(self, score_track: Any, filename: str):
        score_track.save(f"{self.scoretracks_dir}/{filename}")

    @delayed
    def bake_luthier(self, obj_idx: int):
        ml = ModalLuthier(self.entity_manager)
        ml.compute(obj_idx)

    @delayed
    def bake_player(self, player: Any):
        player.compute()

#    @delayed
#    def bake_proxy_synth(self, obj_idx: int):
#        ps = ProxySynth(self.entity_manager)
#        ps.compute(obj_idx, self.total_samples)

    @delayed
    def bake_save(self, player: Any):
        player.save_synth_tracks()
