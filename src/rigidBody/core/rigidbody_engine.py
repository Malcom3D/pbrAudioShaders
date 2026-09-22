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

from pbrAudioCommon import EntityManager, ScoreTrack, ForceDataSequence, ModalVertices, CollisionData, ResumeData
from pbrAudioCommon import _update_status
from pbrAudioCommon import TrajectoryData

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
    total_samples: int = 1

    def __post_init__(self):
        resume_data = ResumeData(self.entity_manager)
        resume_data.load_data()

        config = self.entity_manager.get('config')
        self.physical_core = config.system.physical_core
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
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
            if config_obj.proxy_type is False or config_obj.proxy_type in [3,4,5]:
                self.obj_modal.append(config_obj.idx)
            elif config.system.enable_proxy_synth and config_obj.proxy_type in [0,1,2]:
                self.obj_proxy_synth.append(config_obj.idx)
        for i in range(len(config.objects)):
            for j in range(i + 1, len(config.objects)):
                self.obj_pairs.append([config.objects[i].idx, config.objects[j].idx])

        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if isinstance(trajectories[t_idx], TrajectoryData) and not trajectories[t_idx].static:
                trajectory = trajectories[t_idx]
                self.total_samples = int(trajectory.get_x()[-1])
                break

    def prebake(self):
        if os.path.exists(f"{self.status_dir}/step_done"):
            with open(f"{self.status_dir}/step_done", 'r') as file:
                step_done = file.read().split()
    
            if '_modal' not in step_done:
                self._modal()
                self._save()
            if '_proxy' not in step_done:
                self._proxy()
                self._save()

    def _modal(self):
        tasks_modal = [self.prebake_modal(obj_idx) for obj_idx in self.obj_modal]
        results_modal = compute(*tasks_modal)
        self.progress = _update_status(f"{self.status_dir}", "/prebake", 30)

    def _proxy(self):
        tasks_proxy = [self.prebake_proxy(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_proxy = compute(*tasks_proxy)
        self.progress = _update_status(f"{self.status_dir}", "/prebake", 45)

    def _proxy(self):
        # Init per object final score track
        config = self.entity_manager.get('config')
        for config_obj in config.objects:
            score_track_final = ScoreTrack(obj_idx=config_obj.idx, obj_name=config_obj.name, is_final=True, total_samples=self.total_samples)
            _ = self.entity_manager.register('score_tracks', score_track_final)

        collisions = self.entity_manager.get('collisions')
        tasks_composer = [self.prebake_composer(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_composer = compute(*tasks_composer)
        self.progress = _update_status(f"{self.status_dir}", "/prebake", 90)

    def _save(self):
        # Save modal vertices data
        modal_vertices = self.entity_manager.get('modal_vertices')
        print('Save modal_vertices: ', len(modal_vertices))
        tasks_save_modal_vertices = [self.save_modal_vertices(modal_vertices[m_idx], f"{m_idx:05d}.json") for m_idx in modal_vertices.keys()]
        results_save_modal_vertices = compute(*tasks_save_modal_vertices)

        # Save score tracks data in /tmp
        score_tracks = self.entity_manager.get('score_tracks')
        n_score = []
        for s_idx in score_tracks.keys():
            if score_tracks[s_idx].is_final:
                score_tracks[s_idx].save(f"/tmp/{s_idx:05d}.tar.gz")
                n_score += f"/tmp/{s_idx:05d}.tar.gz"

        # Clean score tracks data
        if os.path.exists(self.scoretracks_dir):
            filenames = os.listdir(self.scoretracks_dir)
            for filename in filenames:
                if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                    os.remove(f"{self.scoretracks_dir}/{filename}")

        # Move score tracks data files from /tmp
        for filename in n_score:
            os.replace(filename,f"{self.scoretracks_dir}/{filename.removeprefix('/tmp/')}")
        print('Saved final score_tracks: ', n_score)

        self.progress = _update_status(f"{self.status_dir}", "/prebake", 99)

    def bake(self):
        step_done = []
        if os.path.exists(f"{self.status_dir}/step_done"):
            with open(f"{self.status_dir}/step_done", 'r') as file:
                step_done = file.read().split()

        self._connected_buffer()
        if '_modal_synth' not in step_done:
            self._modal_synth()
        if '_proxy_synth' not in step_done:
            self._proxy_synth()
        if '_post_process' not in step_done:
            self._post_process()

    def _modal_synth(self):
        modal_dyn_idx = list(set(self.obj_dyn) - set(self.obj_proxy_synth))
        step_done = []
        if os.path.exists(f"{self.status_dir}/step_done"):
            with open(f"{self.status_dir}/step_done", 'r') as file:
                step_done = file.read().split()

        if len(modal_dyn_idx) >= self.physical_core:
            if '_process_group' not in step_done:
                self._process_groups()
        else:
            if '_luthier' not in step_done:
                self._luthier()
            if '_player' not in step_done:
                self._player()
            if '_save' not in step_done:
                self._save()

    def _process_groups(self):
        print('rigidBodyEngine: Warning: cpu core are less than non-static objects.')
        print('rigidBodyEngine: Warning: fallback to groups synthesis: some events can be lost.')
        modal_dyn_idx = list(set(self.obj_dyn) - set(self.obj_proxy_synth))
        for x in range(self.physical_core):
            modal_groups = [modal_dyn_idx[i:i + self.physical_core] for i in range(0, len(modal_dyn_idx), self.physical_core)]
        players = []
        for modal_group in modal_groups:
            self._luthier_group()
            self._player_group(modal_group)
            self._reset_group()
        self.progress = _update_status(f"{self.status_dir}", "/bake", 92)

    def _connected_buffer(self):
        connected_buffer = ConnectedBuffer()
        _ = self.entity_manager.register('connected_buffer', connected_buffer)
        sample_counter = SampleCounter(status_dir=f"{self.status_dir}")
        sample_counter.set_total_samples(self.total_samples)
        _ = self.entity_manager.register('sample_counter', sample_counter)

    def _luthier_group(self):
        tasks_luthier = [self.bake_luthier(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_luthier = compute(*tasks_luthier)

    def _player_group(self, modal_group: List[int]):
        modal_static_idx = list(set(self.obj_static) - set(self.obj_proxy_synth))
        group_players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in modal_group]
        group_players += [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in modal_static_idx]
        tasks_player = [self.bake_player(group_player) for group_player in group_players]
        results_player = compute(*tasks_player)

        print('rigidBodyEngine: Save player')
        tasks_save = [self.bake_save(group_player) for group_player in group_players]
        results_save = compute(*tasks_save)

    def _reset_group(self):
        self.entity_manager.unregister('sample_counter')
        self.entity_manager.unregister('connected_buffer')

        connected_buffer = ConnectedBuffer()
        _ = self.entity_manager.register('connected_buffer', connected_buffer)
        sample_counter = SampleCounter(status_file=f"{self.status_dir}/bake")
        sample_counter.set_total_samples(self.total_samples)
        _ = self.entity_manager.register('sample_counter', sample_counter)

    def _luthier(self):
        tasks_luthier = [self.bake_luthier(obj_idx) for obj_idx in self.obj_dyn + self.obj_static]
        results_luthier = compute(*tasks_luthier)
        self.progress = _update_status(f"{self.status_dir}", "/bake", 10)

    def _player(self):
        modal_obj_idx = list(set(self.obj_dyn + self.obj_static) - set(self.obj_proxy_synth))
        players = [ModalPlayer(self.entity_manager, obj_idx) for obj_idx in modal_obj_idx]
        tasks_player = [self.bake_player(player) for player in players]
        results_player = compute(*tasks_player)
        self.progress = _update_status(f"{self.status_dir}", "/bake", 60)

    def _save(self):
        print('rigidBodyEngine: Save player')
        tasks_save = [self.bake_save(player) for player in players]
        results_save = compute(*tasks_save)
        self.progress = _update_status(f"{self.status_dir}", "/bake", 92)

    def _proxy_synth(self):
        # ProxySynth
        if not len(self.obj_proxy_synth) == 0:
            proxy_engine = ProxyEngine(self.entity_manager)
            tasks_proxy_synth = [proxy_engine.compute(obj_idx, self.total_samples) for obj_idx in self.obj_proxy_synth]
            results_proxy_synth = compute(*tasks_proxy_synth)

        self.progress = _update_status(f"{self.status_dir}", "/bake", 90)

    def _post_process(self):
        post_engine = PostProcessEngine(self.entity_manager)
        post_engine.process_with_modal_player()

        self.progress = _update_status(f"{self.status_dir}", "/bake", 99)

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
    def bake_luthier(self, obj_idx: int):
        ml = ModalLuthier(self.entity_manager)
        ml.compute(obj_idx)

    @delayed
    def bake_player(self, player: Any):
        player.compute()

    @delayed
    def bake_save(self, player: Any):
        player.save_synth_tracks()
