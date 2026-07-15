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
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from physicsSolver import EntityManager
from physicsSolver.lib.score_data import ScoreEvent, ScoreTrack

@dataclass
class ModalComposer:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

    def compute(self, collision: Any) -> None:
        print('ModalComposer compute: ', collision.type.value, collision.obj1_idx, collision.obj2_idx)
        if collision.type.value == 'connected' or not collision.valid:
            return
        config = self.entity_manager.get('config')
        forces_path = f"{config.system.cache_path}/audio_force"
        samples = collision.samples
        if samples.shape[0] == 0:
            return
        sample_start = samples[0]
        sample_stop = samples[-1] + 1

        obj1_idx = collision.obj1_idx
        obj2_idx = collision.obj2_idx

        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
                force1, coupling_strength1 = self._load_audioforce_tracks(forces_path=forces_path, obj_name=config_obj1.name)
            elif conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
                force2, coupling_strength2 = self._load_audioforce_tracks(forces_path=forces_path, obj_name=config_obj2.name)

        score_track1_final, score_track2_final, score_track1, score_track2 = ([] for _ in range(4))
        score_tracks = self.entity_manager.get('score_tracks')
        for idx in score_tracks.keys():
            if score_tracks[idx].obj_idx == obj1_idx and not score_tracks[idx].is_final:
                for event_idx in score_tracks[idx]:
                    if score_tracks[idx][event_idx].coll_obj == obj2_idx:
                        event_track1 = score_tracks[idx][event_idx].coll_obj
            elif score_tracks[idx].obj_idx == obj1_idx and score_tracks[idx].is_final:
                score_track1_final = score_tracks[idx]
            elif score_tracks[idx].obj_idx == obj2_idx and not score_tracks[idx].is_final:
                for event_idx in score_tracks[idx]:
                    if score_tracks[idx][event_idx].coll_obj == obj1_idx:
                        event_track2 = score_tracks[idx][event_idx].coll_obj
            elif score_tracks[idx].obj_idx == obj2_idx and score_tracks[idx].is_final:
                score_track2_final = score_tracks[idx]

        if score_track1_final == []:
            score_track1_final = ScoreTrack(obj_idx=obj1_idx, obj_name=config_obj1.name, is_final=True)
            _ = self.entity_manager.register('score_track', score_track1_final)
        if score_track2_final == []:
            score_track2_final = ScoreTrack(obj_idx=obj2_idx, obj_name=config_obj2.name, is_final=True)
            _ = self.entity_manager.register('score_track', score_track2_final)

        mixed_mask1 = event_track1.type == 5
        mixed_mask2 = event_track2.type == 5
        for force_type in range(1, 5):
            # init zeros array
            final_type1, final_type2 = (np.zeros_like(event_track1.type) for _ in range(2))
            final_vertex_ids1, final_vertex_ids2 = (np.zeros_like(event_track1.vertex_ids) for _ in range(2))
            final_contact_area1, final_contact_area2 = (np.zeros_like(event_track1.contact_area) for _ in range(2))
            final_force1, final_force2 = (np.zeros_like(coupling_strength1) for _ in range(2))
            final_coupling_data1, final_final_coupling_data2 = (np.zeros_like(coupling_strength1) for _ in range(2))

            # score_track1_final
            type_mask1 = event_track1.type == force_type
            n_vertex_ids1 = np.count_nonzero(event_track1.vertex_ids, axis=1)

            final_type1[type_mask1] = force_type
            final_coupling_data1[type_mask1] = coupling_strength1[type_mask1]
            final_contact_area1[type_mask1] = event_track1.contact_area[type_mask1]
            final_vertex_ids1[type_mask1] = event_track1.vertex_ids[type_mask1]
            final_force1[type_mask1] = np.divide(force1[force_type][type_mask1], n_vertex_ids1[type_mask1], out=np.zeros_like(force1[force_type][type_mask1]), where=n_vertex_ids1[type_mask1] != 0)

            if force_type in [2,3,4]:
                final_type1[mixed_mask1] = force_type
                final_coupling_data1[mixed_mask1] = coupling_strength1[mixed_mask1]
                final_contact_area1[mixed_mask1] = event_track1.contact_area[mixed_mask1]
                final_vertex_ids1[mixed_mask1] = event_track1.vertex_ids[mixed_mask1]
                final_force1[mixed_mask1] = np.divide(force1[force_type][mixed_mask1], n_vertex_ids1[mixed_mask1], out=np.zeros_like(force1[force_type][mixed_mask1]), where=n_vertex_ids1[mixed_mask1] != 0)

            score_track1_final.add_event(ScoreEvent(coll_obj=obj2_idx, vertex_ids=final_vertex_ids1, type=final_type1, contact_area=final_contact_area1 force=final_force1, coupling_data=final_coupling_data1))

            # score_track2_final
            type_mask2 = event_track2.type == force_type
            n_vertex_ids2 = np.count_nonzero(event_track2.vertex_ids, axis=1)

            final_type2[type_mask2] = force_type
            final_coupling_data2[type_mask2] = coupling_strength2[type_mask2]
            final_contact_area2[type_mask2] = event_track2.contact_area[type_mask2]
            final_vertex_ids2[type_mask2] = event_track2.vertex_ids[type_mask2]
            final_force2[type_mask2] = np.divide(force2[force_type][type_mask2], n_vertex_ids2[type_mask2], out=np.zeros_like(force2[force_type][type_mask2]), where=n_vertex_ids2[type_mask2] != 0)

            if force_type in [2,3,4]:
                final_type2[mixed_mask2] = force_type
                final_coupling_data2[mixed_mask2] = coupling_strength2[mixed_mask2]
                final_contact_area2[mixed_mask2] = event_track2.contact_area[mixed_mask2]
                final_vertex_ids1[mixed_mask2] = event_track2.vertex_ids[mixed_mask2]
                final_force2[mixed_mask2] = np.divide(force2[force_type][mixed_mask2], n_vertex_ids2[mixed_mask2], out=np.zeros_like(force2[force_type][mixed_mask2]), where=n_vertex_ids2[mixed_mask2] != 0)

            score_track2_final.add_event(ScoreEvent(coll_obj=obj1_idx, vertex_ids=final_vertex_ids2, type=final_type2, contact_area=event_track2.contact_area, force=final_force2, coupling_data=final_coupling_data2))

    def _load_audioforce_tracks(self, forces_path: str, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and list audio-force tracks for obj_name in forces_path"""

        non_collision = np.fromfile(f"{forces_path}/{obj_name}_non_collision.raw", dtype=np.float32)
        impact = np.fromfile(f"{forces_path}/{obj_name}_impact.raw", dtype=np.float32)
        scraping = np.fromfile(f"{forces_path}/{obj_name}_scraping.raw", dtype=np.float32)
        sliding = np.fromfile(f"{forces_path}/{obj_name}_sliding.raw", dtype=np.float32)
        rolling = np.fromfile(f"{forces_path}/{obj_name}_rolling.raw", dtype=np.float32)
        coupling_strength = np.fromfile(f"{forces_path}/{obj_name}_coupling_strength.raw", dtype=np.float32)

        return [non_collision, impact, scraping, sliding, rolling, _mixed, static], coupling_strength
