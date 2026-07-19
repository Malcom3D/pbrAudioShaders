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
import blosc2
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from physicsSolver import EntityManager
from physicsSolver.lib.score_data import ScoreEvent, ScoreTrack

@dataclass
class ModalComposer:
    entity_manager: EntityManager

#    def __post_init__(self):
#        config = self.entity_manager.get('config')

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        forces_path = f"{config.system.cache_path}/audio_force"

        score_track_final, score_track = (None for _ in range(2))
        score_tracks = self.entity_manager.get('score_tracks')
        for idx in score_tracks.keys():
            if score_tracks[idx].obj_idx == obj_idx and not score_tracks[idx].is_final:
                score_track = score_tracks[idx]
            elif score_tracks[idx].obj_idx == obj_idx and score_tracks[idx].is_final:
                score_track_final = score_tracks[idx]

        if score_track is None:
            # object do not collide (static?)
            return

        total_samples = score_track_final.total_samples

        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                force, coupling_strength = self._load_audioforce_tracks(total_samples=total_samples, forces_path=forces_path, obj_name=config_obj.name)

        for event_track in score_track.events:
            obj2_idx = event_track.coll_obj

            mixed_mask = event_track.type == 5
            for event_type in range(1, 5):
                # init zeros array
                final_type = np.zeros_like(event_track.type)
                final_vertex_ids = np.zeros_like(event_track.vertex_ids)
                final_vertex_ids = blosc2.asarray(final_vertex_ids)
                final_contact_area = np.zeros_like(event_track.contact_area)
                final_force = np.zeros_like(coupling_strength)
                final_coupling_data = np.zeros_like(coupling_strength)

                # score_track_final
                type_mask = event_track.type == event_type
                n_vertex_ids = np.count_nonzero(event_track.vertex_ids, axis=1)

                final_type[type_mask] = event_type
                final_coupling_data[type_mask] = coupling_strength[type_mask]
                final_contact_area[type_mask] = event_track.contact_area[type_mask]
                final_vertex_ids[type_mask.reshape(-1,)] = event_track.vertex_ids[type_mask.reshape(-1,)]
                final_force[type_mask] = np.divide(force[event_type][type_mask], n_vertex_ids[type_mask.reshape(-1,)], out=np.zeros_like(force[event_type][type_mask]), where=n_vertex_ids[type_mask.reshape(-1,)] != 0)

                # Rewrite final*[mixed_mask] with type == 2
                if event_type == 2:
                    final_type[mixed_mask] = event_type
                    final_coupling_data[mixed_mask] = coupling_strength[mixed_mask]
                    final_contact_area[mixed_mask] = event_track.contact_area[mixed_mask]
                    final_vertex_ids[mixed_mask.reshape(-1,)] = event_track.vertex_ids[mixed_mask.reshape(-1,)]
                    final_force[mixed_mask] = np.divide(force[event_type][mixed_mask], n_vertex_ids[mixed_mask.reshape(-1,)], out=np.zeros_like(force[event_type][mixed_mask]), where=n_vertex_ids[mixed_mask.reshape(-1,)] != 0)

            score_track_final.add_event(ScoreEvent(coll_obj=obj2_idx, type=final_type, vertex_ids=final_vertex_ids, contact_area=final_contact_area, force=final_force, coupling_data=final_coupling_data))

            # Add event type 3 and 4 to complete mixed event
            for event_type in [3,4]:
                # init zeros array
                final_type = np.zeros_like(event_track.type)
                final_vertex_ids = np.zeros_like(event_track.vertex_ids)
                final_vertex_ids = blosc2.asarray(final_vertex_ids)
                final_contact_area = np.zeros_like(event_track.contact_area)
                final_force = np.zeros_like(coupling_strength)
                final_coupling_data = np.zeros_like(coupling_strength)

                final_type[mixed_mask] = event_type
                final_coupling_data[mixed_mask] = coupling_strength[mixed_mask]
                final_contact_area[mixed_mask] = event_track.contact_area[mixed_mask]
                final_vertex_ids[mixed_mask.reshape(-1,)] = event_track.vertex_ids[mixed_mask.reshape(-1,)]
                final_force[mixed_mask] = np.divide(force[event_type][mixed_mask], n_vertex_ids[mixed_mask.reshape(-1,)], out=np.zeros_like(force[event_type][mixed_mask]), where=n_vertex_ids[mixed_mask.reshape(-1,)] != 0)

                score_track_final.add_event(ScoreEvent(coll_obj=obj2_idx, type=final_type, vertex_ids=final_vertex_ids, contact_area=final_contact_area, force=final_force, coupling_data=final_coupling_data))

    def _load_audioforce_tracks(self, total_samples: int, forces_path: str, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and list audio-force tracks for obj_name in forces_path"""

        non_collision = np.zeros((total_samples,1))
        impact = np.zeros((total_samples,1))
        scraping = np.zeros((total_samples,1))
        sliding = np.zeros((total_samples,1))
        rolling = np.zeros((total_samples,1))
        _mixed = np.zeros((total_samples,1))
        static = np.zeros((total_samples,1))
        coupling_strength = np.zeros((total_samples,1))

        if os.path.exists(f"{forces_path}/{obj_name}_non_collision.raw"):
            non_collision += np.fromfile(f"{forces_path}/{obj_name}_non_collision.raw", dtype=np.float32).reshape((-1,1))
        if os.path.exists(f"{forces_path}/{obj_name}_impact.raw"):
            impact += np.fromfile(f"{forces_path}/{obj_name}_impact.raw", dtype=np.float32).reshape((-1,1))
        if os.path.exists(f"{forces_path}/{obj_name}_scraping.raw"):
            scraping += np.fromfile(f"{forces_path}/{obj_name}_scraping.raw", dtype=np.float32).reshape((-1,1))
        if os.path.exists(f"{forces_path}/{obj_name}_sliding.raw"):
            sliding += np.fromfile(f"{forces_path}/{obj_name}_sliding.raw", dtype=np.float32).reshape((-1,1))
        if os.path.exists(f"{forces_path}/{obj_name}_rolling.raw"):
            rolling += np.fromfile(f"{forces_path}/{obj_name}_rolling.raw", dtype=np.float32).reshape((-1,1))
        if os.path.exists(f"{forces_path}/{obj_name}_coupling_strength.raw"):
            coupling_strength += np.fromfile(f"{forces_path}/{obj_name}_coupling_strength.raw", dtype=np.float32).reshape((-1,1))

        return [non_collision, impact, scraping, sliding, rolling, _mixed, static], coupling_strength
