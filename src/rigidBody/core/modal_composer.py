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

from ..core.entity_manager import EntityManager

@dataclass
class ModalComposer:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.score_path = f"{config.system.cache_path}/score"
        os.makedirs(self.score_path, exist_ok=True)

    def compute(self, collision: Any) -> None:
        if collision.type.value == 'connected':
            return
        config = self.entity_manager.get('config')
        forces_path = f"{config.system.cache_path}/audio_force"
        samples = collision.samples
        sample_start = samples[0]
        sample_stop = samples[-1] + 1

        obj1_idx = collision.obj1_idx
        obj2_idx = collision.obj2_idx

        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
                force1, coupling_strength1 = self._load_audioforce_tracks(samples=samples, forces_path=forces_path, obj_name=config_obj1.name)
#                force1 = force1 / np.max(force1)
            if conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
                force2, coupling_strength2 = self._load_audioforce_tracks(samples=samples, forces_path=forces_path, obj_name=config_obj2.name)
#                force2 = force2 / np.max(force2)

        score_track1, score_track2 = ([] for _ in range(2))
        score_tracks = self.entity_manager.get('score_tracks')
        for idx in score_tracks.keys():
            if score_tracks[idx].obj_idx == obj1_idx:
                score_track1.append(score_tracks[idx])
            elif score_tracks[idx].obj_idx == obj2_idx:
                score_track2.append(score_tracks[idx])

        for sample_idx in range(sample_start, sample_stop):
            index = sample_idx - sample_start
            for score_idx in range(len(score_track1)):
                events = score_track1[score_idx].get_events_at_sample(sample_idx)
                for e_idx in range(len(events)):
                    force_type = int(events[e_idx].type)
                    force = np.divide(force1[force_type][index], events[e_idx].vertex_ids.shape[0], out=np.zeros_like(force1[force_type][index]), where=events[e_idx].vertex_ids.shape[0] != 0)
                    events[e_idx].force = float(force) if not np.isnan(force) else 0.0
                    coupling_data1 = coupling_strength1[index] if not np.isnan(coupling_strength1[index]) else 0.0
                    events[e_idx].coupling_data = np.array([[obj2_idx, coupling_data1]])
            for score_idx in range(len(score_track2)):
                events = score_track2[score_idx].get_events_at_sample(sample_idx)
                for e_idx in range(len(events)):
                    force_type = int(events[e_idx].type)
                    force = np.divide(force2[force_type][index], events[e_idx].vertex_ids.shape[0], out=np.zeros_like(force2[force_type][index]), where=events[e_idx].vertex_ids.shape[0] != 0)
                    events[e_idx].force = float(force) if not np.isnan(force) else 0.0
                    coupling_data2 = coupling_strength2[index] if not np.isnan(coupling_strength2[index]) else 0.0
                    events[e_idx].coupling_data = np.array([[obj1_idx, coupling_data2]])

    def _load_audioforce_tracks(self, samples: np.ndarray, forces_path: str, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and sum audio-force tracks for obj_name in forces_path"""
        sample_start = samples[0]
        sample_stop = samples[-1] + 1

        _no_force = np.zeros(sample_stop - sample_start)
        impact = np.zeros(sample_stop - sample_start)
        scraping = np.zeros(sample_stop - sample_start)
        sliding = np.zeros(sample_stop - sample_start)
        rolling = np.zeros(sample_stop - sample_start)
        non_collision = np.zeros(sample_stop - sample_start)
        coupling_strength = np.zeros(sample_stop - sample_start)

        if os.path.exists(f"{forces_path}/{obj_name}_impact.raw"):
            impact += np.fromfile(f"{forces_path}/{obj_name}_impact.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_scraping.raw"):
            scraping += np.fromfile(f"{forces_path}/{obj_name}_scraping.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_sliding.raw"):
            sliding += np.fromfile(f"{forces_path}/{obj_name}_sliding.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_rolling.raw"):
            rolling += np.fromfile(f"{forces_path}/{obj_name}_rolling.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_non_collision.raw"):
            non_collision += np.fromfile(f"{forces_path}/{obj_name}_non_collision.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_coupling_strength.raw"):
            coupling_strength += np.fromfile(f"{forces_path}/{obj_name}_coupling_strength.raw", dtype=np.float32)[sample_start:sample_stop]

        return [_no_force, impact, scraping, sliding, rolling, non_collision], coupling_strength
