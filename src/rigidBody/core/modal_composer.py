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
        config = self.entity_manager.get('config')
        sample_counter = self.entity_manager.get('sample_counter')
        forces_path = f"{config.system.cache_path}/audio_force"
        collision_area = collision.collision_area
        samples = collision_area[0]
        sample_start = samples[0]
        sample_stop = samples[-1] + 1

        obj1_idx = collision.obj1_idx
        vertex1_ids = collision_area[1]
        num_vertex1_ids = collision_area[2]

        obj2_idx = collision.obj2_idx
        vertex2_ids = collision_area[4]
        num_vertex2_ids = collision_area[5]

        trajectories = self.entity_manager.get('trajectories')

        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
                force1, coupling_strenght1 = self._load_audioforce_tracks(samples=samples, forces_path=forces_path, obj_name=config_obj1.name)
                force1 = np.divide(force1, num_vertex1_ids, out=np.zeros_like(force1), where=num_vertex1_ids != 0)
            if conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
                force2, coupling_strenght2 = self._load_audioforce_tracks(samples=samples, forces_path=forces_path, obj_name=config_obj1.name)
                force2 = np.divide(force2, num_vertex2_ids, out=np.zeros_like(force2), where=num_vertex2_ids != 0)
        if not config_obj1.static or not config_obj2.static:
            for t_idx in trajectories.keys():
                if trajectories[t_idx].obj_idx == obj1_idx:
                    self.trajectory1 = trajectories[t_idx]
                if trajectories[t_idx].obj_idx == obj2_idx:
                    self.trajectory2 = trajectories[t_idx]

        total_samples = int(self.trajectory1.get_x()[-1])
        sample_counter.total_samples = total_samples

        score_obj1 = np.zeros((total_samples, 3), dtype=object)
        score_obj2 = np.zeros((total_samples, 3), dtype=object)

        for sample_idx in range(sample_start, sample_stop):
            index = sample_idx - sample_start
            vertex1_idx = self._reshape_vertex_list(vertex1_ids[index], num_vertex1_ids[index])
            vertex2_idx = self._reshape_vertex_list(vertex2_ids[index], num_vertex2_ids[index])
            score_obj1[sample_idx] = [vertex1_idx, force1[index], np.array([obj2_idx, coupling_strenght1[index]])]
            score_obj2[sample_idx] = [vertex2_idx, force2[index], np.array([obj1_idx, coupling_strenght2[index]])]

        np.savez_compressed(f"{self.score_path}/{config_obj1.name}_{samples[0]:05d}_{samples[-1]:05d}.npz", score_obj1)
        np.savez_compressed(f"{self.score_path}/{config_obj2.name}_{samples[0]:05d}_{samples[-1]:05d}.npz", score_obj2)

    def _reshape_vertex_list(self, vertex_ids: np.ndarray, num_vertex: int) -> np.ndarray:
        if not np.all(vertex_ids == 0) and not vertex_ids.shape[0] == num_vertex:
            _vertex_ids = np.trim_zeros(vertex_ids, trim='b').astype(np.int32)
            if not _vertex_ids.shape[0] == num_vertex:
                delta = num_vertex - _vertex_ids.shape[0]
                _vertex_ids = np.append(_vertex_ids, np.zeros(delta, dtype=np.int32))
            return _vertex_ids
        return np.array([])

    def _load_audioforce_tracks(self, samples: np.ndarray, forces_path: str, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and sum audio-force tracks for obj_name in forces_path"""
        sample_start = samples[0]
        sample_stop = samples[-1] + 1
        audio_force = np.empty(sample_stop - sample_start)
        coupling_strenght = np.empty(sample_stop - sample_start)
        if os.path.exists(f"{forces_path}/{obj_name}_impact.raw"):
            audio_force += np.fromfile(f"{forces_path}/{obj_name}_impact.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_sliding.raw"):
            audio_force += np.fromfile(f"{forces_path}/{obj_name}_sliding.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_scraping.raw"):
            audio_force += np.fromfile(f"{forces_path}/{obj_name}_scraping.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_rolling.raw"):
            audio_force += np.fromfile(f"{forces_path}/{obj_name}_rolling.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_non_collision.raw"):
            audio_force += np.fromfile(f"{forces_path}/{obj_name}_non_collision.raw", dtype=np.float32)[sample_start:sample_stop]
        if os.path.exists(f"{forces_path}/{obj_name}_coupling_strenght.raw"):
            coupling_strenght = np.fromfile(f"{forces_path}/{obj_name}_coupling_strenght.raw", dtype=np.float32)[sample_start:sample_stop]

        return audio_force, coupling_strenght
