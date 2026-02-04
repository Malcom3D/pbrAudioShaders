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
from ..lib.rigidbody_synth import RigidBodySynth

from ..lib.functions import _parse_lib

@dataclass
class ModalLuthier:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        sample_counter = self.entity_manager.get('sample_counter')
        self.connected_buffer = self.entity_manager.get('connected_buffer')
        self.connected_buffer.set_total_samples(sample_counter.total_samples)
        self.dsp_path = f"{config.system.cache_path}/dsp"

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        sample_rate = config.system.sample_rate

        vertex_list = np.empty(0)
        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                collision_data = self.entity_manager.get('collisions')
                for c_idx in collision_data.keys():
                    if collision_data[c_idx].obj1_idx == obj_idx:
                        vertex_list = np.append(vertex_list, collision_data[c_idx].collision_area[3])
                    if collision_data[c_idx].obj2_idx == obj_idx:
                        vertex_list = np.append(vertex_list, collision_data[c_idx].collision_area[6])

        self.connected_buffer.add_obj(obj_idx)

        rigidbody_synth = RigidBodySynth(entity_manager=self.entity_manager, obj_idx=obj_idx, modal_lib=f"{self.dsp_path}/{config_obj.name}.lib", vertex_list=vertex_list, sample_rate=sample_rate)
        self.entity_manager.register('rigidbody_synth', rigidbody_synth, config_obj.idx)

