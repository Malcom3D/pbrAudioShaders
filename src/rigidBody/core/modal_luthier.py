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
from ..lib.resonance_synth import ResonanceSynth

@dataclass
class ModalLuthier:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.connected_buffer = self.entity_manager.get('connected_buffer')
        self.dsp_path = f"{config.system.cache_path}/dsp"

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        sample_rate = config.system.sample_rate
        modal_vertices = self.entity_manager.get('modal_vertices')

        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj

        vertex_list = np.array([])
        connected_area = 0
        for idx in modal_vertices.keys():
            if modal_vertices[idx].obj_idx == obj_idx:
                vertex_list = modal_vertices[idx].get_vertices()
                if isinstance(config_obj.connected, np.ndarray):
                    connected_area = modal_vertices[idx].connected_area

        self.connected_buffer.add_obj(obj_idx)

        print('ModalLuthier: init ', config_obj.name, 'RigidBodySynth')
        rigidbody_synth = RigidBodySynth(entity_manager=self.entity_manager, obj_idx=obj_idx, modal_lib=f"{self.dsp_path}/{config_obj.name}.lib", vertex_list=vertex_list, sample_rate=sample_rate)
        self.entity_manager.register('rigidbody_synth', rigidbody_synth, config_obj.idx)
        print('ModalLuthier: ', config_obj.name, 'RigidBodySynth registered')

        print('ModalLuthier: init ', config_obj.name, 'ResonanceSynth')
        if not connected_area == 0:
            contact_area_scale = contact_area * len(vertex_list)
            resonance_synth = ResonanceSynth(entity_manager=self.entity_manager, obj_idx=obj_idx, modal_lib=f"{self.dsp_path}/{config_obj.name}_resonance.lib", sample_rate=sample_rate, contact_area_scale=contact_area_scale)
        else:
            resonance_synth = ResonanceSynth(entity_manager=self.entity_manager, obj_idx=obj_idx, modal_lib=f"{self.dsp_path}/{config_obj.name}_resonance.lib", sample_rate=sample_rate)
        self.entity_manager.register('resonance_synth', resonance_synth, config_obj.idx)
        print('ModalLuthier: ', config_obj.name, 'ResonanceSynth registered')

