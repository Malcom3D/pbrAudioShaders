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

import numpy as np
from dask import compute
from typing import Any, List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field

from physicsSolver import EntityManager

from physicsSolver.lib.functions import _parse_lib
from ..lib.modal_bank import ModalBank

@dataclass
class RigidBodySynth:
    entity_manager: EntityManager
    obj_idx: int
    modal_lib: str
    vertex_list: np.ndarray
    sample_rate: int

    def __post_init__(self):
        self.connected_buffer = self.entity_manager.get('connected_buffer')
        self.modal_data = _parse_lib(self.modal_lib)
        self.banks = np.zeros(len(self.modal_data['gains']), dtype=object)
        #print('RigidBodySynth: ', len(self.modal_data['gains']), self.vertex_list.shape[0], self.modal_data['nModes'])
        for idx in range(self.vertex_list.shape[0]):
            self.banks[int(self.vertex_list[idx])] = ModalBank(frequencies=self.modal_data['frequencies'], gains=self.modal_data['gains'][int(self.vertex_list[idx])], t60s=self.modal_data['t60s'], sample_rate=self.sample_rate)
        
    def process(self, synth_type: int, vertex_ids: List[int], input_force: float, contact_area: float, other_objs: List[Tuple[float, float]] = None):
        banks_tasks = []
        output_banks = 0
        input_buffer = self.connected_buffer.read_for_obj(self.obj_idx, synth_type)
        for idx in range(len(vertex_ids)):
#            output_banks += self.banks[vertex_ids[idx]].process(input_force + input_buffer)
            banks_tasks += [self.banks[vertex_ids[idx]].process(input_force + input_buffer)]
            output_banks = sum(compute(*banks_tasks))
        if isinstance(other_objs, list):
            for other_idx in range(len(other_objs)):
                other_obj_idx, coupling_strength = other_objs[other_idx]
                self.connected_buffer.write_to_obj(int(other_obj_idx), synth_type, coupling_strength * input_force)
        #print('RigidBodySynth: ', self.obj_idx, output_banks)
        return output_banks 

    def get_banks_state(self) -> List[Union[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        states = []
        for idx in range(len(self.banks)):
            if not isinstance(self.banks[idx], int):
                states.append([self.banks[idx].u1, self.banks[idx].u2, self.banks[idx].s, self.banks[idx].c, self.banks[idx].g])
            else:
                states.append(0)
        return states

    def set_banks_state(self, states: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        for banks_idx in range(len(states)):
            if not isinstance(states[banks_idx], int):
                self.banks[banks_idx].u1 = states[banks_idx][0]
                self.banks[banks_idx].u2 = states[banks_idx][1]
                self.banks[banks_idx].s = states[banks_idx][2]
                self.banks[banks_idx].c = states[banks_idx][3]
                self.banks[banks_idx].g = states[banks_idx][4]
