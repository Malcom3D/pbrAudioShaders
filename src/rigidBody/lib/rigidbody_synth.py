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
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager

from ..lib.functions import _parse_lib

@dataclass
class ConnectedBuffer:
    total_samples: int = 0
    objs_buffer: np.ndarray = None

    def set_total_samples(self, total_samples: int):
        self.total_samples = total_samples
        if isinstance(self.objs_buffer, np.ndarray):
            if not np.all(self.objs_buffer):
                buffer_list = [np.zeros(self.total_samples, dtype=np.float32) for _ in range(self.objs_buffer.shape[0])]
                self.objs_buffer = np.array(buffer_list)
            else:
                buffer_list = []
                for i in range(self.objs_buffer.shape[0]):
                    delta = self.total_samples - self.objs_buffer[i].shape[0]
                    buffer_list += [np.append(self.objs_buffer[i], np.zeros(delta))]
                self.objs_buffer = np.array(buffer_list)

    def add_obj(self, obj_idx: int):
        if not isinstance(self.objs_buffer, np.ndarray):
            buffer_list = [np.zeros(self.total_samples, dtype=np.float32) for _ in range(obj_idx)]
            self.objs_buffer = np.array(buffer_list)
        else:
            new_inst = obj_idx - self.objs_buffer.shape[0]
            if new_inst > 0:
                obj_buffer = np.zeros(self.total_samples, dtype=np.float32)
                for _ in range(new_inst):
                    objs_buffer = self.objs_buffer.tolist()
                    objs_buffer += [obj_buffer.tolist()]
                    self.objs_buffer = np.array(objs_buffer)

    def read_for_obj(self, obj_idx: int, sample_idx: int):
        return self.objs_buffer[obj_idx][sample_idx]

    def write_to_obj(self, obj_idx: int, sample_value: float, sample_idx: int):
        self.objs_buffer[obj_idx][sample_idx] = sample_value

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
        for idx in range(self.vertex_list.shape[0]):
            self.banks[int(self.vertex_list[idx])] = ModalBank(frequencies=self.modal_data['frequencies'], gains=self.modal_data['gains'][int(self.vertex_list[idx])], t60s=self.modal_data['t60s'], sample_rate=self.sample_rate)
        
    def process(self, sample_idx: int, vertex_ids: List[int], input_banks: float, other_objs: List[Tuple[int, float]] = None):
        output_banks = 0
        for vertex in range(len(vertex_ids)):
            input_buffer = self.connected_buffer.read_for_obj(self.obj_idx, sample_idx)
            output_banks += self.banks[int(vertex)].process(input_banks + input_buffer)
        if not other_objs == None:
            for other_idx in range(len(other_objs)):
                other_obj_idx, coupling_strength = other_objs[other_idx]
                self.connected_buffer.write_to_obj(other_obj_idx, coupling_strength * output_banks, sample_idx)
        return output_banks 

    def get_banks_state(self) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        states = []
        for idx in range(self.banks.shape[0]):
            states.append([self.banks[idx].x1, self.banks[idx].x2, self.banks[idx].y1, self.banks[idx].y2])
        return states

    def set_banks_state(self, states: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        for idx in range(len(states)):
            self.banks[states[idx][0]].x1 = states[idx][0]
            self.banks[states[idx][0]].x2 = states[idx][1]
            self.banks[states[idx][0]].y1 = states[idx][2]
            self.banks[states[idx][0]].y2 = states[idx][3]

class ModalBank:
    """
    Fully vectorized modal bank implementation for maximum performance.
    This uses NumPy operations only, no Python loops in processing.
    """
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, t60s: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        self.num_modes = len(frequencies)
        
        # Store parameters
        self.frequencies = np.array(frequencies, dtype=np.float32)
        self.gains = np.array(gains, dtype=np.float32)
        self.t60s = np.array(t60s, dtype=np.float32)
        
        # Initialize coefficients arrays
        self.b0 = np.zeros(self.num_modes, dtype=np.float32)
        self.b1 = np.zeros(self.num_modes, dtype=np.float32)
        self.b2 = np.zeros(self.num_modes, dtype=np.float32)
        self.a1 = np.zeros(self.num_modes, dtype=np.float32)
        self.a2 = np.zeros(self.num_modes, dtype=np.float32)
        
        # State arrays
        self.x1 = np.zeros(self.num_modes, dtype=np.float32)
        self.x2 = np.zeros(self.num_modes, dtype=np.float32)
        self.y1 = np.zeros(self.num_modes, dtype=np.float32)
        self.y2 = np.zeros(self.num_modes, dtype=np.float32)
        
        # Calculate initial coefficients
        self._calculate_all_coefficients()
    
    def _calculate_all_coefficients(self):
        """Calculate coefficients for all modes"""
        omega = 2.0 * np.pi * self.frequencies / self.sample_rate
        
        # Calculate bandwidth for each mode
        with np.errstate(divide='ignore', invalid='ignore'):
            bandwidth = 2.2 / (self.t60s * 2.0 * np.pi)
            bandwidth[self.t60s <= 0] = 0.0
        
        # Calculate alpha (bandwidth parameter)
        sin_omega = np.sin(omega)
        alpha = sin_omega * np.sinh(np.log(2.0) / 2.0 * bandwidth * omega / sin_omega)
        alpha[np.isnan(alpha)] = 0.0  # Handle division by zero
        
        # Bandpass coefficients
        self.b0 = alpha * self.gains
        self.b1 = np.zeros_like(self.b0)
        self.b2 = -alpha * self.gains
        self.a1 = -2.0 * np.cos(omega)
        self.a2 = 1.0 - alpha

    def process(self, excitation: float) -> float:
        """
        Fully vectorized processing by samples using NumPy.
        This is the fastest implementation for large numbers of modes.
        """
        # Vectorized computation for all modes at once
        y = (self.b0 * excitation +
             self.b1 * self.x1 +
             self.b2 * self.x2 -
             self.a1 * self.y1 -
             self.a2 * self.y2)

        # Sum all modal outputs
        output = np.sum(y)

        # Update states for next sample
        self.x2 = self.x1.copy()
        self.x1 = np.full(self.num_modes, excitation, dtype=np.float32)
        self.y2 = self.y1.copy()
        self.y1 = y.copy()

        return output
    
    def process_vectorized(self, excitation_buffer: np.ndarray) -> np.ndarray:
        """
        Fully vectorized processing using NumPy.
        This is the fastest implementation for large numbers of modes.
        """
        buffer_len = len(excitation_buffer)
        output = np.zeros(buffer_len, dtype=np.float32)
        
        # Process sample by sample (still need loop for time dependency)
        for i in range(buffer_len):
            # Vectorized computation for all modes at once
            y = (self.b0 * excitation_buffer[i] + 
                 self.b1 * self.x1 + 
                 self.b2 * self.x2 - 
                 self.a1 * self.y1 - 
                 self.a2 * self.y2)
            
            # Sum all modal outputs
            output[i] = np.sum(y)
            
            # Update states for next sample
            self.x2 = self.x1.copy()
            self.x1 = np.full(self.num_modes, excitation_buffer[i], dtype=np.float32)
            self.y2 = self.y1.copy()
            self.y1 = y.copy()
        
        return output
    
    def reset(self):
        """Reset all states"""
        self.x1[:] = 0.0
        self.x2[:] = 0.0
        self.y1[:] = 0.0
        self.y2[:] = 0.0
