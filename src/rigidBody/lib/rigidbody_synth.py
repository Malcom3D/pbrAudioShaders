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
    objs_buffer: np.ndarray = field(default_factory=lambda: np.array([]))

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
            new_inst = 1 + obj_idx - self.objs_buffer.shape[0]
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
        
    def process(self, sample_idx: int, vertex_ids: List[int], input_banks: float, other_objs: List[Tuple[float, float]] = None):
        output_banks = 0
        for idx in range(len(vertex_ids)):
            input_buffer = self.connected_buffer.read_for_obj(self.obj_idx, sample_idx)
            output_banks += self.banks[vertex_ids[idx]].process(input_banks + input_buffer)
        if isinstance(other_objs, list):
            for other_idx in range(len(other_objs)):
                other_obj_idx, coupling_strength = other_objs[other_idx]
                self.connected_buffer.write_to_obj(int(other_obj_idx), coupling_strength * output_banks, sample_idx)
        return output_banks 

    def get_banks_state(self) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

class ModalBank:
    """
    Fully vectorized modal bank implementation using coupled-form structure
    for maximum numerical stability and performance.
    """
    def __init__(self, frequencies: np.ndarray, gains: np.ndarray, t60s: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        self.num_modes = len(frequencies)
        
        # Store parameters
        self.frequencies = np.array(frequencies, dtype=np.float32)
        self.gains = np.array(gains, dtype=np.float32)
        self.t60s = np.array(t60s, dtype=np.float32)
        
        # Coupled-form coefficients
        self.r = np.zeros(self.num_modes, dtype=np.float32)  # radius (decay)
        self.c = np.zeros(self.num_modes, dtype=np.float32)  # cos(theta)
        self.s = np.zeros(self.num_modes, dtype=np.float32)  # sin(theta)
        self.g = np.zeros(self.num_modes, dtype=np.float32)  # gain scaling
        
        # Coupled-form state variables (u1, u2)
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)
        
        # Calculate initial coefficients
        self._calculate_all_coefficients()
    
    def _calculate_all_coefficients(self):
        """Calculate coupled-form coefficients for all modes"""
        # Angular frequency
        omega = 2.0 * np.pi * self.frequencies / self.sample_rate
        
        # Calculate decay factor r from T60
        with np.errstate(divide='ignore', invalid='ignore'):
            # r = exp(-π * bandwidth / sample_rate)
            # bandwidth = 2.2 / (T60 * 2π) = 0.35 / T60
            bandwidth = 0.35 / self.t60s
            bandwidth[self.t60s <= 0] = 0.0
            
            # Decay per sample
            decay_per_sample = np.pi * bandwidth / self.sample_rate
            self.r = np.exp(-decay_per_sample)
            self.r[np.isnan(self.r)] = 0.0
            self.r[self.t60s <= 0] = 0.0
        
        # Coupled-form rotation matrix elements
        self.c = self.r * np.cos(omega)  # r * cos(θ)
        self.s = self.r * np.sin(omega)  # r * sin(θ)
        
        # Gain scaling for unity peak gain at resonance
        # For coupled-form, output is taken from u2, so we scale by sin(θ)
        # Additional scaling by (1 - r) ensures proper gain
#        with np.errstate(divide='ignore', invalid='ignore'):
#            # Peak gain of coupled-form resonator: 1/(1 - r) at resonance
#            peak_gain = 1.0 / (1.0 - self.r)
#            peak_gain[np.isinf(peak_gain)] = 1.0
#            
#            # Desired gain is self.gains, so scale accordingly
#            self.g = self.gains * self.s / peak_gain
#        
        # Alternative simpler gain calculation:
        self.g = self.gains * (1.0 - self.r) * self.s
    
    def process(self, excitation: float) -> float:
        """
        Process one sample through all modes using coupled-form structure.
        
        Coupled-form equations:
        u1[n] = c * u1[n-1] - s * u2[n-1] + g * x[n]
        u2[n] = s * u1[n-1] + c * u2[n-1]
        y[n] = u2[n]
        """
        # Vectorized coupled-form update
        u1_new = self.c * self.u1 - self.s * self.u2 + self.g * excitation
        u2_new = self.s * self.u1 + self.c * self.u2
        
        # Update states
        self.u1 = u1_new
        self.u2 = u2_new
        
        # Sum outputs from all modes (output is u2)
        output = np.sum(u2_new)
        
        return output
    
    def reset(self):
        """Reset all state variables to zero"""
        self.u1 = np.zeros(self.num_modes, dtype=np.float32)
        self.u2 = np.zeros(self.num_modes, dtype=np.float32)
    
    def update_frequency(self, mode_idx: int, frequency: float):
        """Update frequency for a specific mode"""
        self.frequencies[mode_idx] = frequency
        self._update_mode_coefficients(mode_idx)
    
    def update_gain(self, mode_idx: int, gain: float):
        """Update gain for a specific mode"""
        self.gains[mode_idx] = gain
        self._update_mode_coefficients(mode_idx)
    
    def update_t60(self, mode_idx: int, t60: float):
        """Update T60 for a specific mode"""
        self.t60s[mode_idx] = t60
        self._update_mode_coefficients(mode_idx)
    
    def _update_mode_coefficients(self, mode_idx: int):
        """Update coefficients for a single mode"""
        # Recalculate for this specific mode
        omega = 2.0 * np.pi * self.frequencies[mode_idx] / self.sample_rate
        
        # Calculate decay factor
        if self.t60s[mode_idx] > 0:
            bandwidth = 0.35 / self.t60s[mode_idx]
            decay_per_sample = np.pi * bandwidth / self.sample_rate
            self.r[mode_idx] = np.exp(-decay_per_sample)
        else:
            self.r[mode_idx] = 0.0
        
        # Update coupled-form coefficients
        self.c[mode_idx] = self.r[mode_idx] * np.cos(omega)
        self.s[mode_idx] = self.r[mode_idx] * np.sin(omega)
        
        # Update gain scaling
        if self.r[mode_idx] < 1.0:
            peak_gain = 1.0 / (1.0 - self.r[mode_idx])
            self.g[mode_idx] = self.gains[mode_idx] * self.s[mode_idx] / peak_gain
        else:
            self.g[mode_idx] = 0.0
