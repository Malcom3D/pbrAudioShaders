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
from typing import Any, List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from scipy.signal import fftconvolve
from numba import jit, prange

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _parse_lib
from ..lib.modal_bank import ModalBank

@dataclass
class RigidBodySynth:
    """Optimized RigidBodySynth with convolution matrix approach"""
    entity_manager: EntityManager
    obj_idx: int
    modal_lib: str
    vertex_list: np.ndarray
    sample_rate: int
    
    def __post_init__(self):
        self.connected_buffer = self.entity_manager.get('connected_buffer')
        self.modal_data = _parse_lib(self.modal_lib)
        
        # Initialize modal banks
        n_vertices = len(self.vertex_list)
        n_modes = self.modal_data['nModes']
        
        self.banks = np.zeros(n_vertices, dtype=object)
        for idx in range(n_vertices):
            vertex_id = int(self.vertex_list[idx])
            if vertex_id < len(self.modal_data['gains']):
                self.banks[idx] = ModalBank(
                    frequencies=self.modal_data['frequencies'],
                    gains=self.modal_data['gains'][vertex_id],
                    t60s=self.modal_data['t60s'],
                    sample_rate=self.sample_rate
                )
        
        # Pre-compute impulse responses for all vertices
        self._precompute_impulse_responses()
        
        # Cache for batch processing
        self._ir_matrix = None
        self._last_batch_size = 0
    
    def _precompute_impulse_responses(self, max_samples: int = 48000):
        """Pre-compute impulse responses for all active banks"""
        self._ir_matrix = np.zeros((len(self.banks), max_samples), dtype=np.float32)
        
        for idx, bank in enumerate(self.banks):
            if isinstance(bank, ModalBank):
                ir = bank.compute_impulse_response(max_samples)
                self._ir_matrix[idx, :len(ir)] = ir
    
    def process(self, synth_type: int, vertex_ids: List[int], 
                input_force: float, contact_area: float, 
                other_objs: List[Tuple[float, float]] = None) -> float:
        """Process a single sample (streaming mode)"""
        input_buffer = self.connected_buffer.read_for_obj(self.obj_idx, synth_type)
        total_excitation = input_force + input_buffer
        
        if total_excitation == 0 or len(vertex_ids) == 0:
            return 0.0
        
        output = 0.0
        for vertex_id in vertex_ids:
            if vertex_id < len(self.banks) and isinstance(self.banks[vertex_id], ModalBank):
                output += self.banks[vertex_id].process(total_excitation)
        
        # Handle coupling to other objects
        if other_objs is not None and input_force != 0:
            for other_idx in range(len(other_objs)):
                other_obj_idx, coupling_strength = other_objs[other_idx]
                self.connected_buffer.write_to_obj(
                    int(other_obj_idx), synth_type, 
                    coupling_strength * input_force
                )
        
        return output
    
    def process_batch(self, events_data: Dict) -> np.ndarray:
        """
        Process a batch of events using convolution matrix approach.
        
        Args:
            events_data: Dictionary containing:
                - 'excitations': np.ndarray of shape (n_events, n_samples)
                - 'vertex_ids': List of vertex ID lists for each event
                - 'synth_types': List of synth types
                - 'contact_areas': List of contact areas
                
        Returns:
            Output audio signal as np.ndarray
        """
        excitations = events_data['excitations']
        vertex_ids_list = events_data['vertex_ids']
        synth_types = events_data['synth_types']
        contact_areas = events_data.get('contact_areas', [1.0] * len(excitations))
        
        n_events = len(excitations)
        n_samples = max(len(e) for e in excitations) if isinstance(excitations, list) else excitations.shape[1]
        
        # Initialize output buffer
        output = np.zeros(n_samples, dtype=np.float32)
        
        # Process each event
        for event_idx in range(n_events):
            excitation = excitations[event_idx] if isinstance(excitations, list) else excitations[event_idx]
            vertex_ids = vertex_ids_list[event_idx]
            contact_area = contact_areas[event_idx]
            
            # Apply contact area scaling
            scaled_excitation = excitation * contact_area
            
            # Convolve with each vertex's impulse response
            for vertex_id in vertex_ids:
                if vertex_id < len(self.banks) and isinstance(self.banks[vertex_id], ModalBank):
                    # Get impulse response
                    ir = self._ir_matrix[vertex_id, :len(scaled_excitation)]
                    
                    # FFT convolution
                    conv_result = fftconvolve(scaled_excitation, ir, mode='full')[:n_samples]
                    output += conv_result
        
        return output
    
    def process_batch_toeplitz(self, events_data: Dict) -> np.ndarray:
        """
        Process batch using Toeplitz matrix structure for maximum SIMD efficiency.
        
        This creates a Toeplitz matrix from the impulse response and uses
        matrix multiplication for convolution.
        """
        excitations = events_data['excitations']
        vertex_ids_list = events_data['vertex_ids']
        
        n_events = len(excitations)
        n_samples = max(len(e) for e in excitations) if isinstance(excitations, list) else excitations.shape[1]
        
        output = np.zeros(n_samples, dtype=np.float32)
        
        for event_idx in range(n_events):
            excitation = excitations[event_idx] if isinstance(excitations, list) else excitations[event_idx]
            vertex_ids = vertex_ids_list[event_idx]
            
            # Create excitation vector
            x = np.array(excitation, dtype=np.float32)
            
            for vertex_id in vertex_ids:
                if vertex_id < len(self.banks) and isinstance(self.banks[vertex_id], ModalBank):
                    # Get impulse response
                    h = self._ir_matrix[vertex_id, :n_samples]
                    
                    # Create Toeplitz matrix (first column)
                    # T = toeplitz(h, [h[0], zeros])
                    # Result = T @ x
                    
                    # Efficient implementation: use FFT
                    conv_result = fftconvolve(x, h, mode='full')[:n_samples]
                    output += conv_result
        
        return output
    
    def get_banks_state(self) -> List:
        """Get current state of all banks"""
        states = []
        for idx in range(len(self.banks)):
            if isinstance(self.banks[idx], ModalBank):
                states.append([
                    self.banks[idx].u1.copy(),
                    self.banks[idx].u2.copy()
                ])
            else:
                states.append(None)
        return states
    
    def set_banks_state(self, states: List):
        """Set state of all banks"""
        for banks_idx in range(len(states)):
            if states[banks_idx] is not None and isinstance(self.banks[banks_idx], ModalBank):
                self.banks[banks_idx].u1 = states[banks_idx][0]
                self.banks[banks_idx].u2 = states[banks_idx][1]

