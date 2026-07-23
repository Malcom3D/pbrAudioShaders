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
from scipy.sparse import lil_matrix, csr_matrix

@dataclass
class ConnectedBuffer:
    """Optimized ConnectedBuffer with matrix operations"""
    objs_buffer: np.ndarray = field(default_factory=lambda: np.zeros((1, 7), dtype=np.float32))
    _coupling_matrix: Optional[np.ndarray] = None
    _batch_mode: bool = False
    _batch_buffer: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self._batch_buffer = np.zeros_like(self.objs_buffer)
    
    def add_obj(self, obj_idx: int):
        """Add object to buffer"""
        if obj_idx >= self.objs_buffer.shape[0]:
            # Expand buffer
            new_buffer = np.zeros((obj_idx + 1, 7), dtype=np.float32)
            new_buffer[:self.objs_buffer.shape[0], :] = self.objs_buffer
            self.objs_buffer = new_buffer
            
            new_batch = np.zeros((obj_idx + 1, 7), dtype=np.float32)
            new_batch[:self._batch_buffer.shape[0], :] = self._batch_buffer
            self._batch_buffer = new_batch
    
    def set_coupling_matrix(self, coupling_matrix: np.ndarray):
        """Set the coupling strength matrix between objects"""
        self._coupling_matrix = coupling_matrix
    
    def read_for_obj(self, obj_idx: int, synth_type: int) -> float:
        """Read and clear buffer for a specific object and synth type"""
        if self._batch_mode:
            value = self._batch_buffer[obj_idx][synth_type]
            self._batch_buffer[obj_idx][synth_type] = 0
        else:
            value = self.objs_buffer[obj_idx][synth_type]
]
            self.objs_buffer[obj_idx][synth_type] = 0
        return value
    
    def write_to_obj(self, obj_idx: int, synth_type: int, sample_value: float):
        """Write sample value to buffer for a specific object"""
        if self._batch_mode:
            self._batch_buffer[obj_idx][synth_type] += sample_value
        else:
            self.objs_buffer[obj_idx][synth_type] += sample_value
    
    def process_batch_coupling(self, excitations: np.ndarray, 
                                obj_indices: List[int],
                                synth_types: List[int]) -> np.ndarray:
        """
        Process coupling for a batch of excitations.
        
        Args:
            excitations: Shape (n_events, n_samples) or (n_samples,)
            obj_indices: List of object indices for each event
            synth_types: List of synth types for each event
            
        Returns:
            Coupled excitations matrix
        """
        if self._coupling_matrix is None:
            return excitations
        
        # Convert to 2D if necessary
        if excitations.ndim == 1:
            excitations = excitations.reshape(1, -1)
        
        n_events = len(obj_indices)
        n_samples = excitations.shape[1]
        
        # Create coupling matrix for this batch
        coupled = np.zeros_like(excitations)
        
        for i in range(n_events):
            obj_idx = obj_indices[i]
            synth_type = synth_types[i]
            
            # Get coupling strengths for this object
            coupling_strengths = self._coupling_matrix[obj_idx]
            
            # Apply coupling
            coupled[i] = excitations[i] * coupling_strengths[synth_type]
        
        return coupled
    
    def enable_batch_mode(self):
        """Enable batch processing mode"""
        self._batch_mode = True
        self._batch_buffer.fill(0)
    
    def disable_batch_mode(self):
        """Disable batch processing mode and flush buffer"""
        self._batch_mode = False
        self.objs_buffer += self._batch_buffer
        self._batch_buffer.fill(0)
