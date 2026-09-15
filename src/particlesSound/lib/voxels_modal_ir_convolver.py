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
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from scipy import signal

from pbrAudioCommon import _parse_lib
from pbrAudioCommon import debug_print

@dataclass
class VoxelsModalIRConvolver:
    """
    A bank of modal IR convolvers for different materials.
    It manages the state (velocity and displacement) for each mode of each material.
    """
    sample_rate: int
    modal_libs: Dict[int, str]  # Map obj_idx to its modal .lib file path
    # State for each mode of each material
    # {obj_idx: {'u1': np.ndarray, 'u2': np.ndarray}}
    state: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # Pre-computed coefficients for each mode
    # {obj_idx: {'c': np.ndarray, 's': np.ndarray, 'g': np.ndarray}}
    coeffs: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self):
        for obj_idx, lib_path in self.modal_libs.items():
            try:
                modal_data = _parse_lib(lib_path)
                n_modes = len(modal_data['frequencies'])
                
                # Calculate modal filter coefficients
                r = np.zeros(n_modes)
                c = np.zeros(n_modes)
                s = np.zeros(n_modes)
                g = np.zeros(n_modes)
                
                omega = 2.0 * np.pi * modal_data['frequencies'] / self.sample_rate
                
                for i in range(n_modes):
                    if modal_data['t60s'][i] > 0:
                        bandwidth = 0.35 / modal_data['t60s'][i]
                        decay_per_sample = np.pi * bandwidth / self.sample_rate
                        r[i] = np.exp(-decay_per_sample)
                    
                    c[i] = r[i] * np.cos(omega[i])
                    s[i] = r[i] * np.sin(omega[i])
                    
                    # Use the average gain for the IR convolver
                    avg_gain = np.mean(modal_data['gains'][i]) if modal_data['gains'][i].size > 0 else 0
                    if r[i] < 1.0 and r[i] > 0:
                        g[i] = avg_gain * (1.0 - r[i]) * s[i]

                self.coeffs[obj_idx] = {'c': c, 's': s, 'g': g}
                self.state[obj_idx] = {
                    'u1': np.zeros(n_modes, dtype=np.float32),
                    'u2': np.zeros(n_modes, dtype=np.float32)
                }
            except Exception as e:
                debug_print(f"Failed to load or process modal lib for obj {obj_idx}: {e}")

    def process(self, obj_idx: int, excitation: float) -> float:
        """
        Excite the modal model for a given object and return the output sample.
        """
        if obj_idx not in self.coeffs:
:
            return 0.0

        c = self.coeffs[obj_idx]['c']
        s = self.coeffs[obj_idx]['s']
        g = self.coeffs[obj_idx]['g']
        u1 = self.state[obj_idx]['u1']
        u2 = self.state[obj_idx]['u2']
        
        # Vectorized update for all modes
        u1_new = c * u1 - s * u2 + g * excitation
        u2_new = s * u1 + c * u2
        
        self.state[obj_idx]['u1'] = u1_new
        self.state[obj_idx]['u2'] = u2_new
        
        return np.sum(u2_new)

