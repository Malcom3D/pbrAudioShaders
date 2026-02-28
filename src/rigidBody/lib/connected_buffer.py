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

@dataclass
class ConnectedBuffer:
    objs_buffer: np.ndarray = field(default_factory=lambda: np.array([]))

    def add_obj(self, obj_idx: int):
        new_inst = 1 + obj_idx - self.objs_buffer.shape[0]
        if new_inst > 0:
            buffer_type = [0 for _ in range(6)]
            for _ in range(new_inst):
                if len(self.objs_buffer.tolist()) == 0:
                    self.objs_buffer = np.array([buffer_type])
                else:
                    self.objs_buffer = np.array(self.objs_buffer.tolist() + [buffer_type])

    def read_for_obj(self, obj_idx: int, synth_type: int):
        sample_value = self.objs_buffer[obj_idx][synth_type]
        self.objs_buffer[obj_idx][synth_type] = 0
        return sample_value

    def write_to_obj(self, obj_idx: int, synth_type: int, sample_value: float):
        self.objs_buffer[obj_idx][synth_type] += sample_value
