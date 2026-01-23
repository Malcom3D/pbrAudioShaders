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
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..core.entity_manager import EntityManager
from ..tools.pym2f import Pym2f

@dataclass
class Mesh2Modal:
    entity_manager: EntityManager
    
    def __post_init__(self):
        self.py_m2f = Pym2f(self.entity_manager)

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        
        self.py_m2f.compute(obj_idx)
