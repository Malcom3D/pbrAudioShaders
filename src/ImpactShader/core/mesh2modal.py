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
import trimesh
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..core.impact_manager import ImpactManager
from ..tools.pym2f import Pym2f

@dataclass
class Mesh2Modal:
    impact_manager: ImpactManager
    
    def __post_init__(self):
        print('Mesh2Modal.post_init')
        self.py_m2f = Pym2f(self.impact_manager)

    def compute(self, obj_idx: int) -> None:
        print('Mesh2Modal.compute')
        config = self.impact_manager.get('config')
        
        # Find the object configuration
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:

                # Find the optimized .obj file in the directory
                obj_path = config_obj.obj_path
                optimized_obj_path = os.path.join(obj_path, f"optimized_{config_obj.name}.obj")
                expos = self.impact_manager.get_expos(config_obj.idx)
                self.py_m2f.compute(config_obj.idx, expos)
