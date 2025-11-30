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

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..lib.impact_data import Collision, ObjCollision, ImpactData

@dataclass
class ObjectConfig:
    idx: int
    obj_path: str
    young_modulus: float
    poisson_ration: float
    density: float
    damping: float

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.data = json.load(f)

        # Handle objects physical data
        self.objects = []
        for o in self.data.get('objects', []):
            physical_properties = o.get('physical_properties', {})

            object_config = ObjectConfig(
                **{k: v for k, v in o.items() if != 'physical_properties'},
                young_modulus = physical_properties.get('young_modulus', []),
                poisson_ration = physical_properties.get('poisson_ration', []),
                density = physical_properties.get('density', []),
                damping = physical_properties.get('damping', [])
            )
            self.objects.append(object_config)
