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

import os, sys
import numpy as np
from typing import List
from dataclasses import dataclass

from dask import delayed, compute

from ..core.impact_manager import ImpactManager
from ..tools.faust_render import FaustRender

@dataclass
class ModalSynth:
    impact_manager: ImpactManager

    def __post_init__(self):
        self.faust_render = FaustRender()
        
    def compute(self):
        config = self.impact_manager.get('config')
        output_path = config.system.output_path
        os.makedirs(output_path, exist_ok=True)
        dsp_files = [os.path.join(f"{output_path}/dsp", f) for f in os.listdir(f"{output_path}/dsp") if os.path.isfile(os.path.join(f"{output_path}/dsp", f)) and f.endswith('dsp')]
        tasks = [self.faust_render.compute(dsp_file, dsp_file.replace('.dsp', '.raw')) for dsp_file in dsp_files]
        compute(*tasks)
