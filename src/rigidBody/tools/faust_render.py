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

@dataclass
class FaustRender:
    faust_render: str = "render_faust_snd"
    duration: float = 3.0
    """
        Input parameters

        - dsp_file: file with Faust code
        - output_file: RAW PCM FLOAT32 audio file
    """

    def __post_init__(self):
        bin_dir = f"{os.path.dirname(os.path.abspath(sys.modules[FaustRender.__module__].__file__))}/../bin"
        os.environ['LD_LIBRARY_PATH'] = bin_dir
        self.faust_render = f"{bin_dir}/{self.faust_render}"

    def compute(self, dsp_file: str, output_file: str, duration: float = None):
        self.duration = duration if duration != None else self.duration
        cmd = f"{self.faust_render} {dsp_file} {output_file} {self.duration}"
        print(cmd)
        exit_code = os.system(cmd)
        if not exit_code == 0:
            print('Error')

#audio_data = np.fromfile('yourfile.raw', dtype=np.float32)
