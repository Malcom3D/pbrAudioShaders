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

# pbrAudioShaders/src/fractureSound/__init__.py
"""
Fracture Sound Synthesis Module

Implements physically-based fracture sound synthesis based on:
"Fracture Sound: A physically based approach to the synthesis of fracture sounds"
by K. van den Doel, P.G. Kry, and D.K. Pai
https://www.cs.cornell.edu/projects/FractureSound/files/fractureSound_comp.pdf
"""

__version__ = "0.2.0"
__author__ = "Malcom3D"
__description__ = "Physically plausible fracture sound synthesis"

import os
import sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.fracture_engine import fractureEngine
from .lib.fracture_data import FractureEvent, FractureType, FragmentData
from .lib.fracture_modal import FractureModalModel
from .lib.fracture_synth import FractureSynth
from .lib.fracture_detector import FractureDetector

__all__ = [
    'fractureEngine',
    'FractureEvent',
    'FractureType',
    'FragmentData',
    'FractureModalModel',
    'FractureSynth',
    'FractureDetector'
]
