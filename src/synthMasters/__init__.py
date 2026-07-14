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

__version__ = "0.2.33"
__author__ = "Malcom3D"
__description__ = "Composer, luthier and players for physically plausible sound synthesis"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.modal_composer import ModalComposer
from .core.modal_luthier import ModalLuthier
from .core.modal_player import ModalPlayer

from .lib.score_data import ScoreEvent, ScoreTrack
from ..lib.sample_counter import SampleCounter
from ..lib.connected_buffer import ConnectedBuffer

__all__ = [
     'ModalComposer',
     'ModalLuthier',
     'ModalPlayer',
     'ScoreEvent',
     'ScoreTrack',
     'ConnectedBuffer',
     'SampleCounter'
]

