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

__version__ = "0.2.21"
__author__ = "Malcom3D"
__description__ = "Physically plausible impact sound for rigid body simulation"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.entity_manager import EntityManager
from .core.collision_engine import CollisionEngine
from .lib.samples import SampleCounter
from .lib.rigidbody_synth import ConnectedBuffer

class rigidBody:
    def __init__(self, config_file: str):
        self.em = EntityManager(config_file)
        sample_counter = SampleCounter()
        connected_buffer = ConnectedBuffer()
        self.em.register('sample_counter', sample_counter)
        self.em.register('connected_buffer', connected_buffer)
        self.ce = CollisionEngine(self.em)

    def prebake(self):
        self.ce.prebake()

    def bake(self):
        self.ce.bake()
