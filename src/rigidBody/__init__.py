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
__description__ = "Physically plausible collision sound for rigid body simulation"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .tools.pym2f import Pym2f
from .tools.faust_render import FaustRender
#from .core.force_synth import ForceSynth
from .core.rigidbody_engine import rigidBodyEngine
from .core.modal_composer import ModalComposer
from .core.modal_luthier import ModalLuthier
from .core.modal_player import ModalPlayer
from .core.mesh2modal import Mesh2Modal
from .lib.connected_buffer import ConnectedBuffer
from .lib.rigidbody_synth import RigidBodySynth
from .lib.filter import LinkwitzRileyFilter
from .lib.modal_bank import ModalBank
from .lib.resonance_synth import ResonanceSynth
from .lib.sample_counter import FunctionLocker
from .lib.sample_counter import SampleCounter

__all__ = [
     'Pym2f',
     'FaustRender',
#     'ForceSynth',
     'rigidBodyEngine',
     'ModalComposer',
     'ModalLuthier',
     'ModalPlayer',
     'Mesh2Modal',
     'ConnectedBuffer',
     'RigidBodySynth',
     'LinkwitzRileyFilter',
     'ModalBank',
     'ResonanceSynth',
     'FunctionLocker',
     'SampleCounter'
]
#from physicsSolver import EntityManager, physicsEngine
#from .core.rigidbody_engine import rigidBodyEngine
#from .lib.sample_counter import SampleCounter
#from .lib.connected_buffer import ConnectedBuffer
#
#class rigidBody:
#    def __init__(self, config_file: str):
#        self.em = EntityManager(config_file)
#        sample_counter = SampleCounter()
#        connected_buffer = ConnectedBuffer()
#        self.em.register('sample_counter', sample_counter)
#        self.em.register('connected_buffer', connected_buffer)
#        self.physics_engine = PhysicsEngine(self.em)
#        self.rigidbody_engine = rigidBodyEngine(self.em)
#
#    def prebake(self):
#        self.physics_engine.prebake()
#        self.rigidbody_engine.prebake()
#
#    def bake(self):
#        self.physics_engine.bake()
#        self.rigidbody_engine.bake()
#
#    def prerender(self):
#        self.physics_engine.prerender()
#        self.rigidbody_engine.prerender()
#
#    def render(self):
#        self.rigidbody_engine.render()
