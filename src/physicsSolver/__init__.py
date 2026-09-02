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

__version__ = "0.2.38"
__author__ = "Malcom3D"
__description__ = "Reverse physics engine"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.position_solver import PositionSolver
from .core.rotation_solver import RotationSolver
from .core.vertex_solver import VertexSolver
from .core.normal_solver import NormalSolver
from .core.flight_path import FlightPath
from .core.distance_solver import DistanceSolver
from .core.force_solver import ForceSolver
from .core.force_synth import ForceSynth
from .core.collision_solver import CollisionSolver
from .core.particles_solver import ParticlesSolver
from .core.physics_engine import physicsEngine
from .lib.contact_geometry import ContactGeometry
from .lib.hertzian_contact import HertzianContact
from .lib.force_data import ContactType, ForceData, ForceDataSequence
from .lib.modal_vertices import ModalVertices
from .lib.trajectory_data import tmpTrajectoryData, TrajectoryData
from .lib.collision_data import CollisionType, CollisionData
from .lib.particle_trajectory_data import ParticleTrajectoryData


__all__ = [
    'PositionSolver',
    'RotationSolver',
    'VertexSolver',
    'NormalSolver',
    'FlightPath',
    'DistanceSolver',
    'ForceSolver',
    'ForceSynth',
    'CollisionSolver',
    'ParticlesSolver',
    'physicsEngine',
    'ContactGeometry',
    'HertzianContact',
    'ContactType',
    'ForceData',
    'ForceDataSequence',
    'ModalVertices',
    'tmpTrajectoryData',
    'TrajectoryData',
    'ParticleTrajectoryData',
    'CollisionType',
    'CollisionData'
]
