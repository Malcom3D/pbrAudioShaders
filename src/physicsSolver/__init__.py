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
__description__ = "Reverse physics engine"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.entity_manager import EntityManager
from .core.position_solver import PositionSolver
from .core.rotation_solver import RotationSolver
from .core.vertex_solver import VertexSolver
from .core.normal_solver import NormalSolver
from .core.flight_path import FlightPath
from .core.distance_solver import DistanceSolver
from .core.force_solver import ForceSolver
from .core.force_synth import ForceSynth
from .core.collision_solver import CollisionSolver
from .core.physics_engine import physicsEngine
from .lib.score_data import ScoreEvent, ScoreTrack
from .lib.contact_geometry import ContactGeometry
from .lib.hertzian_contact import HertzianContact
from .lib.cubicspline_with_nan import CubicSplineWithNaN
from .lib.force_data import ContactType, ForceData, ForceDataSequence
from .lib.acoustic_shader import AcousticCoefficients, AcousticProperties, AcousticShader
from .lib.modal_vertices import ModalVertices
from .lib.interpolator import FrequencyInterpolator, Frequency3DInterpolator
from .lib.trajectory_data import tmpTrajectoryData, TrajectoryData
from .lib.collision_data import CollisionType, CollisionData
from .utils.config import SystemConfig, ObjectConfig, Config

__all__ = [
    'EntityManager',
    'PositionSolver',
    'RotationSolver',
    'VertexSolver',
    'NormalSolver',
    'FlightPath',
    'DistanceSolver',
    'ForceSolver',
    'ForceSynth',
    'CollisionSolver',
    'physicsEngine',
    'ScoreEvent',
    'ScoreTrack',
    'ContactGeometry',
    'HertzianContact',
    'CubicSplineWithNaN',
    'ContactType',
    'ForceData',
    'ForceDataSequence',
    'AcousticCoefficients',
    'AcousticProperties',
    'AcousticShader',
    'ModalVertices',
    'FrequencyInterpolator',
    'Frequency3DInterpolator',
    'tmpTrajectoryData',
    'TrajectoryData',
    'CollisionType',
    'CollisionData',
    'SystemConfig',
    'ObjectConfig',
    'Config'
]
