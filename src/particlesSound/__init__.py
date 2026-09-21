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

__version__ = "0.1.3"
__author__ = "Malcom3D"
__description__ = "Physically plausible particles collision sound synthesis"

import os
import sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.particles_engine import particlesEngine
from .lib.surface_voxel_size import SurfaceVoxelSize
from .lib.surface_voxel_object import SurfaceVoxelObject
from .lib.particles_collisions_points import ParticlesCollisionsPoints
from .lib.particles_collisions_voxels import ParticlesCollisionsVoxels
from .lib.voxels_modal_ir_convolver import VoxelsModalIRConvolver

__all__ = [
    'particlesEngine',
    'SurfaceVoxelSize',
    'SurfaceVoxelObject',
    'ParticlesCollisionsPoints',
    'ParticlesCollisionsVoxels',
    'VoxelsModalIRConvolver'
]
