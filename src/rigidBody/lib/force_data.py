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

import os
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

@dataclass
class ForceData:
    """Container for force event data."""
    frame: float # interpolated frame number
    obj1_idx: int
    obj2_idx: int
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [ωx, ωy, ωz]
    linear_acceleration: np.ndarray  # [ax, ay, az]
    angular_acceleration: np.ndarray  # [αx, αy, αz]
    relative_velocity: np.ndarray
    normal_velocity: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    normal_force_magnitude: np.ndarray
    tangential_force_magnitude: np.ndarray
    stochastic_normal_force: Optional[np.ndarray] = None       
    stochastic_tangential_force: Optional[np.ndarray] = None
