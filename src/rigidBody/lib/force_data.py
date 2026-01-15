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
from scipy.interpolate import CubicSpline

@dataclass
class ForceData:
    """Container for force event data."""
    frame: float # interpolated frame number
    obj1_idx: int
    obj2_idx: int
    relative_velocity: np.ndarray
    normal_velocity: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    normal_force_magnitude: np.ndarray
    tangential_force_magnitude: np.ndarray
    stochastic_normal_force: Optional[np.ndarray] = None       
    stochastic_tangential_force: Optional[np.ndarray] = None

@dataclass
class ForceDataSequence:
    """Container for forces sequences data."""
    frames: np.ndarray  # interpolated frame number
    obj1_idx: int
    other_obj_idx: np.ndarray
    relative_velocity: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_velocity: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_force: Tuple[CubicSpline, CubicSpline, CubicSpline]
    tangential_force: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_force_magnitude: CubicSpline
    tangential_force_magnitude: CubicSpline
    stochastic_normal_force: Optional[Tuple[CubicSpline, CubicSpline, CubicSpline]] = None       
    stochastic_tangential_force: Optional[Tuple[CubicSpline, CubicSpline, CubicSpline]] = None
