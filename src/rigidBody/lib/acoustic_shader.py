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

from dataclasses import dataclass, field
from typing import Union, Optional, Any
from typing import List
import numpy as np

from ..lib.interpolator import FrequencyInterpolator

@dataclass
class AcousticCoefficients:
    """Represents frequency-dependent coefficients using numpy arrays."""
    frequencies: np.ndarray  # Frequency values array
    coefficients: np.ndarray  # Corresponding coefficient values array

    def __post_init__(self):
        # Create interpolator
        self.coeffs_interpolator = FrequencyInterpolator(self.frequencies, self.coefficients, method='cubic')

    def get_coeffs(self, low_freq: Optional[float] = None, high_freq: Optional[float] = None, num_points: Optional[int] = 0) -> np.ndarray:
        low_freq = low_freq if low_freq else self.frequencies[0]
        high_freq = high_freq if high_freq else self.frequencies[-1]
        num_points = num_points if not num_points == 0 else len(self.frequencies)
        frequencies, coeffs = self.coeffs_interpolator.interpolate_band(low_freq, high_freq, num_points)
        return frequencies, coeffs

    def get_avg_coeffs(self, low_freq: Optional[float] = None, high_freq: Optional[float] = None) -> np.ndarray:
        low_freq = low_freq if low_freq else self.frequencies[0]
        high_freq = high_freq if high_freq else self.frequencies[-1]
        return self.coeffs_interpolator.get_band_average(low_freq, high_freq)

@dataclass
class AcousticProperties:
    """Container for acoustic properties."""
    absorption: Optional[AcousticCoefficients] = None
    refraction: Optional[AcousticCoefficients] = None
    reflection: Optional[AcousticCoefficients] = None
    scattering: Optional[AcousticCoefficients] = None

@dataclass
class AcousticShader:
    sound_speed: float = 343.0  # m/s
    density: float = 1.2        # kg/m³
    young_modulus: float = None
    poisson_ratio: float = None
    density: float = None
    damping: float = None
    friction: float = None
    low_frequency: float = 1.0
    high_frequency: float = 24000.0
    acoustic_properties: Optional[AcousticProperties] = field(default_factory=AcousticProperties)

    def get_data(self, properties: Union[List[str], str]) -> AcousticCoefficients:
        """Retrieve data for one or more acoustic properties."""
        for property in properties:
            if property in self.acoustic_properties:
                return self.acoustic_properties[property]
        return None
