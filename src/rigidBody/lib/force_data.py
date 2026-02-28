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
import pickle
import numpy as np
from enum import IntEnum
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline

from ..lib.cubicspline_with_nan import CubicSplineWithNaN

class ContactType(IntEnum):
    """Enum for different Type of contact mechanics"""
    NO_CONTACT = 0
    IMPACT = 1
    SCRAPING = 2
    SLIDING = 3
    ROLLING = 4
    STATIC = 5

@dataclass
class ForceData:
    """Container for force event data."""
    frame: float # interpolated frame number
    obj1_idx: int
    obj2_idx: int
    restitution: np.ndarray
    relative_velocity: np.ndarray
    normal_velocity: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    tangential_velocity: np.ndarray
    normal_force_magnitude: np.ndarray
    tangential_force_magnitude: np.ndarray
    stochastic_normal_force: Optional[np.ndarray] = None       
    stochastic_tangential_force: Optional[np.ndarray] = None
    contact_type: int = None
    contact_point: np.ndarray = None # Mean contact point
    contact_radius: float = None # Mean contact point
    rolling_radius: float = None # Effective rolling radius (m)
    impact_duration: float = None
    contact_pressure: float = None # Average pressure (Pa)
    penetration_depth: float = None # Penetration depth (m)
    coupling_strength: float = None

@dataclass
class ForceDataSequence:
    """Container for forces sequences data."""
    frames: np.ndarray  # interpolated frame number
    obj_idx: int
    other_obj_idx: np.ndarray
    restitution: CubicSpline
    relative_velocity: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_velocity: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_force: Tuple[CubicSpline, CubicSpline, CubicSpline]
    tangential_force: Tuple[CubicSpline, CubicSpline, CubicSpline]
    tangential_velocity: Tuple[CubicSpline, CubicSpline, CubicSpline]
    normal_force_magnitude: CubicSpline
    tangential_force_magnitude: CubicSpline
    stochastic_normal_force: Optional[Tuple[CubicSpline, CubicSpline, CubicSpline]]            # = None
    stochastic_tangential_force: Optional[Tuple[CubicSpline, CubicSpline, CubicSpline]]        # = None
    contact_type: np.ndarray # array of ContactType enum value
    contact_point: Tuple[CubicSplineWithNaN, CubicSplineWithNaN, CubicSplineWithNaN]
    contact_radius: CubicSplineWithNaN
    rolling_radius: CubicSplineWithNaN
    impact_duration: np.ndarray
    contact_pressure: CubicSplineWithNaN
    penetration_depth: CubicSplineWithNaN
    coupling_strength: CubicSpline

    def get_contact_type(self, frame_idx: float):
        idx = np.where(self.frames == np.min(self.frames[0 < self.frames - frame_idx]))
        return self.contact_type[idx]

    def get_impact_duration(self, frame_idx: float):
        idx = (np.abs(self.frames - frame_idx)).argmin()
        return self.impact_duration[idx]

    def get_contact_point(self, frame_idx: float):
        return np.array([
            self.contact_point[0](frame_idx),
            self.contact_point[1](frame_idx),
            self.contact_point[2](frame_idx)
        ])

    def get_contact_radius(self, frame_idx: float):
        return self.contact_radius(frame_idx)

    def get_coupling_strength(self, frame_idx: float):
        return self.coupling_strength(frame_idx)

    def get_restitution(self, frame_idx: float):
        return self.restitution(frame_idx)

    def get_normal_force_magnitude(self, frame_idx: float):
        return self.normal_force_magnitude(frame_idx)

    def get_normal_force(self, frame_idx: float):
        return np.array([
            self.normal_force[0](frame_idx),
            self.normal_force[1](frame_idx),
            self.normal_force[2](frame_idx)
        ])
        
    def get_tangential_force_magnitude(self, frame_idx: float):
        return self.tangential_force_magnitude(frame_idx)

    def get_tangential_force(self, frame_idx: float):
        return np.array([
            self.tangential_force[0](frame_idx),
            self.tangential_force[1](frame_idx),
            self.tangential_force[2](frame_idx)
        ])

    def get_tangential_velocity(self, frame_idx: float):
        return np.array([
            self.tangential_velocity[0](frame_idx),
            self.tangential_velocity[1](frame_idx),
            self.tangential_velocity[2](frame_idx)
        ])

    def get_normal_velocity(self, frame_idx: float):
        return np.array([
            self.normal_velocity[0](frame_idx),
            self.normal_velocity[1](frame_idx),
            self.normal_velocity[2](frame_idx)
        ])
        
    def get_relative_velocity(self, frame_idx: float):
        return np.array([
            self.relative_velocity[0](frame_idx),
            self.relative_velocity[1](frame_idx),
            self.relative_velocity[2](frame_idx)
        ])

    def get_stochastic_normal_force(self, frame_idx: float):
        return np.array([
            self.stochastic_normal_force[0](frame_idx),
            self.stochastic_normal_force[1](frame_idx),
            self.stochastic_normal_force[2](frame_idx)
        ])

    def get_stochastic_tangential_force(self, frame_idx: float):
        return np.array([
            self.stochastic_tangential_force[0](frame_idx),
            self.stochastic_tangential_force[1](frame_idx),
            self.stochastic_tangential_force[2](frame_idx)
        ])

    def save(self, filepath: str) -> None:
        """
        Save the ForceDataSequence object to a file using pickle.
        
        Parameters
        ----------
        filepath : str
            Path to the file where the object should be saved.
            If the directory doesn't exist, it will be created.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Save the object using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ForceDataSequence':
        """
        Load a ForceDataSequence object from a pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to the pickle file containing the saved object.
            
        Returns
        -------
        ForceDataSequence
            The loaded ForceDataSequence object.
            
        Raises
        ------
        FileNotFoundError
            If the specified file doesn't exist.
        pickle.UnpicklingError
            If the file cannot be unpickled or doesn't contain a valid object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        # Verify that the loaded object is of the correct type
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        
        return obj
