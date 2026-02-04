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

import numpy as np
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclasslass, field
from scipy import signal
import warnings

from ..core.entity_manager import EntityManager

@dataclass
class CouplingStrength:
    entity_manager: EntityManager
    # Configuration parameters
    velocity_weight: float = 0.3
    force_weight: float = 0.25
    material_weight: float = 0.2
    geometry_weight: float = 0.15
    acoustic_weight: float = 0.1
    # Smoothing parameters
    smoothing_window: int = 5
    # Thresholds
    min_velocity_for_coupling: float = 0.01  # m/s
    min_force_for_coupling: float = 0.1  # N

    def __post_init__(self):
        """Initialize history buffers for smoothing."""
        self.coupling_history = []
        self.velocity_history = []
        self.force_history = []

    def compute(self, sample_idx: int, obj1_idx: int, obj2_idx: int, hertzian_data: Optional[Dict] = None) -> float:
        """
        Evaluates coupling strength (0-1) between colliding objects.
    
        Coupling strength represents how strongly two objects are acoustically coupled
        during collision/contact, considering both mechanical and acoustic factors.
        
        Parameters:
        -----------
        sample_idx : int
            Current sample index
        obj1_idx : int
            Index of first object
        obj2_idx : int
            Index of second object
        hertzian_data : Optional[Dict]
            Hertzian contact analysis data if available
            
        Returns:
        --------
        float : Coupling strength between 0 (no coupling) and 1 (strong coupling)
        """
        config = self.entity_manager.get('config')
        forces = self.entity_manager.get('forces')
        
        # Get object configurations
        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
            elif conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
        
        # Get force data
        force_data = None
        for f_idx in forces.keys():
            if forces[f_idx].obj_idx == obj1_idx and obj2_idx in forces[f_idx].other_obj_idx:
                force_data = forces[f_idx]
                break
        
#        if force_data is None:
#            # Try the reverse direction
#            for f_idx in forces.keys():
#                if forces[f_idx].obj_idx == obj2_idx and obj1_idx in forces[f_idx].other_obj_idx:
#                    force_data = forces[f_idx]
#                    break
        
        # Calculate individual coupling factors
        velocity_factor = self._calculate_velocity_factor(sample_idx, force_data)
        force_factor = self._calculate_force_factor(sample_idx, force_data)
        material_factor = self._calculate_material_factor(config_obj1, config_obj2)
        geometry_factor = self._calculate_geometry_factor(config_obj1, config_obj2, hertzian_data)
        acoustic_factor = self._calculate_acoustic_factor(config_obj1, config_obj2)
        
        # Combine factors with weights
        raw_coupling = (
            self.velocity_weight * velocity_factor +
            self.force_weight * force_factor +
            self.material_weight * material_factor +
            self.geometry_weight * geometry_factor +
            self.acoustic_weight * acoustic_factor
        )
        
        # Apply thresholds
        if velocity_factor < 0..1 and force_factor < 0.1:
            raw_coupling *= 0.5  # Reduce coupling for very weak interactions
        
        # Smooth over time
        smoothed_coupling = self._apply_smoothing(raw_coupling)
        
        # Ensure output is in [0, 1] range
        coupling_strength = np.clip(smoothed_coupling, 0.0, 1.0)
        
        return coupling_strength
    
    def _calculate_velocity_factor(self, sample_idx: int, force_data) -> float:
        """
        Calculate coupling factor based on relative velocity.
        
        Higher relative velocities generally lead to stronger coupling.
        """
        relative_velocity = force_data.get_relative_velocity(sample_idx)
        velocity_magnitude = np.linalg.norm(relative_velocity)
        
        # Store for history
        self.velocity_history.append(velocity_magnitude)
        if len(self.velocity_history) > self.smoothing_window:
            self.velocity_history_history.pop(0)
        
        # Normalize velocity (assuming typical range 0-10 m/s for impact sounds)
        # Use logarithmic scaling as coupling increases with log(velocity)
        if velocity_magnitude < self.min_velocity_for_coupling:
            return 0.0
        
        # Sigmoid function for velocity factor
        v_norm = velocity_magnitude / 5.0  # Normalize to typical range
        velocity_factor = 1.0 / (1.0 + np.exp(-3.0 * (v_norm - 0.5)))
        
        return float(velocity_factor)
    
    def _calculate_force_factor(self, sample_idx: int, force_data) -> float:
        """
        Calculate coupling factor based on contact forces.
        
        Higher normal forces generally lead to stronger coupling.
        """
        normal_force_magnitude = force_data.get_normal_force_magnitude(sample_idx)
        tangential_force_magnitude = force_data.get_tangential_force_magnitude(sample_idx)
        
        # Store for history
        total_force = normal_force_magnitude + 0.3 * tangential_force_magnitude
        self.force_history.append(total_force)
        if len(self.force_history) > self.smoothing_window:
            self.force_history.pop(0)
        
        if total_force < self.min_force_for_coupling:
            return 0.0
        
        # Normalize force (assuming typical range 0-1000 N for impact sounds)
        # Use square root scaling as coupling increases with sqrt(force)
        f_norm = np.sqrt(total_force / 1000.0)
        force_factor = np.clip(f_norm, 0.0, 1.0)
        
        return float(force_factor)
    
    def _calculate_material_factor(self, config_obj1, config_obj2) -> float:
        """
        Calculate coupling factor based on material properties.
        
        Similar materials couple better than dissimilar ones.
        """
        # Get material properties
        young1 = config_obj1.acoustic_shader.young_modulus
        young2 = config_obj2.acoustic_shader.young_modulus
        density1 = config_obj1.acoustic_shader.density
        density2 = config_obj2.acoustic_shader.density
        
        # Calculate impedance mismatch
        # Acoustic impedance Z = density * speed_of_sound
        # Speed of sound in solid: c = sqrt(E/ρ) for longitudinal waves
        c1 = np.sqrt(young1 / density1) if young1 > 0 and density1 > 0 else 1.0
        c2 = np.sqrt(young2 / density2) if young2 > 0 and density2 > 0 else 1.0
        
        Z1 = density1 * c1
        Z2 = density2 * c2
        
        # Transmission coefficient for normal incidence
        # T = 4*Z1*Z2 / (Z1 + Z2)^2
        if Z1 + Z2 > 0:
            transmission = 4 * Z1 * Z2 / (Z1 + Z2)**2
        else:
            transmission = 0.0
        
        # Material similarity factor (0-1)
        # Similar materials have transmission close to 1
        material_factor = np.clip(transmission, 0.0, 1.0)
        
        return float(material_factor)
    
    def _calculate_geometry_factor(self, config_obj1, config_obj2, 
                                 hertzian_data: Optional[Dict] = None) -> float:
        """
        Calculate coupling factor based on contact geometry.
        
        Larger contact areas and conforming geometries lead to better coupling.
        """
        # Use hertzian data
        try:
            # Contact area factor
            contact_area = hertzian_data.get('contact_area', 0.0)
            max_expected_area = 0.01  # 100 cm² as reasonable maximum
            area_factor = np.clip(contact_area / max_expected_area, 0.0, 1.0)
            
            # Pen Penetration depth factor
            penetration = hertzian_data.get('penetration_depth', 0.0)
            max_penetration = 0.01  # 1 cm as reasonable maximum
            penetration_factor = np.clip(penetration / max_penetration, 0.0, 1.0)
            
            # Contact stiffness factor (softer contact = better coupling)
            stiffness = hertzian_data.get('contact_stiffness', 1e9)
            min_stiffness = 1e6  # 1 MN/m
            max_stiffness = 1e10  # 10 GN/m
            stiffness_norm = (np.log10(stiffness) - np.log10(min_stiffness)) / (np.log10(max_stiffness) - np.log10(min_stiffness))
            stiffness_factor = 1.0 - np.clip(stiffness_norm, 0.0, 1.0)
            
            # Combine geometry factors
            geometry_factor = 0.4 * area_factor + 0.3 * penetration_factor + 0.3 * stiffness_factor
            
            return float(geometry_factor)
            
        except (KeyError, AttributeError):
            return 0.5
    
    def _calculate_acoustic_factor(self, config_obj1, config_obj2) -> float:
        """
        Calculate coupling factor based on acoustic properties.
        
        Objects with similar acoustic properties couple better.
        """
        # Get acoustic properties
        props1 = config_obj1.acoustic_shader.acoustic_properties
        props2 = config_obj2.acoustic_shader.acoustic_properties
        
        if props1 is None or props2 is None:
            return 0.5
        
        # Compare absorption coefficients if available
        absorption_similarity = 0.5
        if props1.absorption is not None and props2.absorption is not None:
            try:
                               # Get average absorption in audible range
                freq1, abs1 = props1.absorption.get_avg_coeffs(20, 20000)
                freq2, abs2 = props2.absorption.get_avg_coeffs(20, 20000)
                
                if abs1 > 0 and abs2 > 0:
                    absorption_ratio = min(abs1, abs2) / max(abs1, abs2)
                    absorption_similarity = np.clip(absorption_ratio, 0.0, 1.0)
            except:
                pass
        
        # Compare other acoustic properties
        # For now, use a simple average
        acoustic_factor = absorption_similarity
        
        return float(acoustic_factor)
    
    def _apply_smoothing(self, raw_coupling: float) -> float:
        """
        Apply temporal smoothing to coupling strength.
        
        This prevents rapid fluctuations in coupling strength.
        """
        self.coupling_history.append(raw_coupling)
        
        if len(self.coupling_history) > self.smoothing_window:
            self.coupling_history.pop(0)
        
        # Use weighted moving average (more weight to recent values)
        weights = np.linspace(0.5, 1.5, len(self.coupling_history))
        weights = weights / np.sum(weights)
        
        smoothed = np.average(self.coupling_history, weights=weights)
        
        return float(smoothed)
