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
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class PhysicalProperties:
    """
    Container for estimated physical properties from audio analysis.
    """
    young_modulus: float  # Young's modulus (Pa)
    poisson_ratio: float  # Poisson's ratio
    density: float  # Density (kg/m³)
    damping: float  # Damping ratio
    sound_speed: float  # Speed of sound in material (m/s)
    
    # Additional properties
    stiffness: float = None  # Material stiffness
    impedance: float = None  # Acoustic impedance
    quality_factor: float = None  # Overall quality factor
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'damping': self.damping,
            'sound_speed': self.sound_speed,
            'stiffness': self.stiffness,
            'impedance': self.impedance,
            'quality_factor': self.quality_factor
        }

class PhysicalPropertiesEstimator:
    """
    Estimate physical properties of materials from audio analysis.
    
    Uses modal analysis results to estimate:
    - Young's modulus from frequency ratios
    - Poisson's ratio from mode frequency ratios
    - Density from mass and volume
    - Damping from T60 values
    - Sound speed from material properties
    """
    
    def __init__(self, entity_manager: EntityManager):
        """
        Initialize the physical properties estimator.
        
        Parameters:
        -----------
        entity_manager : EntityManager
            Entity manager for accessing configuration
        """
        self.entity_manager = entity_manager
        self.config = entity_manager.get('config')
        
        set_debug(self.config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        # Material property ranges for validation
        self.material_ranges = {
            'young_modulus': (1e6, 1e12),  # 1 MPa to 1 TPa
            'poisson_ratio': (0.1, 0.5),   # 0.1 to 0.5
            'density': (100, 20000),       # 100 kg/m³ to 20000 kg/m³
            'damping': (0.001, 0.1),       # 0.1% to 10%
            'sound_speed': (100, 10000)    # 100 m/s to 10000 m/s
        }
    
    def estimate_from_modal_params(self, frequencies: np.ndarray, 
                                   t60s: np.ndarray,
                                   damping_ratios: np.ndarray,
                                   volume: float = None,
                                   mass: float = None) -> PhysicalProperties:
        """
        Estimate physical properties from modal analysis results.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Modal frequencies (Hz)
        t60s : np.ndarray
            T60 reverberation times (seconds)
        damping_ratios : np.ndarray
            Damping ratios
        volume : float, optional
            Object volume (m³)
        mass : float, optional
            Object mass (kg)
            
        Returns:
        --------
        PhysicalProperties
            Estimated physical properties
        """
        if len(frequencies) < 2:
            debug_print("Need at least 2 modes for property estimation")
            return self._get_default_properties()
        
        # Estimate sound speed from frequency ratios
        # For a simple bar: f_n = (c/2L) * n
        # For a plate: f_n ∝ c * (n/L)
        # We'll use the fundamental frequency to estimate
        
        # Get fundamental frequency (lowest mode)
        f0 = frequencies[0]
        
        # Estimate sound speed
        # Assume characteristic dimension of 0.1m (typical object size)
        char_dim = 0.1
        sound_speed = 2 * f0 * char_dim
        
        # Clamp to reasonable range
        sound_speed = np.clip(sound_speed, 100, 10000)
        
        # Estimate density from mass and volume
        if mass is not None and volume is not None and volume > 0:
            density = mass / volume
        else:
            density = 1000.0  # Default: water-like density
        
        # Clamp density
        density = np.clip(density, 100, 20000)
        
        # Estimate Young's modulus from sound speed and density
        # c = sqrt(E/ρ) for longitudinal waves
        young_modulus = sound_speed**2 * density
        
        # Clamp Young's modulus
        young_modulus = np.clip(young_modulus, 1e6, 1e12)
        
        # Estimate Poisson's ratio from mode frequency ratios
        # For a plate: f_21/f_11 ratio depends on ν
        # Simplified: use frequency spacing
        freq_ratios = frequencies[1:] / frequencies[:-1]
        avg_ratio = np.mean(freq_ratios)
        
        # Map frequency ratio to Poisson's ratio
        # This is a simplified mapping
        if avg_ratio > 2.5:
            poisson_ratio = 0.1  # Very stiff
        elif avg_ratio > 2.0:
            poisson_ratio = 0.2  # Stiff
        elif avg_ratio > 1.7:
            poisson_ratio = 0.3  # Average
        elif avg_ratio > 1.5:
            poisson_ratio = 0.4  # Ductile
        else:
            poisson_ratio = 0.45  # Very ductile
        
        # Clamp Poisson's ratio
        poisson_ratio = np.clip(poisson_ratio, 0.1, 0.5)
        
        # Estimate damping from T60 values
        # T60 = 6.9078 / (ζ * ω_n)
        # ζ = 6.9078 / (T60 * ω_n)
        if len(t60s) > 0 and len(damping_ratios) > 0:
            # Use average damping ratio
            damping = np.mean(damping_ratios)
        else:
            # Estimate from T60
            omega_n = 2 * np.pi * f0
            damping = 6.9078 / (np.mean(t60s) * omega_n)
        
        # Clamp damping
        damping = np.clip(damping, 0.001, 0.1)
        
        # Compute derived properties
        stiffness = young_modulus
        
        # Acoustic impedance
        impedance = density * sound_speed
        
        # Overall quality factor
        quality_factor = 1.0 / (2.0 * damping)
        
        return PhysicalProperties(
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
            damping=damping,
            sound_speed=sound_speed,
            stiffness=stiffness,
            impedance=impedance,
            quality_factor=quality_factor
        )
    
    def estimate_from_frequency_response(self, frequencies: np.ndarray,
                                        magnitude: np.ndarray,
                                        volume: float = None) -> PhysicalProperties:
        """
        Estimate physical properties from frequency response data.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency axis (Hz)
        magnitude : np.ndarray
            Magnitude response (linear scale)
        volume : float, optional
            Object volume (m³)
            
        Returns:
        --------
        PhysicalProperties
            Estimated physical properties
        """
        # Find peaks in frequency response
        from scipy.signal import find_peaks
        
        # Convert to dB for peak detection
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        # Find peaks
        peaks, properties = find_peaks(
            magnitude_db,
            height=np.max(magnitude_db) - 40,  # -40 dB threshold
            distance=10
        )
        
        if len(peaks) < 2:
            debug_print("Not enough peaks for property estimation")
            return self._get_default_properties()
        
        # Extract modal frequencies
        modal_freqs = frequencies[peaks]
        
        # Extract T60 from peak widths
        t60s = []
        for peak_idx in peaks:
            # Find half-power bandwidth
            half_power = magnitude_db[peak_idx] - 3  # -3 dB
            # Find left and right crossing points
            left_idx = peak_idx
            while left_idx > 0 and magnitude_db[left_idx] > half_power:
                left_idx -= 1
            right_idx = peak_idx
            while right_idx < len(magnitude_db) - 1 and magnitude_db[right_idx] > half_power:
                right_idx += 1
            
            # Bandwidth
            bandwidth = frequencies[right_idx] - frequencies[left_idx]
            
            # Q factor = f0 / bandwidth
            if bandwidth > 0:
                Q = frequencies[peak_idx] / bandwidth
                # T60 = 6.9078 / ( (2π * f * ζ) = 6.9078 * Q / (π * f)
                t60 = 6.9078 * Q / (np.pi * frequencies[peak_idx])
                t60s.append(t60)
        
        # Use modal analysis for estimation
        return self.estimate_from_modal_params(
            frequencies=modal_freqs,
            t60s=np.array(t60s),
            damping_ratios=np.array([1.0 / (2.0 * frequencies[peaks[i]] * t60s[i] / 6.9078) 
                                    for i in range(len(peaks))]),
            volume=volume
        )
    
    def _get_default_properties(self) -> PhysicalProperties:
        """Get default physical properties."""
        return PhysicalProperties(
            young_modulus=1e9,
            poisson_ratio=0.3,
            density=1000.0,
            damping=0.02,
            sound_speed=1000.0,
            stiffness=1e9,
            impedance=1e6,
            quality_factor=25.0
        )
    
    def validate(self, properties: PhysicalProperties) -> bool:
        """
        Validate estimated physical properties against known ranges.
        
        Parameters:
        -----------
        properties : PhysicalProperties
            Properties to validate
            
        Returns:
        --------
        bool
            True if all properties are within valid ranges
        """
        valid = True
        
        if not (self.material_ranges['young_modulus'][0] <= properties.young_modulus <= 
                self.material_ranges['young_modulus'][1]):
            debug_print(f"Warning: Young's modulus {properties.young_modulus} out of range")
            valid = False
        
        if not (self.material_ranges['poisson_ratio'][0] <= properties.poisson_ratio <= 
                self.material_ranges['poisson_ratio'][1]):
            debug_print(f"Warning: Poisson's ratio {properties.poisson_ratio} out of range")
            valid = False
        
        if not (self.material_ranges['density'][0] <= properties.density <= 
                self.material_ranges['density'][1]):
            debug_print(f"Warning: Density {properties.density} out of range")
            valid = False
        
        if not (self.material_ranges['damping'][0] <= properties.damping <= 
                self.material_ranges['damping'][1]):
            debug_print(f"Warning: Damping {properties.damping} out of range")
            valid = False
        
        if not (self.material_ranges['sound_speed'][0] <= properties.sound_speed <= 
                self.material_ranges['sound_speed'][1]):
            debug_print(f"Warning: Sound speed {properties.sound_speed} out of range")
            valid = False
        
        return valid
