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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..lib.primitive_geometry import PrimitiveGeometry
from ..lib.shape_properties import ShapeType, ShapeProperties

@dataclass
class ModalParameters:
    """Parameters for approximate modal model."""
    frequencies: np.ndarray  # Modal frequencies (Hz)
    t60s: np.ndarray  # Decay times (seconds)
    gains: np.ndarray  # Mode gains (n_modes x n_vertices)
    n_modes: int
    n_vertices: int

@dataclass
class Approx2Faust:
    """
    Generate approximate modal models for primitive shapes.
    Uses analytical solutions for simple geometries.
    """
    
    def __post_init__(self):
        self.primitive_geometry = PrimitiveGeometry()
    
    def compute(self, vertices: np.ndarray, faces: np.ndarray, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Compute approximate modal parameters for a mesh.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex positions (N, 3)
        faces : np.ndarray
            Face indices (M, 3)
        young_modulus : float
            Young's modulus (Pa)
        poisson_ratio : float
            Poisson's ratio
        density : float
            Density (kg/m³)
        damping : float
            Damping ratio ratio
        min_freq : float
            Minimum frequency (Hz)
        max_freq : float
            Maximum frequency (Hz)
        n_modes : int
            Number of modes to generate
            
        Returns:
        --------
        ModalParameters
            Generated modal parameters
        """
        # Classify the shape
        shape_props = self.primitive_geometry.classify(vertices, faces)
        
        # Get characteristic dimension
        char_dim = self.primitive_geometry.get_characteristic_dimension(shape_props)
        
        # Generate modal parameters based on shape type
        if shape_props.shape_type == ShapeType.SPHERE:
            return self._sphere_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
        elif shape_props.shape_type == ShapeType.CUBE:
            return self._cube_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
        elif shape_props.shape_type == ShapeType.CYLINDER:
            return self._cylinder_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
        elif shape_props.shape_type == ShapeType.PLATE:
            return self._plate_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
        elif shape_props.shape_type == ShapeType.BEAM:
            return self._beam_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
        else:
            # Fallback: use equivalent sphere
            return self._sphere_modes(vertices, shape_props, young_modulus, poisson_ratio, density, damping, min_freq, max_freq, n_modes)
    
    def _compute_wave_speeds(self, young_modulus: float, poisson_ratio: float, density: float) -> Tuple[float, float]:
        """
        Compute longitudinal and shear wave speeds.
        
        Returns:
            (c_longitudinal, c_shear) in m/s
        """
        if young_modulus <= 0 or density <= 0:
            return 343.0, 200.0  # Default values
        
        # Longitudinal wave speed
        c_long = np.sqrt(young_modulus * (1 - poisson_ratio) / (density * (1 + poisson_ratio) * (1 - 2 * poisson_ratio)))
        
        # Shear wave speed
        c_shear = np.sqrt(young_modulus / (2 * density * (1 + poisson_ratio)))
        
        return c_long, c_shear
    
    def _sphere_modes(self, vertices: np.ndarray, shape_props: ShapeProperties, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Generate modes for a sphere.
        
        Uses analytical solution for sphere vibrations:
        - Spheroidal modes (radial and torsional)
        - Frequencies proportional to (n+1)(n+2)/2 for mode order n
        """
        radius = shape_props.dimensions.get('radius', 0.1)
        c_long, c_shear = self._compute_wave_speeds(young_modulus, poisson_ratio, density)
        
        n_vertices = len(vertices)
        
        # Generate mode frequencies
        frequencies = []
        
        # Spheroidal modes: n=0 (breathing), n=1 (dipole), n=2 (quadrupole), etc.
        # Frequency scaling: f_n ∝ c_shear / R * sqrt(n(n+1))
        for n in range(1, 20):  # Mode order
            for l in range(0, n + 1):  # Angular momentum
                # Fundamental frequency for this mode
                f_base = c_shear / (2 * np.pi * radius) * np.sqrt(n * (n + 1))
                
                # Skip if below minimum frequency
                if f_base < min_freq:
                    continue
                
                # Skip if above maximum frequency
                if f_base > max_freq:
                    break
                
                frequencies.append(f_base)
                
                # Add overtones
                for k in range(1, 4):
                    f_ot = f_base * (1 + k * 0.5)
                    if f_ot <= max_freq:
                        frequencies.append(f_ot)
                
                if len(frequencies) >= n_modes * 2:
                    break
            if len(frequencies) >= n_modes * 2:
                break
        
        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]
        
        # Pad if we don't have enough modes
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))
        
        # Compute T60 values (decay times)
        t60s = self._compute_t60s(frequencies, damping, shape_props)
        
        # Compute gains for each vertex
        gains = self._compute_sphere_gains(vertices, shape_props, frequencies)
        
        return ModalParameters(frequencies=frequencies, t60s=t60s, gains=gains, n_modes=n_modes, n_vertices=n_vertices)
    
    def _cube_modes(self, vertices: np.ndarray, shape_props: ShapeProperties, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Generate modes for a cube.
        
        Uses analytical solution for rectangular parallelepiped:
        - Modes are combinations of 1D standing waves in each dimension
        - f_ijk = c/2 * sqrt((i/Lx)² + (j/Ly)² + (k/Lz)²)
        """
        side = shape_props.dimensions.get('side', 0.1)
        c_long, c_shear = self._compute_wave_speeds(young_modulus, poisson_ratio, density)
        
        n_vertices = len(vertices)
        
        # Generate mode frequencies
        frequencies = []
        
        # Mode indices (i, j, k) for standing waves
        for i in range(0, 6):
            for j in range(0, 6):
                for k in range(0, 6):
                    if i == 0 and j == 0 and k == 0:
                        continue  # Skip rigid body mode
                    
                    # Frequency for this mode
                    f = c_long / 2 * np.sqrt((i/side)**2 + (j/side)**2 + (k/side)**2)
                    
                    if f < min_freq:
                        continue
                    if f > max_freq:
                        break
                    
                    frequencies.append(f)
                    
                    if len(frequencies) >= n_modes * 2:
                        break
                if len(frequencies) >= n_modes * 2:
                    break
            if len(frequencies) >= n_modes * 2:
                break
        
        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]
        
        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))
        
        # Compute T60 values
        t60s = self._compute_t60s(frequencies, damping, shape_props)
        
        # Compute gains
        gains = self._compute_cube_gains(vertices, shape_props, frequencies)
        
        return ModalParameters(frequencies=frequencies, t60s=t60s, gains=gains, n_modes=n_modes, n_vertices=n_vertices)
    
    def _cylinder_modes(self, vertices: np.ndarray, shape_props: ShapeProperties, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Generate modes for a cylinder.
        
        Combines:
        - Radial modes (Bessel functions)
        - Longitudinal modes (standing waves along axis)
        - Torsional modes
        """
        radius = shape_props.dimensions.get('radius', 0.05)
        height = shape_props.dimensions.get('height', 0.1)
        c_long, c_shear = self._compute_wave_speeds(young_modulus, poisson_ratio, density)
        
        n_vertices = len(vertices)
        
        # Generate mode frequencies
        frequencies = []
        
        # Bessel function zeros for radial modes
        bessel_zeros = [3.832, 7.016, 10.173, 13.324, 16.471]  # J₁'(x) = 0
        
        # Longitudinal modes
        for n in range(1, 5):  # Longitudinal mode order
            f_long = n * c_long / (2 * height)
            
            if f_long < min_freq:
                continue
            if f_long > max_freq:
                break
            
            frequencies.append(f_long)
            
            # Add radial overtones
            for m, bz in enumerate(bessel_zeros[:3]):
                f_radial = bz * c_shear / (2 * np.pi * radius)
                f_combined = np.sqrt(f_long**2 + f_radial**2)
                
                if f_combined <= max_freq:
                    frequencies.append(f_combined)
        
        # Torsional modes
        for n in range(1, 4):
            f_torsion = n * c_shear / (2 * height)
            if min_freq <= f_torsion <= max_freq:
                frequencies.append(f_torsion)
        
        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]
        
        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))
        
        # Compute T60 values
        t60s = self._compute_t60s(frequencies, damping, shape_props)
        
        # Compute gains
        gains = self._compute_cylinder_gains(vertices, shape_props, frequencies)
        
        return ModalParameters(frequencies=frequencies, t60s=t60s, gains=gains, n_modes=n_modes, n_vertices=n_vertices)
    
    def _plate_modes(self, vertices: np.ndarray, shape_props: ShapeProperties, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Generate modes for a thin plate.
        
        Uses Kirchhoff plate theory:
        - f_mn = (π/2) * sqrt(D/ρh) * ((m/Lx)² + (n/Ly)²)
        - D = E*h³/(12*(1-ν²))
        """
        length = shape_props.dimensions.get('length', 0.2)
        width = shape_props.dimensions.get('width', 0.15)
        thickness = shape_props.dimensions.get('thickness', 0.01)
        
        n_vertices = len(vertices)
        
        # Bending stiffness
        D = (young_modulus * thickness**3) / (12 * (1 - poisson_ratio**2))
        
        # Mass per unit area
        rho_h = density * thickness
        
        # Generate mode frequencies
        frequencies = []
        
        for m in range(1, 8):  # Mode order in x
            for n in range(1, 8):  # Mode order in y
                f = (np.pi / 2) * np.sqrt(D / rho_h) * ((m/length)**2 + (n/width)**2)
                
                if f < min_freq:
                    continue
                if f > max_freq:
                    break
                
                frequencies.append(f)
                
                if len(frequencies) >= n_modes * 2:
                    break
            if len(frequencies) >= n_modes * 2:
                break
        
        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]
        
        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))
        
        # Compute T60 values
        t60s = self._compute_t60s(frequencies, damping, shape_props)
        
        # Compute gains
        gains = self._compute_plate_gains(vertices, shape_props, frequencies)
        
        return ModalParameters(frequencies=frequencies, t60s=t60s, gains=gains, n_modes=n_modes, n_vertices=n_vertices)
    
    def _beam_modes(self, vertices: np.ndarray, shape_props: ShapeProperties, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int) -> ModalParameters:
        """
        Generate modes for a beam.
        
        Uses Euler-Bernoulli beam theory:
        - f_n = (β_n²/2πL²) * sqrt(EI/ρA)
        - β_n: eigenvalues for clamped-free beam
        """
        length = shape_props.dimensions.get('length', 0.2)
        width = shape_props.dimensions.get('width', 0.02)
        depth = shape_props.dimensions.get('depth', 0.02)
        
        n_vertices = len(vertices)
        
        # Cross-sectional area and moment of inertia
        A = width * depth
        I = width * depth**3 / 12  # Bending about neutral axis
        
        # Eigenvalues for clamped-free beam
        beta_L = [1.875, 4.694, 7.855, 10.996, 14.137, 17.279]
        
        # Generate mode frequencies
        frequencies = []
        
        for n, beta in enumerate(beta_L):
            f = (beta**2 / (2 * np.pi * length**2)) * np.sqrt(young_modulus * I / (density * A))
            
            if f < min_freq:
                continue
            if f > max_freq:
                break
            
            frequencies.append(f)
            
            # Add torsional modes
            if n > 0:
                f_torsion = f * 1.5  # Approximate torsional frequency
                if f_torsion <= max_freq:
                    frequencies.append(f_torsion)
        
        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]
        
        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))
        
        # Compute T60 values
        t60s = self._compute_t60s(frequencies, damping, shape_props)
        
        # Compute gains
        gains = self._compute_beam_gains(vertices, shape_props, frequencies)
        
        return ModalParameters(frequencies=frequencies, t60s=t60s, gains=gains, n_modes=n_modes, n_vertices=n_vertices)
    
    def _compute_t60s(self, frequencies: np.ndarray, damping: float, shape_props: ShapeProperties) -> np.ndarray:
        """
        Compute T60 (reverberation time) for each mode.
        
        T60 = 3 * ln(10) / (π * damping * f)
        """
        if damping <= 0:
            damping = 0.02  # Default conservative value
        
        t60s = 3 * np.log(10) / (np.pi * damping * np.maximum(frequencies, 1))
        
        # Scale by shape-dependent factor
        if shape_props.shape_type == ShapeType.SPHERE:
            t60s *= 1.2  # Spheres have longer decay
        elif shape_props.shape_type == ShapeType.PLATE:
            t60s *= 0.8  # Plates have shorter decay
        elif shape_props.shape_type == ShapeType.BEAM:
            t60s *= 0.9  # Beams have slightly shorter decay
        
        # Clamp to reasonable range
        t60s = np.clip(t60s, 0.001, 10.0)
        
        return t60s
    
    def _compute_sphere_gains(self, vertices: np.ndarray, shape_props: ShapeProperties, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute modal gains for sphere.
        
        Gains are based on spherical harmonics evaluated at vertex positions.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)
        
        centroid = shape_props.centroid
        radius = shape_props.dimensions.get('radius', 0.1)
        
        # Normalize vertex positions to unit sphere
        directions = (vertices - centroid) / radius
        
        gains = np.zeros((n_modes, n_vertices))
        
        for mode_idx in range(n_modes):
            # Use spherical harmonics for mode shapes
            l = (mode_idx // 2) + 1  # Angular momentum number
            m = mode_idx % (2 * l + 1) - l  # Magnetic quantum number
            
            for vertex_idx in range(n_vertices):
                x, y, z = directions[vertex_idx]
                
                # Simple spherical harmonic approximation
                r = np.sqrt(x**2 + y**2 + z**2)
                if r > 0:
                    theta = np.arccos(z / r)
                    phi = np.arctan2(y, x)
                    
                    # Legendre polynomial approximation
                    if l == 1:
                        if m == -1:
                            gain = np.sin(theta) * np.sin(phi)
                        elif m == 0:
                            gain = np.cos(theta)
                        else:
                            gain = np.sin(theta) * np.cos(phi)
                    elif l == 2:
                        if m == -2:
                            gain = np.sin(theta)**2 * np.sin(2*phi)
                        elif m == -1:
                            gain = np.sin(2*theta) * np.sin(phi)
                        elif m == 0:
                            gain = 3*np.cos(theta)**2 - 1
                        elif m == 1:
                            gain = np.sin(2*theta) * np.cos(phi)
                        else:
                            gain = np.sin(theta)**2 * np.cos(2*phi)
                    else:
                        # Higher modes: use simpler approximation
                        gain = np.sin(l * theta) * np.cos(m * phi)
                    
                    gains[mode_idx, vertex_idx] = gain * 0.1  # Scale

        return gains

    def _compute_cube_gains(self, vertices: np.ndarray, shape_props: ShapeProperties,
                            frequencies: np.ndarray) -> np.ndarray:
        """
        Compute modal gains for a cube.
        
        Gains are based on standing wave patterns in 3D.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)
        
        centroid = shape_props.centroid
        side = shape_props.dimensions.get('side', 0.1)
        
        # Normalize vertex positions to [-0.5, 0.5] range
        normalized_vertices = (vertices - centroid) / side
        
        gains = np.zeros((n_modes, n_vertices))
        
        for mode_idx in range(n_modes):
            # Determine mode indices from frequency order
            # Simple pattern: increasing complexity
            i = (mode_idx // 4) + 1
            j = ((mode_idx % 4) // 2) + 1
            k = (mode_idx % 2) + 1
            
            for vertex_idx in range(n_vertices):
                x, y, z = normalized_vertices[vertex_idx]
                
                # Standing wave pattern
                gain = (np.sin(i * np.pi * (x + 0.5)) * 
                       np.sin(j * np.pi * (y + 0.5)) * 
                       np.sin(k * np.pi * (z + 0.5)))
                
                gains[mode_idx, vertex_idx] = gain * 0.1
        
        return gains
    
    def _compute_cylinder_gains(self, vertices: np.ndarray, shape_props: ShapeProperties,
                                 frequencies: np.ndarray) -> np.ndarray:
        """
        Compute modal gains for a cylinder.
        
        Combines Bessel functions for radial modes and sine waves for axial modes.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)
        
        centroid = shape_props.centroid
        radius = shape_props.dimensions.get('radius', 0.05)
        height = shape_props.dimensions.get('height', 0.1)
        
        gains = np.zeros((n_modes, n_vertices))
        
        for mode_idx in range(n_modes):
            # Mode indices
            n_axial = (mode_idx % 3) + 1  # Axial mode order
            m_radial = (mode_idx // 3) + 1  # Radial mode order
            
            for vertex_idx in range(n_vertices):
                # Convert to cylindrical coordinates
                dx = vertices[vertex_idx, 0] - centroid[0]
                dy = vertices[vertex_idx, 1] - centroid[1]
                dz = vertices[vertex_idx, 2] - centroid[2]
                
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                z_norm = dz / height + 0.5  # Normalize to [0, 1]
                
                # Bessel function approximation for radial mode
                if r > 0:
                    rho = r / radius
                    # Simple approximation of J₀ and J₁
                    if m_radial == 1:
                        radial_gain = np.cos(2.405 * rho)  # J₀ approximation
                    else:
                        radial_gain = np.cos(2.405 * rho) * np.cos(m_radial * theta)
                else:
                    radial_gain = 1.0
                
                # Axial mode
                axial_gain = np.sin(n_axial * np.pi * z_norm)
                
                gain = radial_gain * axial_gain * 0.1
                gains[mode_idx, vertex_idx] = gain
        
        return gains
    
    def _compute_plate_gains(self, vertices: np.ndarray, shape_props: ShapeProperties, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute modal gains for a plate.
        
        Uses 2D standing wave patterns.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)
        
        centroid = shape_props.centroid
        length = shape_props.dimensions.get('length', 0.2)
        width = shape_props.dimensions.get('width', 0.15)
        
        gains = np.zeros((n_modes, n_vertices))
        
        for mode_idx in range(n_modes):
            # Mode indices for plate
            m = (mode_idx % 4) + 1
            n = (mode_idx // 4) + 1
            
            for vertex_idx in range(n_vertices):
                # Project onto plate plane (assume XY plane)
                x_norm = (vertices[vertex_idx, 0] - centroid[0]) / length + 0.5
                y_norm = (vertices[vertex_idx, 1] - centroid[1]) / width + 0.5
                
                # Clamp to [0, 1]
                x_norm = np.clip(x_norm, 0, 1)
                y_norm = np.clip(y_norm, 0, 1)
                
                # 2D standing wave pattern
                gain = (np.sin(m * np.pi * x_norm) * 
                       np.sin(n * np.pi * y_norm))
                
                gains[mode_idx, vertex_idx] = gain * 0.1
        
        return gains
    
    def _compute_beam_gains(self, vertices: np.ndarray, shape_props: ShapeProperties, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute modal gains for a beam.
        
        Uses Euler-Bernoulli beam mode shapes.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)
        
        centroid = shape_props.centroid
        length = shape_props.dimensions.get('length', 0.2)
        
        # Mode shape constants for clamped-free beam
        # φ_n(x) = cosh(β_nx) - cos(β_nx) - σ_n(sinh(β_nx) - sin(β_nx))
        beta_L = [1.875, 4.694, 7.855, 10.996, 14.137, 17.279]
        sigma = [0.734, 1.018, 0.999, 1.000, 1.000, 1.000]
        
        gains = np.zeros((n_modes, n_vertices))
        
        for mode_idx in range(n_modes):
            if mode_idx >= len(beta_L):
                break
            
            beta = beta_L[mode_idx] / length
            sigma_n = sigma[mode_idx]
            
            for vertex_idx in range(n_vertices):
                # Position along beam axis (assume Z-axis)
                z = vertices[vertex_idx, 2] - centroid[2]
                z_norm = z / length + 0.5  # Normalize to [0, 1]
                z_norm = np.clip(z_norm, 0, 1)
                
                x = beta * z_norm * length
                
                # Mode shape
                gain = (np.cosh(x) - np.cos(x) - 
                       sigma_n * (np.sinh(x) - np.sin(x)))
                
                gains[mode_idx, vertex_idx] = gain * 0.1
        
        return gains
    
    def to_faust_lib(self, modal_params: ModalParameters, output_name: str, min_freq: float, max_freq: float) -> str:
        """
        Generate Faust .lib file content from modal parameters.
        
        Parameters:
        -----------
        modal_params : ModalParameters
            Generated modal parameters
        output_name : str
            Name for the output library
        min_freq : float
            Minimum frequency (Hz)
        max_freq : float
            Maximum frequency (Hz)
            
        Returns:
        --------
        str
            Faust .lib file content
        """
        n_modes = modal_params.n_modes
        n_vertices = modal_params.n_vertices
        
        # Format frequencies
        freq_str = ", ".join([f"{f:.6f}" for f in modal_params.frequencies])
        
        # Format T60s
        t60_str = ", ".join([f"{t:.6f}" for t in modal_params.t60s])
        
        # Format gains (flattened array)
        gains_flat = modal_params.gains.flatten()
        gain_str = ", ".join([f"{g:.10f}" for g in gains_flat])
        
        # Generate Faust code
        lib_content = f"""
// ------------------------------------------------------------
// Approximate modal model for {output_name}
// Generated by Approx2Faust fallback mechanism
// Shape type: {self._get_shape_type_name()}
// Modes: {n_modes}, Vertices: {n_vertices}
// Frequency range: {min_freq:.1f} - {max_freq:.1f} Hz
// ------------------------------------------------------------

declare name        "{output_name}";
declare version     "0.1";
declare author      "Approx2Faust";
declare license     "GPL";

import("stdfaust.lib");

// Modal parameters
nModes = {n_modes};
nExPos = {n_vertices};

// Mode frequencies (Hz)
modeFreqsUnscaled = ba.take(nModes, ({freq_str}));

// T60 decay times (seconds)
modesT60s = t60Scale : ba.take(nModes, ({t60_str}));

// Mode gains (nModes x nExPos)
modesGains = waveform{{{gain_str}}};

// Frequency scaling factor
freqScale = 1.0;

// T60 scaling factor
t60Scale = 1.0;

// Process function
process = no.process;
"""
        
        return lib_content
    
    def _get_shape_type_name(self) -> str:
        """Get the name of the last classified shape type."""
        if hasattr(self, '_last_shape_type'):
            return self._last_shape_type.value
        return "unknown"

