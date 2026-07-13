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
import re
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _parse_lib, _compute_rayleigh_damping


@dataclass
class Modal4Proxy:
    """
    Adapt modal models for proxy shapes.

    For proxy_type 0 (octahedron) and 1 (dodecahedron):
    - Generates an approximate modal model based on the original object's
      acoustic material and mesh shape using (proxy_type + 2) modal modes.

    For proxy_type 2,3,4 (icosahedron with subdivisions):
    - Adapts the modal model from mesh2faust of the original mesh to the
      proxy's mesh shape using (proxy_type + 2) modal modes.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')

    def compute(self, obj_idx: int) -> None:
        """
        Compute modal model for a proxied object.

        Args:
            obj_idx: Object index
        """
        config = self.config

        # Find the object config
        config_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx: 
                if obj.proxy_type is False:
                    return  # Not a proxied object
                else:
                    config_obj = obj
                    proxy_type = config_obj.proxy_type

        # Get material properties
        young_modulus = config_obj.acoustic_shader.young_modulus
        poisson_ratio = config_obj.acoustic_shader.poisson_ratio
        density = config_obj.acoustic_shader.density

        damping = config_obj.acoustic_shader.damping if config_obj.acoustic_shader.damping is not None else 0.02
        min_freq = config_obj.acoustic_shader.low_frequency
        max_freq = config_obj.acoustic_shader.high_frequency

        # Number of modes based on proxy_type
        n_modes = proxy_type + 2

        # Load original mesh for shape analysis
        vertices, normals, faces = _load_mesh(config_obj, 0)

        # Compute proxy mesh vertices (local coordinates)
        proxy_vertices, _ = self._get_proxy_vertices(config_obj, proxy_type)

        # Generate or adapt modal model
        if proxy_type in [0, 1]:
            # Approximate modal model for simple proxies
            self._generate_approximate_modal_model(
                config_obj=config_obj,
                proxy_type=proxy_type,
                proxy_vertices=proxy_vertices,
                original_vertices=vertices,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                density=density,
                damping=damping,
                min_freq=min_freq,
                max_freq=max_freq,
                n_modes=n_modes
            )
        elif proxy_type in [2, 3, 4]:
            # Adapt mesh2faustust modal model for subdivided icosahedron
            self._adapt_mesh2faust_modal_model(
                config_obj=config_obj,
                proxy_type=proxy_type,
                proxy_vertices=proxy_vertices,
                original_vertices=vertices,
                original_faces=faces,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                density=density,
                damping=damping,
                min_freq=min_freq,
                max_freq=max_freq,
                n_modes=n_modes
            )

    def _get_proxy_vertices(self, config_obj: Any, proxy_type: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get proxy mesh vertices in local coordinates.

        Args:
            config_obj: Object configuration
            proxy_type: Proxy type (0-4)

        Returns:
            Tuple of (vertices, faces) in local coordinates
        """
        # Load original mesh to get extents
        vertices, normals, faces = _load_mesh(config_obj, 0)

        # Compute bounding box extents
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        extents = max_coords - min_coords
        center_local = (min_coords + max_coords) / 2

        # Generate proxy vertices
        if proxy_type == 0:
            proxy_vertices, proxy_faces = self._create_octahedron()
        elif proxy_type == 1:
            proxy_vertices, proxy_faces = self._create_dodecahedron()
        elif proxy_type in [2, 3, 4]:
            subdivisions = proxy_type - 2
            proxy_vertices, proxy_faces = self._create_icosahedron(subdivisions=subdivisions)
        else:
            proxy_vertices, proxy_faces = self._create_octahedron()

        # Scale to match extents
        norms = np.linalg.norm(proxy_vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        proxy_vertices_normalized = proxy_vertices / norms
        half_extents = extents / 2.0
        proxy_vertices_scaled = proxy_vertices_normalized * half_extents[np.newaxis, :]

        # Center at origin
        proxy_vertices_scaled = proxy_vertices_scaled - np.mean(proxy_vertices_scaled, axis=0)

        return proxy_vertices_scaled, proxy_faces

    def _generate_approximate_modal_model(
        self,
        config_obj: Any,
        proxy_type: int,
        proxy_vertices: np.ndarray,
        original_vertices: np.ndarray,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
        damping: float,
        min_freq: float,
        max_freq: float,
        n_modes: int
    ) -> None:
        """
        Generate an approximate modal model for simple proxy shapes (0, 1).

        Uses analytical mode shapes for octahedron and dodecahedron,
        scaled by material properties and object dimensions.
        """
        # Compute effective radius from original mesh volume
        volume = self._compute_volume(original_vertices)
        effective_radius = (3 * volume / (4 * np.pi)) ** (1/3)

        # Compute wave speeds from material properties
        c_long, c_shear = self._compute_wave_speeds(young_modulus, poisson_ratio, density)

        # Generate modal frequencies based on proxy shape
        if proxy_type == 0:
            # Octahedron modes (8 faces)
            # Analytical approximation for octahedral vibrations
            frequencies = self._octahedron_mode_frequencies(
                n_modes=n_modes,
                c_shear=c_shear,
                c_long=c_long,
                radius=effective_radius,
                min_freq=min_freq,
                max_freq=max_freq
            )
        else:
            # Dodecahedron modes (12 faces)
            frequencies = self._dodecahedron_mode_frequencies(
                n_modes=n_modes,
                c_shear=c_shear,
                c_long=c_long,
                radius=effective_radius,
                min_freq=min_freq,
                max_freq=max_freq
            )

        # Compute T60 values
        t60s = self._compute_t60s(frequencies, damping)

        # Compute gains for each proxy vertex
        gains = self._compute_proxy_gains(
            vertices=proxy_vertices,
            frequencies=frequencies,
            proxy_type=proxy_type
        )

        # Generate Faust .lib file
        self._save_faust_lib(
            config_obj=config_obj,
            proxy_type=proxy_type,
            frequencies=frequencies,
            t60s=t60s,
            gains=gains,
            n_modes=n_modes,
            n_vertices=len(proxy_vertices)
        )

    def _adapt_mesh2faust_modal_model(
        self,
        config_obj: Any,
        proxy_type: int,
        proxy_vertices: np.ndarray,
        original_vertices: np.ndarray,
        original_faces: np.ndarray,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
        damping: float,
        min_freq: float,
        max_freq: float,
        n_modes: int
    ) -> None:
        """
        Adapt the mesh2faust modal model from the original mesh to the proxy mesh.

        For icosahedron proxies (2,3,4), we:
        1. Try to use mesh2faust on the original mesh
        2. Adapt the modal model to the proxy's vertex count and shape
        3. Interpolate gains from original vertices to proxy vertices
        """
        cache_path = self.config.system.cache_path
        dsp_path = f"{cache_path}/dsp"

        # Check if mesh2faust already generated a .lib file for this object
        lib_file = f"{dsp_path}/{config_obj.name}.lib"

        if os.path.exists(lib_file):
            # Parse the existing modal model
            modal_data = _parse_lib(lib_file)

            # Adapt frequencies (scale by volume ratio)
            original_volume = self._compute_volume(original_vertices)
            proxy_volume = self._compute_volume(proxy_vertices)

            if original_volume > 0 and proxy_volume > 0:
                # Frequency scaling: f ∝ 1/L ∝ 1/V^(1/3)
                volume_ratio = (original_volume / proxy_volume) ** (1/3)
                adapted_frequencies = modal_data['frequencies'] * volume_ratio
            else:
                adapted_frequencies = modal_data['frequencies']

            # Clamp to frequency range
            adapted_frequencies = np.clip(adapted_frequencies, min_freq, max_freq)

            # Adapt T60 values (scale by volume ratio)
            # Larger objects have longer decay times
            if original_volume > 0 and proxy_volume > 0:
                t60_ratio = (proxy_volume / original_volume) ** (1/3)
                adapted_t60s = modal_data['t60s'] * t60_ratio
            else:
                adapted_t60s = modal_data['t60s']

            # Interpolate gains from original vertices to proxy vertices
            adapted_gains = self._interpolate_gains_to_proxy(
                original_vertices=original_vertices,
                original_gains=modal_data['gains'],
                proxy_vertices=proxy_vertices,
                n_modes=n_modes
            )

            # Truncate or pad to n_modes
            n_available = len(adapted_frequencies)
            if n_available >= n_modes:
                frequencies = adapted_frequencies[:n_modes]
                t60s = adapted_t60s[:n_modes]
                gains = adapted_gains[:n_modes]
            else:
                # Pad with higher modes
                frequencies = np.pad(adapted_frequencies, (0, n_modes - n_available),
                                    mode='linear_ramp', end_values=(max_freq,))
                t60s = np.pad(adapted_t60s, (0, n_modes - n_available),
                             mode='constant', constant_values=(np.mean(adapted_t60s),))
                gains = np.pad(adapted_gains, ((0, n_modes - n_available), (0, 0)),
                              mode='constant', constant_values=(0,))
        else:
            # No existing modal model - generate approximate one
            # Use icosahedron-specific mode frequencies
            volume = self._compute_volume(proxy_vertices)
            effective_radius = (3 * volume / (4 * np.pi)) ** (1/3)
            c_long, c_shear = self._compute_wave_speeds(young_modulus, poisson_ratio, density)

            frequencies = self._icosahedron_mode_frequencies(
                n_modes=n_modes,
                c_shear=c_shear,
                c_long=c_long,
                radius=effective_radius,
                min_freq=min_freq,
                max_freq=max_freq,
                subdivisions=proxy_type - 2
            )

            t60s = self._compute_t60s(frequencies, damping)
            gains = self._compute_proxy_gains(
                vertices=proxy_vertices,
                frequencies=frequencies,
                proxy_type=proxy_type
            )

        # Save the adapted modal model
        self._save_faust_lib(
            config_obj=config_obj,
            proxy_type=proxy_type,
            frequencies=frequencies,
            t60s=t60s,
            gains=gains,
            n_modes=n_modes,
            n_vertices=len(proxy_vertices)
        )

    def _interpolate_gains_to_proxy(
        self,
        original_vertices: np.ndarray,
        original_gains: List[np.ndarray],
        proxy_vertices: np.ndarray,
        n_modes: int
    ) -> np.ndarray:
        """
        Interpolate modal gains from original vertices to proxy vertices.

        Uses nearest-neighbor interpolation based on spatial proximity.
        """
        from scipy.spatial import cKDTree

        n_proxy = len(proxy_vertices)
        n_modes_orig = min(len(original_gains), n_modes)

        # Build KD-tree for original vertices
        tree = cKDTree(original_vertices)

        # For each proxy vertex, find nearest original vertices
        distances, indices = tree.query(proxy_vertices, k=3)  # 3 nearest neighbors

        # Interpolate gains
        adapted_gains = np.zeros((n_modes, n_proxy))

        for mode_idx in range(n_modes_orig):
            for proxy_idx in range(n_proxy):
                # Weighted average of nearest neighbors
                weights = 1.0 / (distances[proxy_idx] + 1e-10)
                weights = weights / np.sum(weights)

                gain = 0.0
                for k in range(len(indices[proxy_idx])):
                    orig_idx = indices[proxy_idx][k]
                    if orig_idx < len(original_gains[mode_idx]):
                        gain += weights[k] * original_gains[mode_idx][orig_idx]

                adapted_gains[mode_idx, proxy_idx] = gain

        return adapted_gains

    def _octahedron_mode_frequencies(
        self,
        n_modes: int,
        c_shear: float,
        c_long: float,
        radius: float,
        min_freq: float,
        max_freq: float
    ) -> np.ndarray:
        """
        Compute approximate mode frequencies for an octahedron.

        Octahedron has 8 triangular faces. Mode frequencies are approximated
        using spherical harmonic-like patterns adapted for octahedral symmetry.
        """
        frequencies = []

        # Mode families for octahedral symmetry
        # l = angular momentum number, m = magnetic quantum number
        mode_families = [
            (1, 0, 1.0),    # Breathing mode
            (1, 1, 1.5),    # Dipole mode
            (2, 0, 2.0),    # Quadrupole mode 1
            (2, 1, 2.3),    # Quadrupole mode 2
            (2, 2, 2.7),    # Quadrupole mode 3
            (3, 0, 3.0),    # Octupole mode 1
            (3, 1, 3.4),    # Octupole mode 2
            (3, 2, 3.8),    # Octupole mode 3
            (4, 0, 4.2),    # Higher mode 1
            (4, 1, 4.6),    # Higher mode 2
            (4, 2, 5.0),    # Higher mode 3
            (5, 0, 5.5),    # Higher mode 4
        ]

        for l, m, factor in mode_families:
            # Base frequency for this mode family
            f_base = factor * c_shear / (2 * np.pi * radius)

            if f_base < min_freq:
                continue
            if f_base > max_freq:
                break

            frequencies.append(f_base)

            # Add overtones with slight frequency shifts
            for k in range(1, 3):
                f_ot = f_base * (1 + k * 0.3 * (l + 1) / (m + 1))
                if f_ot <= max_freq:
                    frequencies.append(f_ot)

            if len(frequencies) >= n_modes * 2:
                break

        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]

        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.15))

        return frequencies

    def _dodecahedron_mode_frequencies(
        self,
        n_modes: int,
        c_shear: float,
        c_long: float,
        radius: float,
        min_freq: float,
        max_freq: float
    ) -> np.ndarray:
        """
        Compute approximate mode frequencies for a dodecahedron.

        Dodecahedron has 12 pentagonal faces. Mode frequencies are approximated
        using icosahedral symmetry (dual to dodecahedron).
        """
        frequencies = []

        # Mode families for icosahedral/dodecahedral symmetry
        mode_families = [
            (1, 0, 1.0),    # Breathing mode
            (1, 1, 1.4),    # Dipole mode
            (2, 0, 1.8),    # Quadrupole mode 1
            (2, 1, 2.1),    # Quadrupole mode 2
            (2, 2, 2.5),    # Quadrupole mode 3
            (3, 0, 2.8),    # Octupole mode 1
            (3, 1, 3.2),    # Octupole mode 2
            (3, 2, 3.6),    # Octupole mode 3
            (4, 0, 4.0),    # Higher mode 1
            (4, 1, 4.4),    # Higher mode 2
            (4, 2, 4.8),    # Higher mode 3
            (5, 0, 5.2),    # Higher mode 4
            (5, 1, 5.6),    # Higher mode 5
            (5, 2, 6.0),    # Higher mode 6
        ]

        for l, m, factor in mode_families:
            # Base frequency for this mode family
            # Dodecahedron has slightly higher frequencies than octahedron for same size
            f_base = factor * c_shear / (2 * np.pi * radius)

            if f_base < min_freq:
                continue
            if f_base > max_freq:
                break

            frequencies.append(f_base)

            # Add overtones
            for k in range(1, 3):
                f_ot = f_base * (1 + k * 0.25 * (l + 1) / (m + 1))
                if f_ot <= max_freq:
                    frequencies.append(f_ot)

            if len(frequencies) >= n_modes * 2:
                break

        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]

        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.12))

        return frequencies

    def _icosahedron_mode_frequencies(
        self,
        n_modes: int,
        c_shear: float,
        c_long: float,
        radius: float,
        min_freq: float,
        max_freq: float,
        subdivisions: int = 0
    ) -> np.ndarray:
        """
        Compute approximate mode frequencies for an icosahedron.

        Higher subdivisions create smoother shapes that approach spherical behavior.
        """
        frequencies = []

        # Subdivision factor affects mode spacing
        # More subdivisions = more modes in the same frequency range
        sub_factor = 1.0 + subdivisions * 0.3

        # Mode families for icosahedral symmetry
        mode_families = [
            (1, 0, 1.0),    # Breathing mode
            (1, 1, 1.3),    # Dipole mode
            (2, 0, 1.7),    # Quadrupole mode 1
            (2, 1, 2.0),    # Quadrupole mode 2
            (2, 2, 2.4),    # Quadrupole mode 3
            (3, 0, 2.7),    # Octupole mode 1
            (3, 1, 3.1),    # Octupole mode 2
            (3, 2, 3.5),    # Octupole mode 3
            (4, 0, 3.8),    # Higher mode 1
            (4, 1, 4.2),    # Higher mode 2
            (4, 2, 4.6),    # Higher mode 3
            (5, 0, 5.0),    # Higher mode 4
            (5, 1, 5.4),    # Higher mode 5
            (5, 2, 5.8),    # Higher mode 6
            (6, 0, 6.2),    # Higher mode 7
            (6, 1, 6.6),    # Higher mode 8
        ]

        for l, m, factor in mode_families:
            # Base frequency for this mode family
            f_base = factor * c_shear / (2 * np.pi * radius) * sub_factor

            if f_base < min_freq:
                continue
            if f_base > max_freq:
                break

            frequencies.append(f_base)

            # Add overtones with subdivision-dependent spacing
            for k in range(1, 3):
                f_ot = f_base * (1 + k * 0.2 * sub_factor)
                if f_ot <= max_freq:
                    frequencies.append(f_ot)

            if len(frequencies) >= n_modes * 2:
                break

        # Sort and take the lowest n_modes
        frequencies = np.sort(frequencies)[:n_modes]

        # Pad if needed
        if len(frequencies) < n_modes:
            last_freq = frequencies[-1] if len(frequencies) > 0 else min_freq
            for i in range(n_modes - len(frequencies)):
                frequencies = np.append(frequencies, last_freq * (1 + (i + 1) * 0.1))

        return frequencies

    def _compute_wave_speeds(self, young_modulus: float, poisson_ratio: float, density: float) -> Tuple[float, float]:
        """
        Compute longitudinal and shear wave speeds.

        Returns:
            (c_longitudinal, c_shear) in m/s
        """
        if young_modulus is None or young_modulus <= 0 or density is None or density <= 0:
            return 343.0, 200.0  # Default values

        # Longitudinal wave speed
        if poisson_ratio is not None and poisson_ratio != 1.0:
            c_long = np.sqrt(young_modulus * (1 - poisson_ratio) /
                            (density * (1 + poisson_ratio) * (1 - 2 * poisson_ratio)))
        else:
            c_long = np.sqrt(young_modulus / density)

        # Shear wave speed
        if poisson_ratio is not None:
            c_shear = np.sqrt(young_modulus / (2 * density * (1 + poisson_ratio)))
        else:
            c_shear = np.sqrt(young_modulus / (2 * density))

        return c_long, c_shear

    def _compute_t60s(self, frequencies: np.ndarray, damping: float) -> np.ndarray:
        """
        Compute T60 (reverberation time) for each mode.

        T60 = 3 * ln(10) / (π * damping * f)
        """
        if damping is None or damping <= 0:
            damping = 0.02  # Default conservative value

        t60s = 3 * np.log(10) / (np.pi * damping * np.maximum(frequencies, 1))

        # Clamp to reasonable range
        t60s = np.clip(t60s, 0.001, 10.0)

        return t60s

    def _compute_proxy_gains(self, vertices: np.ndarray, frequencies: np.ndarray, proxy_type: int) -> np.ndarray:
        """
        Compute modal gains for each vertex of the proxy mesh.

        Gains are based on the mode shapes of the proxy geometry.
        """
        n_vertices = len(vertices)
        n_modes = len(frequencies)

        # Normalize vertices to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        directions = vertices / norms

        gains = np.zeros((n_modes, n_vertices))

        for mode_idx in range(n_modes):
            # Determine mode order from frequency index
            mode_order = (mode_idx // 2) + 1

            for vertex_idx in range(n_vertices):
                x, y, z = directions[vertex_idx]

                # Spherical coordinates
                r = np.sqrt(x**2 + y**2 + z**2)
                if r > 0:
                    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
                    phi = np.arctan2(y, x)

                    # Mode shape based on spherical harmonics
                    gain = self._spherical_harmonic_approximation(
                        l=mode_order,
                        m=mode_idx % (2 * mode_order + 1) - mode_order,
                        theta=theta,
                        phi=phi
                    )

                    gains[mode_idx, vertex_idx] = gain * 0.1  # Scale

        return gains

    def _spherical_harmonic_approximation(self, l: int, m: int, theta: float, phi: float) -> float:
        """
        Approximate spherical harmonic function for mode shapes.

        This is a simplified approximation that captures the essential
        angular dependence of vibration modes.
        """
        if l == 0:
            return 1.0  # Breathing mode
        elif l == 1:
            if m == -1:
                return np.sin(theta) * np.sin(phi)
            elif m == 0:
                return np.cos(theta)
            else:
                return np.sin(theta) * np.cos(phi)
        elif l == 2:
            if m == -2:
                return np.sin(theta)**2 * np.sin(2 * phi)
            elif m == -1:
                return np.sin(2 * theta) * np.sin(phi)
            elif m == 0:
                return 3 * np.cos(theta)**2 - 1
            elif m == 1:
                return np.sin(2 * theta) * np.cos(phi)
            else:
                return np.sin(theta)**2 * np.cos(2 * phi)
        elif l == 3:
            if m == -3:
                return np.sin(theta)**3 * np.sin(3 * phi)
            elif m == -2:
                return np.sin(theta)**2 * np.cos(theta) * np.sin(2 * phi)
            elif m == -1:
                return np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.sin(phi)
            elif m == 0:
                return 5 * np.cos(theta)**3 - 3 * np.cos(theta)
            elif m == 1:
                return np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.cos(phi)
            elif m == 2:
                return np.sin(theta)**2 * np.cos(theta) * np.cos(2 * phi)
            else:
                return np.sin(theta)**3 * np.cos(3 * phi)
        else:
            # Higher modes: use simpler approximation
            return np.sin(l * theta) * np.cos(m * phi)

    def _compute_volume(self, vertices: np.ndarray) -> float:
        """Estimate volume from vertex cloud using convex hull."""
        if len(vertices) < 4:
            return 0.0

        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            # Fallback: approximate as bounding box volume / 2
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
            return np.prod(dimensions) / 2.0

    def _save_faust_lib(
        self,
        config_obj: Any,
        proxy_type: int,
        frequencies: np.ndarray,
        t60s: np.ndarray,
        gains: np.ndarray,
        n_modes: int,
        n_vertices: int
    ) -> None:
        """
        Save the modal model as a Faust .lib file.

        The file is saved to the DSP cache directory with a proxy suffix.
        """
        cache_path = self.config.system.cache_path
        dsp_path = f"{cache_path}/dsp"
        os.makedirs(dsp_path, exist_ok=True)

        # Output name with proxy suffix
        output_name = f"{config_obj.name}_proxy_{proxy_type}"
        lib_file = f"{dsp_path}/{output_name}.lib"

        # Format frequencies
        freq_str = ", ".join([f"{f:.6f}" for f in frequencies])

        # Format T60s
        t60_str = ", ".join([f"{t:.6f}" for t in t60s])

        # Format gains (flattened array)
        gains_flat = gains.flatten()
        gain_str = ", ".join([f"{g:.10f}" for g in gains_flat])

        # Get material properties for documentation
        young_modulus = config_obj.acoustic_shader.young_modulus if config_obj.acoustic_shader else "N/A"
        poisson_ratio = config_obj.acoustic_shaderader.poisson_ratio if config_obj.acoustic_shader else "N/A"
        density = config_obj.acoustic_shader.density if config_obj.acoustic_shader else "N/A"

        # Proxy shape name
        proxy_names = {0: "octahedron", 1: "dodecahedron", 2: "icosahedron", 3: "icosahedron_sub1", 4: "icosaosahedron_sub2"}
        proxy_name = proxy_names.get(proxy_type, "unknown")

        # Generate Faust code
        lib_content = f"""
// ------------------------------------------------------------
// Adapted modal model for {config_obj.name} (proxy)
// Generated by Modal4Proxy
// Proxy type: {proxy_type} ({proxy_name})
// Modes: {n_modes}, Vertices: {n_vertices}
// Material: E={young_modulus}, ν={poisson_ratio}, ρ={density}
// ------------------------------------------------------------

declare name        "{output_name}";
declare version     "0.1";
declare author      "Modal4Proxy";
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

        # Write the .lib file
        with open(lib_file, 'w') as f:
            f.write(lib_content)

        # Also save a resonance version if needed
        if config_obj.resonance:
            resonance_modes = config_obj.resononance_modes if config_obj.resonance_modes else max(5, n_modes // 2)
            resonance_frequencies = frequencies[:resonance_modes]
            resonance_t60s = t60s[:resonance_modes] * 1.2  # Slightly longer decay for resonance
            resonance_gains = gains[:resonance_modes]

            resonance_freq_str = ", ".join([f"{f:.6f}" for f in resonance_frequencies])
            resonance_t60_str = ", ".join([f"{t:.6f}" for t in resonance_t60s])
            resonance_gains_flat = resonance_gains.flatten()
            resonance_gain_str = ", ".join([f"{g:.10f}" for g in resonance_gains_flat])

            resonance_content = f"""
// ------------------------------------------------------------
// Resonance modal model for {config_obj.name} (proxy)
// Generated by Modal4Proxy
// Proxy type: {proxy_type} ({proxy_name})
// Modes: {resonance_modes}, Vertices: {n_vertices}
// ------------------------------------------------------------

declare name        "{output_name}_resonance";
declare version     "0.1";
declare author      "Modal4Proxy";
declare license     "GPL";

import("stdfaust.lib");

nModes = {resonance_modes};
nExPos = {n_vertices};

modeFreqsUnscaled = ba.take(nModes, ({resonance_freq_str}));

modesT60s = t60Scale : ba.take(nModes, ({resonance_t60_str}));

modesGains = waveform{{{resonance_gain_str}}};

freqScale = 1.0;
t60Scale = 1.0;

process = no.process;
"""

            resonance_file = f"{dsp_path}/{output_name}_resonance.lib"
            with open(resonance_file, 'w') as f:
                f.write(resonance_content)

            print(f"Modal4Proxy: Saved resonance model to {resonance_file}")

        print(f"Modal4Proxy: Saved modal model to {lib_file}")
