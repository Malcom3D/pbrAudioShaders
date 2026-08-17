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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import toeplitz

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _parse_lib, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix


@dataclass
class ProxyIRTable:
    """
    Precomputed Impulse Response table for proxy meshes.

    The table is indexed by:
    - Material group (based on acoustic properties)
    - Proxy shape type (pyramid=0, octahedron=1, cube=2)
    - Size scale (interpolated between min and max size)

    Uses precomputed Toeplitz-like impulse response matrix from Modal4Proxy .lib files.
    """
    entity_manager: EntityManager

    # IR parameters
    sample_rate: int = 48000
    max_ir_length: int = 8192  # Maximum IR length in samples

    # Internal state
    ir_table: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    size_steps: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    min_size: Dict[Tuple[int, int], float] = field(default_factory=dict)
    max_size: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Map proxy_type to shape name
    _shape_names: Dict[int, str] = field(default_factory=lambda: {
        0: "pyramid",
        1: "octahedron",
        2: "cube"
    })

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = config.system.sample_rate

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        # Cache for parsed lib files
        self._lib_cache = {}

    def compute_ir_table(self, proxy_meshes: List[Any]) -> None:
        """
        Compute the IR table from proxy meshes.

        Groups meshes by (material_key, proxy_type) and builds Toeplitz-like IR matrices.

        Parameters:
        -----------
        proxy_meshes : List of proxy mesh configurations (only proxy_type 0,1,2)
        """
        # Filter only valid proxy types (0,1,2)
        valid_meshes = [m for m in proxy_meshes if m.proxy_type in [0, 1, 2]]

        if not valid_meshes:
            debug_print("No valid proxy meshes (types 0,1,2) found for IR table")
            return

        # Group meshes by (material_key, proxy_type)
        groups = self._group_by_material_and_shape(valid_meshes)

        for (material_key, proxy_type), meshes in groups.items():
            # Compute size range for this group
            sizes = [self._compute_mesh_size(mesh) for mesh in meshes]
            self.min_size[(material_key, proxy_type)] = min(sizes)
            self.max_size[(material_key, proxy_type)] = max(sizes)

            # Determine number of size steps from number of meshes
            n_steps = max(2, len(meshes))
            size_steps = np.linspace(0, 1, n_steps)
            self.size_steps[(material_key, proxy_type)] = size_steps

            # Get material properties from first mesh
            material_props = self._get_material_properties(meshes[0])

            # Build IR table for this group
            ir_matrix = self._build_toeplitz_ir_matrix(
                meshes=meshes,
                material_key=material_key,
                proxy_type=proxy_type,
                material_props=material_props,
                size_steps=size_steps,
                n_steps=n_steps
            )

            self.ir_table[(material_key, proxy_type)] = ir_matrix

            debug_print(f"Built IR table for material {material_key}, shape {self._shape_names[proxy_type]}, "
                       f"steps: {n_steps}, size range: {self.min_size[(material_key, proxy_type)]:.4f} - "
                       f"{self.max_size[(material_key, proxy_type)]:.4f}m")

    def _group_by_material_and_shape(self, proxy_meshes: List[Any]) -> Dict[Tuple[int, int], List[Any]]:
        """
        Group proxy meshes by material and proxy shape type.

        Returns:
            Dict with key (material_key, proxy_type) and value list of meshes
        """
        groups = {}

        for mesh in proxy_meshes:
            # Create material key from acoustic properties
            if mesh.acoustic_shader:
                material_key = (
                    mesh.acoustic_shader.young_modulus,
                    mesh.acoustic_shader.poisson_ratio,
                    mesh.acoustic_shader.density,
                    mesh.acoustic_shader.damping
                )
            else:
                material_key = (None, None, None, None)

            # Use a hashable key for the material
            material_key = hash(material_key)

            key = (material_key, mesh.proxy_type)

            if key not in groups:
                groups[key] = []
            groups[key].append(mesh)

        return groups

    def _compute_mesh_size(self, mesh: Any) -> float:
        """Compute characteristic size of a mesh."""
        try:
            vertices, _, _ = _load_mesh(mesh, 0, use_proxy_path=True)
            if len(vertices) > 0:
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                return float(np.linalg.norm(max_coords - min_coords))
        except Exception as e:
            debug_print(f"Error computing mesh size: {e}")
        return 0.1

    def _get_material_properties(self, mesh: Any) -> Dict[str, float]:
        """Extract material properties from mesh config."""
        if mesh.acoustic_shader:
            return {
                'young_modulus': mesh.acoustic_shader.young_modulus or 1e9,
                'poisson_ratio': mesh.acoustic_shader.poisson_ratio or 0.3,
                'density': mesh.acoustic_shader.density or 1000.0,
                'damping': mesh.acoustic_shader.damping or 0.02
            }
        return {
            'young_modulus': 1e9,
            'poisson_ratio': 0.3,
            'density': 1000.0,
            'damping': 0.02
        }

    def _get_lib_file(self, mesh: Any, proxy_type: int) -> Optional[str]:
        """
        Get the path to the .lib file for a proxy mesh.

        Uses Modal4Proxy .lib files.
        """
        config = self.entity_manager.get('config')
        dsp_path = f"{config.system.cache_path}/dsp"

        # Try proxy-specific lib file
        lib_file = f"{dsp_path}/{mesh.name}_proxy_{proxy_type}.lib"

        if not os.path.exists(lib_file):
            # Try fallback to original lib
            lib_file = f"{dsp_path}/{mesh.name}.lib"

        if os.path.exists(lib_file):
            return lib_file

        return None

    def _parse_lib_file(self, lib_file: str) -> Dict:
        """Parse a .lib file with caching."""
        if lib_file in self._lib_cache:
            return self._lib_cache[lib_file]

        try:
            modal_data = _parse_lib(lib_file)
            self._lib_cache[lib_file] = modal_data
            return modal_data
        except Exception as e:
            debug_print(f"Error parsing lib file {lib_file}: {e}")
            return None

    def _build_toeplitz_ir_matrix(
        self,
        meshes: List[Any],
        material_key: int,
        proxy_type: int,
        material_props: Dict[str, float],
        size_steps: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Build a Toeplitz-like impulse response matrix for all modal modes.

        The matrix has shape (n_steps, max_ir_length) where each row is
        the impulse response for a specific size step.

        Uses the modal parameters from the .lib file to generate the IR.
        """
        # Collect modal parameters from all meshes in this group
        all_modal_params = []

        for mesh in meshes:
            lib_file = self._get_lib_file(mesh, proxy_type)
            if lib_file is None:
                continue

            modal_data = self._parse_lib_file(lib_file)
            if modal_data is None:
                continue

            # Compute mesh size
            size = self._compute_mesh_size(mesh)

            all_modal_params.append({
                'size': size,
                'frequencies': modal_data['frequencies'],
                't60s': modal_data['t60s'],
                'gains': modal_data['gains'],
                'nModes': modal_data['nModes']
            })

        if not all_modal_params:
            debug_print(f"No modal data found for group {material_key}, {proxy_type}")
            return np.zeros((n_steps, self.max_ir_length), dtype=np.float32)

        # Sort by size
        all_modal_params.sort(key=lambda x: x['size'])

        # Build IR for each size step using interpolation of modal parameters
        ir_matrix = np.zeros((n_steps, self.max_ir_length), dtype=np.float32)

        # Determine min and max sizes
        sizes = np.array([p['size'] for p in all_modal_params])
        min_s = self.min_size[(material_key, proxy_type)]
        max_s = self.max_size[(material_key, proxy_type)]

        for step_idx, size_scale in enumerate(size_steps):
            target_size = min_s + size_scale * (max_s - min_s)

            # Find nearest modal parameters
            # Use linear interpolation between the two closest sizes
            if len(all_modal_params) == 1:
                params = all_modal_params[0]
            else:
                # Find the two closest sizes
                size_diffs = np.abs(sizes - target_size)
                sorted_indices = np.argsort(size_diffs)
                idx1 = sorted_indices[0]
                idx2 = sorted_indices[1] if len(sorted_indices) > 1 else idx1

                # Interpolate modal parameters
                params = self._interpolate_modal_params(
                    all_modal_params[idx1],
                    all_modal_params[idx2],
                    target_size,
                    sizes[idx1],
                    sizes[idx2]
                )

            # Generate IR from modal parameters
            ir = self._modal_params_to_ir(params, material_props)
            ir = ir[:self.max_ir_length]

            # Pad or truncate
            if len(ir) < self.max_ir_length:
                ir = np.pad(ir, (0, self.max_ir_length - len(ir)))

            ir_matrix[step_idx] = ir[:self.max_ir_length]

        return ir_matrix

    def _interpolate_modal_params(
        self,
        params1: Dict,
        params2: Dict,
        target_size: float,
        size1: float,
        size2: float
    ) -> Dict:
        """
        Interpolate modal parameters between two size points.
        """
        if size2 == size1:
            return params1

        # Weight based on distance in size space
        w1 = 1.0 - (target_size - size1) / (size2 - size1)
        w2 = 1.0 - w1

        # Ensure we have the same number of modes
        n_modes1 = len(params1['frequencies'])
        n_modes2 = len(params2['frequencies'])
        n_modes = min(n_modes1, n_modes2)

        # Interpolate frequencies
        freqs1 = params1['frequencies'][:n_modes]
        freqs2 = params2['frequencies'][:n_modes]
        frequencies = w1 * freqs1 + w2 * freqs2

        # Interpolate T60s
        t60s1 = params1['t60s'][:n_modes]
        t60s2 = params2['t60s'][:n_modes]
        t60s = w1 * t60s1 + w2 * t60s2

        # Interpolate gains (per vertex)
        n_vertices1 = len(params1['gains'][0]) if params1['gains'] else 0
        n_vertices2 = len(params2['gains'][0]) if params2['gains'] else 0
        n_vertices = min(n_vertices1, n_vertices2)

        gains = []
        for i in range(min(len(params1['gains']), len(params2['gains']))):
            if i < n_modes:
                g1 = params1['gains'][i][:n_vertices] if i < len(params1['gains']) else np.zeros(n_vertices)
                g2 = params2['gains'][i][:n_vertices] if i < len(params2['gains']) else np.zeros(n_vertices)
                gains.append(w1 * g1 + w2 * g2)

        return {
            'frequencies': frequencies,
            't60s': t60s,
            'gains': gains,
            'nModes': n_modes
        }

    def _modal_params_to_ir(self, modal_params: Dict, material_props: Dict[str, float]) -> np.ndarray:
        """
        Generate impulse response from modal parameters.

        Uses Toeplitz-like matrix approach: each mode contributes a damped sinusoid
        that is convolved with an excitation impulse.
        """
        frequencies = modal_params['frequencies']
        t60s = modal_params['t60s']
        gains = modal_params['gains']

        n_modes = len(frequencies)

        if n_modes == 0:
            return np.zeros(self.max_ir_length)

        # Determine IR length based on max T60
        max_t60 = np.max(t60s) if len(t60s) > 0 else 0.1
        ir_length = min(int(max_t60 * self.sample_rate * 2), self.max_ir_length)

        # Time axis
        t = np.arange(ir_length) / self.sample_rate

        # Initialize IR
        ir = np.zeros(ir_length, dtype=np.float32)

        # Damping factor from material properties
        damping = material_props.get('damping', 0.02)

        # Build modal contributions using Toeplitz-like structure
        # Each mode contributes: gain * exp(-decay*t) * sin(2*pi*freq*t)
        for i in range(n_modes):
            freq = frequencies[i]
            t60 = t60s[i]

            if freq <= 0 or t60 <= 0:
                continue

            # Decay rate
            decay = 3 * np.log(10) / max(t60, 0.001)

            # Get gains for this mode (average over vertices if multiple)
            if i < len(gains):
                mode_gains = gains[i]
                if isinstance(mode_gains, (list, np.ndarray)):
                    gain = np.mean(mode_gains)
                else:
                    gain = mode_gains
            else:
                gain = 0.1

            # Damped sinusoid for this mode
            mode_ir = gain * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)

            # Add to total IR
            ir += mode_ir

        # Apply overall damping scaling
        ir *= (1.0 - damping * 0.5)

        # Normalize
        max_val = np.max(np.abs(ir))
        if max_val > 0:
            ir = ir / max_val * 0.9

        return ir.astype(np.float32)

    def get_ir(self, material_key: int, proxy_type: int, size_scale: float) -> np.ndarray:
        """
        Get interpolated IR for a given material, shape, and size scale.

        Parameters:
        -----------
        material_key : int
            Material group key
        proxy_type : int
            Proxy shape type (0, 1, 2)
        size_scale : float
            Normalized size (0-1)

        Returns:
        --------
        np.ndarray : Interpolated IR (max_ir_length)
        """
        key = (material_key, proxy_type)

        if key not in self.ir_table:
            debug_print(f"IR table not found for key {key}")
            return np.zeros(self.max_ir_length, dtype=np.float32)

        # Clamp size scale
        size_scale = np.clip(size_scale, 0, 1)

        # Get size steps for this key
        size_steps = self.size_steps.get(key)
        if size_steps is None:
            return self.ir_table[key][0]

        # Find interpolation indices
        n_steps = len(size_steps)
        idx = size_scale * (n_steps - 1)
        idx_low = int(np.floor(idx))
        idx_high = min(idx_low + 1, n_steps - 1)
        frac = idx - idx_low

        # Linear interpolation between size steps
        ir_low = self.ir_table[key][idx_low]
        ir_high = self.ir_table[key][idx_high]

        # Vectorized interpolation
        ir = ir_low * (1 - frac) + ir_high * frac

        return ir

    def save(self, filepath: str) -> None:
        """Save IR table to file."""
        save_data = {
            'ir_table': self.ir_table,
            'size_steps': self.size_steps,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'sample_rate': self.sample_rate
        }
        np.savez_compressed(filepath, **save_data)

    def load(self, filepath: str) -> None:
        """Load IR table from file."""
        data = np.load(filepath, allow_pickle=True)
        self.ir_table = data['ir_table'].item()
        self.size_steps = data['size_steps'].item()
        self.min_size = data['min_size'].item()
        self.max_size = data['max_size'].item()
        self.sample_rate = data['sample_rate']
