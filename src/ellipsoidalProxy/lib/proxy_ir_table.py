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

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _parse_lib, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix


@dataclass
class ProxyIRTable:
    """
    Precomputed Impulse Response table for proxy meshes using Toeplitz matrix structure.
    
    The table is indexed by:
    - Material group (based on acoustic properties)
    - Proxy shape type (pyramid=0, octahedron=1, cube=2)
    - Size scale (interpolated between min and max size)
    
    Each IR is stored as a Toeplitz matrix of shape (num_modes, max_ir_length)
    where each row corresponds to a modal frequency band.
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
    num_modes: Dict[Tuple[int, int], int] = field(default_factory=dict)  # Number of modes per entry

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
        
        Each IR is stored as a Toeplitz matrix of shape (num_modes, max_ir_length).
        Each row is the impulse response for a specific modal frequency band.

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
            debug_print(f"Building IR table for material {material_key}, shape {self._shape_names[proxy_type]}")

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

            # Build Toeplitz IR table for this group
            ir_matrix, num_modes = self._build_toeplitz_ir_matrix(
                meshes=meshes,
                material_key=material_key,
                proxy_type=proxy_type,
                material_props=material_props,
                size_steps=size_steps,
                n_steps=n_steps
            )

            self.ir_table[(material_key, proxy_type)] = ir_matrix
            self.num_modes[(material_key, proxy_type)] = num_modes

            debug_print(f"Built Toeplitz IR table for material {material_key}, shape {self._shape_names[proxy_type]}, "
                       f"steps: {n_steps}, modes: {num_modes}, size range: {self.min_size[(material_key, proxy_type)]:.4f} - "
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
    ) -> Tuple[np.ndarray, int]:
        """
        Build a Toeplitz impulse response matrix.

        The matrix has shape (num_modes, max_ir_length) where each row is
        the impulse response for a specific modal frequency band.

        Uses the modal parameters from the .lib file to generate the IR.

        Returns:
            (ir_matrix, num_modes) where ir_matrix has shape (num_modes, max_ir_length)
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
            return np.zeros((1, self.max_ir_length), dtype=np.float32), 1

        # Sort by size
        all_modal_params.sort(key=lambda x: x['size'])

        # Determine number of modes (use the maximum from all meshes)
        num_modes = max(p['nModes'] for p in all_modal_params)
        debug_print(f"Using {num_modes} modes for Toeplitz matrix")

        # Build IR for each size step
        # Shape: (n_steps, num_modes, max_ir_length)
        ir_matrix = np.zeros((n_steps, num_modes, self.max_ir_length), dtype=np.float32)

        # Determine min and max sizes
        sizes = np.array([p['size'] for p in all_modal_params])
        min_s = self.min_size[(material_key, proxy_type)]
        max_s = self.max_size[(material_key, proxy_type)]

        for step_idx, size_scale in enumerate(size_steps):
            target_size = min_s + size_scale * (max_s - min_s)

            # Find nearest modal parameters with interpolation
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
                    sizes[idx2],
                    num_modes
                )

            # Generate IR for each mode (Toeplitz matrix rows)
            mode_irs = self._modal_params_to_toeplitz_rows(params, material_props, num_modes)
            
            # Store in matrix
            ir_matrix[step_idx] = mode_irs

        return ir_matrix, num_modes

    def _interpolate_modal_params(
        self,
        params1: Dict,
        params2: Dict,
        target_size: float,
        size1: float,
        size2: float,
        num_modes: int
    ) -> Dict:
        """
        Interpolate modal parameters between two size points.
        """
        if size2 == size1:
            return params1

        # Weight based on distance in size space
        w1 = 1.0 - (target_size - size1) / (size2 - size1)
        w2 = 1.0 - w1

        # Get frequencies
        freqs1 = params1['frequencies']
        freqs2 = params2['frequencies']
        
        # Pad or truncate to num_modes
        if len(freqs1) < num_modes:
            freqs1 = np.pad(freqs1, (0, num_modes - len(freqs1)), mode='edge')
        else:
            freqs1 = freqs1[:num_modes]
            
        if len(freqs2) < num_modes:
            freqs2 = np.pad(freqs2, (0, num_modes - len(freqs2)), mode='edge')
        else:
            freqs2 = freqs2[:num_modes]

        # Interpolate frequencies
        frequencies = w1 * freqs1 + w2 * freqs2

        # Interpolate T60s
        t60s1 = params1['t60s']
        t60s2 = params2['t60s']
        
        if len(t60s1) < num_modes:
            t60s1 = np.pad(t60s1, (0, num_modes - len(t60s1)), mode='edge')
        else:
            t60s1 = t60s1[:num_modes]
            
        if len(t60s2) < num_modes:
            t60s2 = np.pad(t60s2, (0, num_modes - len(t60s2)), mode='edge')
        else:
            t60s2 = t60s2[:num_modes]
            
        t60s = w1 * t60s1 + w2 * t60s2

        # Interpolate gains (per vertex, average over vertices)
        gains1 = params1['gains']
        gains2 = params2['gains']
        
        # Average gains across vertices for each mode
        if gains1 and len(gains1) > 0:
            avg_gains1 = np.array([np.mean(g) if isinstance(g, (list, np.ndarray)) else g for g in gains1])
        else:
            avg_gains1 = np.ones(num_modes) * 0.1
            
        if gains2 and len(gains2) > 0:
            avg_gains2 = np.array([np.mean(g) if isinstance(g, (list, np.ndarray)) else g for g in gains2])
        else:
            avg_gains2 = np.ones(num_modes) * 0.1
            
        if len(avg_gains1) < num_modes:
            avg_gains1 = np.pad(avg_gains1, (0, num_modes - len(avg_gains1)), mode='edge')
        else:
            avg_gains1 = avg_gains1[:num_modes]
            
        if len(avg_gains2) < num_modes:
            avg_gains2 = np.pad(avg_gains2, (0, num_modes - len(avg_gains2)), mode='edge')
        else:
            avg_gains2 = avg_gains2[:num_modes]
            
        gains = w1 * avg_gains1 + w2 * avg_gains2

        return {
            'frequencies': frequencies,
            't60s': t60s,
            'gains': gains,
            'nModes': num_modes
        }

    def _modal_params_to_toeplitz_rows(
        self,
        modal_params: Dict,
        material_props: Dict[str, float],
        num_modes: int
    ) -> np.ndarray:
        """
        Generate Toeplitz matrix rows from modal parameters.

        Each row is the impulse response for a specific mode.
        The matrix has shape (num_modes, max_ir_length).

        Returns:
            np.ndarray of shape (num_modes, max_ir_length)
        """
        frequencies = modal_params['frequencies']
        t60s = modal_params['t60s']
        gains = modal_params['gains']

        # Ensure we have the right number of modes
        if len(frequencies) < num_modes:
            frequencies = np.pad(frequencies, (0, num_modes - len(frequencies)), mode='edge')
            t60s = np.pad(t60s, (0, num_modes - len(t60s)), mode='edge')
            if isinstance(gains, (list, np.ndarray)):
                if len(gains) < num_modes:
                    gains = np.pad(gains, (0, num_modes - len(gains)), mode='edge')
        else:
            frequencies = frequencies[:num_modes]
            t60s = t60s[:num_modes]
            if isinstance(gains, (list, np.ndarray)):
                gains = gains[:num_modes]

        # Damping factor from material properties
        damping = material_props.get('damping', 0.02)

        # Initialize IR matrix (num_modes, max_ir_length)
        ir_matrix = np.zeros((num_modes, self.max_ir_length), dtype=np.float32)

        # Time axis
        t = np.arange(self.max_ir_length) / self.sample_rate

        # Build each mode's impulse response
        for mode_idx in range(num_modes):
            freq = frequencies[mode_idx]
            t60 = t60s[mode_idx]

            if freq <= 0 or t60 <= 0:
                continue

            # Decay rate
            decay = 3 * np.log(10) / max(t60, 0.001)

            # Get gain for this mode
            if isinstance(gains, (list, np.ndarray)) and mode_idx < len(gains):
                gain = gains[mode_idx]
                if isinstance(gain, (list, np.ndarray)):
                    gain = np.mean(gain)
            else:
                gain = 0.1

            # Damped sinusoid for this mode
            # Toeplitz row: impulse response of a second-order filter
            mode_ir = gain * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)

            # Add to matrix row
            ir_matrix[mode_idx] = mode_ir

        # Apply overall damping scaling
        ir_matrix *= (1.0 - damping * 0.5)

        # Normalize each row
        for mode_idx in range(num_modes):
            max_val = np.max(np.abs(ir_matrix[mode_idx]))
            if max_val > 0:
                ir_matrix[mode_idx] = ir_matrix[mode_idx] / max_val * 0.9

        return ir_matrix

    def get_ir(self, material_key: int, proxy_type: int, size_scale: float) -> np.ndarray:
        """
        Get interpolated IR matrix for a given material, shape, and size scale.

        Returns a Toeplitz matrix of shape (num_modes, max_ir_length).

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
        np.ndarray : Interpolated IR matrix (num_modes, max_ir_length)
        """
        key = (material_key, proxy_type)

        if key not in self.ir_table:
            debug_print(f"IR table not found for key {key}")
            return np.zeros((1, self.max_ir_length), dtype=np.float32)

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

    def get_num_modes(self, material_key: int, proxy_type: int) -> int:
        """Get the number of modes for a given material and shape."""
        key = (material_key, proxy_type)
        return self.num_modes.get(key, 1)

    def save(self, filepath: str) -> None:
        """Save IR table to file."""
        save_data = {
            'ir_table': self.ir_table,
            'size_steps': self.size_steps,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'num_modes': self.num_modes,
            'sample_rate': self.sample_rate,
            'max_ir_length': self.max_ir_length
        }
        np.savez_compressed(filepath, **save_data)

    def load(self, filepath: str) -> None:
        """Load IR table from file."""
        data = np.load(filepath, allow_pickle=True)
        self.ir_table = data['ir_table'].item()
        self.size_steps = data['size_steps'].item()
        self.min_size = data['min_size'].item()
        self.max_size = data['max_size'].item()
        self.num_modes = data['num_modes'].item()
        self.sample_rate = int(data['sample_rate'])
        self.max_ir_length = int(data['max_ir_length'])
