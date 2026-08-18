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

from ..lib.voxel_mesh_modal_analysis import VoxelMeshModalAnalysis
from pbrAudioCommon import _compute_rayleigh_damping

@dataclass
class Approx2Faust:
    """
    Generate modal models from Voxel-based approximation of mesh.
    """
    
    def __post_init__(self):
        pass
    
    def compute(self, vertices: np.ndarray, faces: np.ndarray, young_modulus: float, poisson_ratio: float, density: float, damping: float, min_freq: float, max_freq: float, n_modes: int, voxel_size: float) -> ModalParameters:
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

        mesh = trimesh.Trimesh(vertices, faces=faces)
        # Voxelize the mesh
        voxel_grid = self._voxelize_mesh(mesh, voxel_size)

        rayleigh_alpha, rayleigh_beta = _compute_rayleigh_damping(min_freq, max_freq, damping)

        material_properties = {
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'density': density,
            'rayleigh_alpha': rayleigh_alpha,
            'rayleigh_beta': rayleigh_beta
        }

        # Create modal analysis object
        modal_analysis = VoxelMeshModalAnalysis(voxel_grid, voxel_size, material_properties)

        # Compute modes
        frequencies, mode_shapes = modal_analysis.compute_modes(n_modes=n_modes, min_freq=min_freq, max_freq=max_freq)

        # Prepare results for Faust
        results = self._prepare_faust_output(frequencies, mode_shapes, modal_analysis, voxel_size, material_properties, min_freq, max_freq)

        return results

    def _voxelize_mesh(self, mesh: trimesh.Trimesh, voxel_size: float) -> np.ndarray:
        """
        Convert mesh to voxel grid

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Input mesh

        Returns:
        --------
        np.ndarray
            3D boolean array representing voxelized mesh
        """
        # Get mesh bounds
        bounds = mesh.bounds
        min_bound = bounds[0]
        max_bound = bounds[1]

        # Calculate grid dimensions
        dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 2
        dims = np.maximum(dims, 1)

        # Voxelize using trimesh
        voxelized = mesh.voxelized(voxel_size)

        # Convert to boolean grid
        voxel_grid = np.zeros(dims, dtype=bool)

        # Get voxel indices from trimesh voxelization
        if hasattr(voxelized, 'points'):
            # For trimesh VoxelGrid
            voxel_matrix = voxelized.matrix
            # Pad the matrix to match our dimensions
            voxel_grid[:voxel_matrix.shape[0],
                      :voxel_matrix.shape[1],
                      :voxel_matrix.shape[2]] = voxel_matrix
        else:
            # Fallback: use filled voxelization
            voxel_grid = voxelized.fill().matrix

        return voxel_grid

    def _prepare_faust_output(self, frequencies: np.ndarray, mode_shapes: np.ndarray, modal_analysis: VoxelMeshModalAnalysis, voxel_size: float, material_properties: Dict[str, Any], min_freq: float, max_freq: float) -> Dict[str, Any]:
        """
        Prepare modal analysis results for Faust synthesis

        Parameters:
        -----------
        frequencies : np.ndarray
            Natural frequencies
        mode_shapes : np.ndarray
            Mode shapes
        modal_analysis : VoxelMeshModalAnalysis
            Modal analysis object with material properties

        Returns:
        --------
        dict
            Faust-compatible modal synthesis parameters
        """
        if len(frequencies) == 0:
            return {
                'modes': [],
                'metadata': {
                    'voxel_size': voxel_size,
                    'n_voxels': int(np.sum(modal_analysis.voxel_grid)),
                    'material': material_properties
                }
            }

        # Prepare mode data for Faust
        modes = []
        for i, (freq, shape) in enumerate(zip(frequencies, mode_shapes.T)):
            # Calculate mode parameters
            omega = 2 * np.pi * freq
            damping = (material_properties['rayleigh_alpha'] / (2 * omega) +
                      material_properties['rayleigh_beta'] * omega / 2)

            mode_data = {
                'frequency': float(freq),
                'damping': float(damping),
                'amplitude': 1.0,  # Default amplitude
                'mode_shape': shape.tolist() if len(shape) < 1000 else []
            }
            modes.append(mode_data)

        return {
            'modes': modes,
            'metadata': {
                'voxel_size': voxel_size,
                'n_voxels': int(np.sum(modal_analysis.voxel_grid)),
                'n_dofs': len(mode_shapes),
                'material': material_properties,
                'frequency_range': [float(min_freq), float(max_freq)]
            }
        }

    def to_faust_lib(self, results: Dict[str, Any], output_name: str, min_freq: float, max_freq: float) -> str:
        """
        Generate Faust .lib file content for modal synthesis
        
        Parameters:
        -----------
        results : dict
            Modal analysis results from compute()
        output_name : str
            Name for the Faust output
        min_freq : float
            Minimum frequency for the modal model
        max_freq : float
            Maximum frequency for the modal model
        
        Returns:
        --------
        str
            Faust .lib file content
        """
        modes = results.get('modes', [])
        metadata = results.get('metadata', {})
        n_modes = len(modes)
        n_voxels = metadata.get('n_voxels', 0)
        n_vertices = metadata.get('n_dofs', 1)  # Use DOFs as proxy for vertices
        
        # Filter modes by frequency range if specified
        filtered_modes = []
        for mode in modes:
            freq = mode['frequency']
            if min_freq <= freq <= max_freq:
                filtered_modes.append(mode)
        
        n_filtered_modes = len(filtered_modes)
        
        if n_filtered_modes == 0:
            # Return empty lib with zero modes
            return self._generate_empty_lib(output_name, min_freq, max_freq, n_voxels)
        
        # Prepare frequency string
        freq_values = [f"{mode['frequency']:.6f}" for mode in filtered_modes]
        freq_str = ", ".join(freq_values)
        
        # Calculate T60 decay times
        # T60 = ln(1000) / damping ≈ 6.9078 / damping
        t60_values = []
        for mode in filtered_modes:
            damping = mode['damping']
            if damping > 0:
                t60 = 6.9078 / damping
            else:
                t60 = 10.0  # Default T60 for no damping
            t60_values.append(f"{t60:.6f}")
        t60_str = ", ".join(t60_values)
        
        # Calculate T60 scale factor (needed for Faust implementation)
        # This is typically used to convert between different damping representations
        t60_scale = 1.0 / (6.9078 * max_freq) if max_freq > 0 else 1.0
        t60_scale_str = f"{t60_scale:.10f}"
        
        # Prepare gains matrix (nModes x nExPos)
        # For now, use uniform gains since we don't have position-specific data
        # In a full implementation, this would be computed from mode shapes and excitation positions
        gain_lines = []
        
        # Header for gains matrix
        gain_lines.append(f"// Mode gains matrix ({n_filtered_modes} modes x {n_vertices} positions)")
        gain_lines.append(f"// Using uniform gains as placeholder")
        
        # Generate gain values
        # Each mode gets a row in the matrix
        gain_values = []
        for i, mode in enumerate(filtered_modes):
            # For each mode, create gains for each excitation position
            # Using 1.0 as default gain for all positions
            mode_gains = [f"{mode['amplitude']:.6f}"] * n_vertices
            gain_values.append(f"        {', '.join(mode_gains)}")
        
        gain_str = ",\n".join(gain_values)
        
        # Build the complete Faust lib content
        faust_lib = f"""// ------------------------------------------------------------
    // Voxel approximated mesh modal model for {output_name}
    // Generated by Approx2Faust fallback mechanism
    // Modes: {n_filtered_modes}, Voxels: {n_voxels}
    // Frequency range: {min_freq:.1f} - {max_freq:.1f} Hz
    // ------------------------------------------------------------

    declare name        "{output_name}";
    declare version     "0.1";
    declare author      "Approx2Faust";
    declare license     "GPL";

    import("stdfaust.lib");

    // Modal parameters
    nModes = {n_filtered_modes};
    nExPos = {n_vertices};

    // T60 scale factor for damping calculation
    t60Scale = {t60_scale_str};

    // Mode frequencies (Hz)
    modeFreqsUnscaled = ba.take(nModes, ({freq_str}));

    // T60 decay times (seconds)
    modesT60s = t60Scale : ba.take(nModes, ({t60_str}));

    // Mode gains (nModes x nExPos)
    modesGains = waveform{{
    {gain_str}
    }};

    // Process function
    process = no.process;
    """
        
        return faust_lib

    def _generate_empty_lib(self, output_name: str, min_freq: float, 
                           max_freq: float, n_voxels: int) -> str:
        """
        Generate empty Faust lib when no modes are found
        
        Parameters:
        -----------
        output_name : str
            Name for the Faust output
        min_freq : float
            Minimum frequency
        max_freq : float
            Maximum frequency
        n_voxels : int
            Number of voxels
        
        Returns:
        --------
        str
            Empty Faust .lib file content
        """
        faust_lib = f"""// ------------------------------------------------------------
    // Voxel approximated mesh modal model for {output_name}
    // Generated by Approx2Faust fallback mechanism
    // Modes: 0, Voxels: {n_voxels}
    // Frequency range: {min_freq:.1f} - {max_freq:.1f} Hz
    // ------------------------------------------------------------

    declare name        "{output_name}";
    declare version     "0.1";
    declare author      "Approx2Faust";
    declare license     "GPL";

    import("stdfaust.lib");

    // Modal parameters
    nModes = 0;
    nExPos = 0;

    // T60 scale factor
    t60Scale = 1.0;

    // Mode frequencies (Hz)
    modeFreqsUnscaled = ba.take(nModes, ());

    // T60 decay times (seconds)
    modesT60s = t60Scale : ba.take(nModes, ());

    // Mode gains (nModes x nExPos)
    modesGains = waveform{{}};

    // Process function
    process = no.process;
    """
        return faust_lib
