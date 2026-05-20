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
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import trimesh

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _parse_lib
from rigidBody import ModalBank

from .fracture_data import FractureEvent, FragmentData

@dataclass
class FractureModalModel:
    """
    Compute modal models for fracture fragments based on the original object's
    modal properties and the fracture pattern.
    
    Implements the modal modifications described in the FractureSound paper:
    - Frequency shifting due to changed geometry
    - Damping changes due to new boundaries
    - Mode coupling at fracture interface
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.dsp_path = f"{config.system.cache_path}/dsp"
        self.fracture_modal_path = f"{config.system.cache_path}/fracture_modal"
        os.makedirs(self.fracture_modal_path, exist_ok=True)
    
    def compute_fragment_modal(self, event: FractureEvent, fragment_idx: int) -> None:
        """
        Compute modified modal model for a fracture fragment.
        
        Parameters:
        -----------
        event : FractureEvent
            The fracture event
        fragment_idx : int
            Index of the fragment to compute modal model for
        """
        config = self.entity_manager.get('config')
        
        # Get fragment object config
        fragment_obj = None
        for obj in config.objects:
            if obj.idx == fragment_idx:
                fragment_obj = obj
                break
        
        if not fragment_obj:
            raise ValueError(f"Fragment {fragment_idx} not found")
        
        # Load original modal model
        original_lib = f"{self.dsp_path}/{fragment_obj.name}.lib"
        
        # If fragment modal doesn't exist yet, create modified version
        fragment_lib = f"{self.fracture_modal_path}/{fragment_obj.name}_fracture.lib"
        
        if os.path.exists(fragment_lib):
            return  # Already computed
        
        # Parse original modal data
        modal_data = _parse_lib(original_lib)
        
        # Get fragment geometry
        fragment_geo = self._get_fragment_geometry(event, fragment_idx)
        
        # Apply frequency modifications based on fragment size
        modified_frequencies = self._modify_frequencies(
            modal_data['frequencies'],
            fragment_geo,
            event
        )
        
        # Apply damping modifications
        modified_t60s = self._modify_damping(
            modal_data['t60s'],
            fragment_geo,
            event
        )
        
        # Apply gain modifications (mode shapes affected by new boundaries)
        modified_gains = self._modify_gains(
            modal_data['gains'],
            fragment_geo,
            event
        )
        
        # Create modified modal model
        self._write_fracture_lib(
            fragment_lib,
            fragment_obj.name,
            modified_frequencies,
            modified_t60s,
            modified_gains
        )
        
        print(f"Created fracture modal model for {fragment_obj.name}")
    
    def _get_fragment_geometry(self, event: FractureEvent, fragment_idx: int) -> FragmentData:
        """Get or compute fragment geometry data."""
        # Check if already computed
        if fragment_idx == event.fragment1_idx and event.fragment1_data:
            return event.fragment1_data
        elif fragment_idx == event.fragment2_idx and event.fragment2_data:
            return event.fragment2_data
        
        # Compute fragment geometry
        config = self.entity_manager.get('config')
        
        # Get fragment object
        fragment_obj = None
        for obj in config.objects:
            if obj.idx == fragment_idx:
                fragment_obj = obj
                break
        
        if not fragment_obj:
            raise ValueError(f"Fragment {fragment_idx} not found")
        
        # Load fragment mesh at fracture frame
        from physicsSolver.lib.functions import _load_mesh
        vertices, normals, faces = _load_mesh(fragment_obj, int(event.frame))
        
        # Create trimesh for volume calculation
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.density = fragment_obj.acoustic_shader.density
        
        fragment_data = FragmentData(
            obj_idx=fragment_idx,
            vertices=vertices,
            normals=normals,
            faces=faces,
            mass=mesh.mass,
            volume=mesh.volume,
            center_of_mass=mesh.center_mass,
            inertia_tensor=mesh.moment_inertia
        )
        
        # Store in event
        if fragment_idx == event.fragment1_idx:
            event.fragment1_data = fragment_data
        else:
            event.fragment2_data = fragment_data
        
        return fragment_data
    
    def _modify_frequencies(self, original_freqs: np.ndarray, fragment: FragmentData, event: FractureEvent) -> np.ndarray:
        """
        Modify modal frequencies based on fragment size and shape.
        
        According to fracture sound theory, frequencies shift proportionally
        to 1/size for thin shells, and more complex for 3D objects.
        """
        # Get original object size (approximate radius)
        original_obj = None
        config = self.entity_manager.get('config')
        for obj in config.objects:
            if obj.idx == event.original_obj_idx:
                original_obj = obj
                break
        
        if not original_obj:
            return original_freqs
        
        # Load original mesh to get size
        from physicsSolver.lib.functions import _load_mesh
        orig_vertices, _, _ = _load_mesh(original_obj, int(event.frame))
        
        # Compute bounding sphere radii
        orig_center = np.mean(orig_vertices, axis=0)
        orig_radius = np.max(np.linalg.norm(orig_vertices - orig_center, axis=1))
        
        frag_radius = np.max(np.linalg.norm(fragment.vertices - fragment.center_of_mass, axis=1))
        
        # Frequency scaling factor (inverse of size ratio)
        size_ratio = orig_radius / frag_radius if frag_radius > 0 else 1.0
        freq_scale = size_ratio
        
        # Add stochastic variation based on fracture type
        if event.fracture_type.value == 'shatter':
            # Shatter causes more random frequency shifts
            stochastic_factor = 1.0 + 0.1 * np.random.randn(len(original_freqs))
        elif event.fracture_type.value == 'crack':
            # Crack causes systematic shift
            stochastic_factor = 1.0 + 0.02 * np.random.randn(len(original_freqs))
        else:
            stochastic_factor = 1.0 + 0.05 * np.random.randn(len(original_freqs))
        
        modified_freqs = original_freqs * freq_scale * stochastic_factor
        
        # Ensure frequencies are within reasonable range
        modified_freqs = np.clip(modified_freqs, 20, 20000)
        
        return modified_freqs
    
    def _modify_damping(self, original_t60s: np.ndarray, fragment: FragmentData, event: FractureEvent) -> np.ndarray:
        """
        Modify damping (T60) based on new boundaries and radiation.
        
        Fracture creates new boundaries which increase damping due to
        energy radiation from the crack.
        """
        # Get size ratio
        config = self.entity_manager.get('config')
        original_obj = None
        for obj in config.objects:
            if obj.idx == event.original_obj_idx:
                original_obj = obj
                break
        
        if not original_obj:
            return original_t60s
        
        from physicsSolver.lib.functions import _load_mesh
        orig_vertices, _, _ = _load_mesh(original_obj, int(event.frame))
        
        orig_surface_area = self._estimate_surface_area(orig_vertices)
        frag_surface_area = self._estimate_surface_area(fragment.vertices)
        
        # New boundaries increase damping
        area_ratio = frag_surface_area / orig_surface_area if orig_surface_area > 0 else 1.0
        
        # Damping increases with new surface area
        damping_factor = 1.0 + 0.5 * (area_ratio - 1.0)
        
        # Fracture type affects damping
        if event.fracture_type.value == 'shatter':
            damping_factor *= 1.5  # More damping for shatter
        elif event.fracture_type.value == 'crack':
            damping_factor *= 1.2  # Moderate damping increase
        
        modified_t60s = original_t60s / damping_factor  # Shorter T60 = more damping
        
        return modified_t60s
    
    def _modify_gains(self, original_gains: List[np.ndarray], fragment: FragmentData, event: FractureEvent) -> List[np.ndarray]:
        """
        Modify modal gains based on new mode shapes.
        
        This is a simplified approach - full mode shape recomputation would
        require solving the eigenproblem for the new geometry.
        """
        modified_gains = []
        
        for gains in original_gains:
            # Apply gain scaling based on fragment size
            # Smaller fragments generally have lower amplitude
            size_scale = np.mean(np.linalg.norm(fragment.vertices, axis=1))
            size_scale = size_scale / 0.1  # Normalize to 0.1m reference
            
            # Add stochastic variation
            stochastic = 1.0 + 0.1 * np.random.randn(len(gains))
            
            modified_gain = gains * size_scale * stochastic
            modified_gains.append(modified_gain)
        
        return modified_gains
    
    def _estimate_surface_area(self, vertices: np.ndarray) -> float:
        """Estimate surface area from vertex cloud."""
        # Simple bounding sphere surface area
        center = np.mean(vertices, axis=0)
        radius = np.max(np.linalg.norm(vertices - center, axis=1))
        return 4 * np.pi * radius**2
    
    def _write_fracture_lib(self, filename: str, obj_name: str, frequencies: np.ndarray, t60s: np.ndarray, gains: List[np.ndarray]):
        """
        Write fracture modal model in Faust .lib format.
        """
        n_modes = len(frequencies)
        
        with open(filename, 'w') as f:
            f.write(f'// Fracture modal model for {obj_name}\n')
            f.write('// Automatically generated by fractureSound\n\n')
            
            # Write frequencies
            f.write('modeFreqsUnscaled = ba.take(' + str(n_modes) + ', ')
            f.write('(' + ', '.join([f'{freq:.6f}' for freq in frequencies]) + '));\n\n')
            
            # Write T60s
            f.write('modesT60s = t60Scale * ba.take(' + str(n_modes) + ', ')
            f.write('(' + ', '.join([f'{t60:.6f}' for t60 in t60s]) + '));\n\n')
            
            # Write gains (flattened)
            flat_gains = []
            for gain_set in gains:
                flat_gains.extend(gain_set)
            
            f.write('modesGains = waveform{')
            f.write(', '.join([f'{gain:.10f}' for gain in flat_gains]))
            f.write('} : ro.hyperplane(' + str(n_modes) + ');\n')
