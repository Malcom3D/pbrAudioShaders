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

# pbrAudioShaders/src/fractureSound/lib/fracture_modal.py
"""
Modal model adaptation for fracture fragments.
"""

import os
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import trimesh

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _parse_lib, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

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
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        self.dsp_path = f"{config.system.cache_path}/dsp"
        self.fracture_modal_path = f"{config.system.cache_path}/fracture_modal"
        os.makedirs(self.fracture_modal_path, exist_ok=True)
    
    def compute(self, event: FractureEvent, fragment_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute modified modal model for a fracture fragment.
        
        Parameters:
        -----------
        event : FractureEvent
            The fracture event
        fragment_idx : int
            Index of the fragment to compute modal model for
            
        Returns:
        --------
        Dict with modal parameters, or None if failed
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
        
        # Get fragment data from event
        fragment_data = event.get_fragment_by_idx(fragment_idx)
        if fragment_data is None:
            # Try to load from saved data
            fragment_data = self._load_fragment_data(event, fragment_idx)
        
        if fragment_data is None:
            debug_print(f"Could not get fragment data for {fragment_idx}")
            return None
        
        # Check if fracture modal already exists
        lib_file = f"{self.fracture_modal_path}/{fragment_obj.name}_fracture.lib"
        if os.path.exists(lib_file):
            return _parse_lib(lib_file)
        
        # Load original modal model
        original_lib = f"{self.dsp_path}/{fragment_obj.name}.lib"
        if fragment_obj.proxy_type is not False:
            original_lib = f"{self.dsp_path}/{fragment_obj.name}_proxy_{fragment_obj.proxy_type}.lib"
        
        if not os.path.exists(original_lib):
            debug_print(f"Original modal model not found for {fragment_obj.name}")
            return None
        
        # Parse original modal data
        modal_data = _parse_lib(original_lib)
        
        # Apply frequency modifications based on fragment size
        modified_frequencies = self._modify_frequencies(modal_data['frequencies'], fragment_data, event)
        
        # Apply damping modifications
        modified_t60s = self._modify_damping(modal_data['t60s'], fragment_data, event)
        
        # Apply gain modifications (mode shapes affected by new boundaries)
        modified_gains = self._modify_gains(modal_data['gains'], fragment_data, event)
        
        # Create modified modal model
        self._write_fracture_lib(lib_file, fragment_obj.name, modified_frequencies, modified_t60s, modified_gains)
        
        debug_print(f"Created fracture modal model for {fragment_obj.name}")
        
        return {
            'frequencies': modified_frequencies,
            't60s': modified_t60s,
            'gains': modified_gains
        }
    
    def _load_fragment_data(self, event: FractureEvent, fragment_idx: int) -> Optional[FragmentData]:
        """Load fragment data from event or compute it."""
        # Check if already in event
        for frag_data in event.fragment_data:
            if frag_data.obj_idx == fragment_idx:
                return frag_data
        
        # Try to compute from trajectory
        config = self.entity_manager.get('config')
        trajectories = self.entity_manager.get('trajectories')
        
        for traj in trajectories.values():
            if hasattr(traj, 'obj_idx') and traj.obj_idx == fragment_idx:
                # Get fragment geometry at fracture frame
                vertices = traj.get_vertices(event.frame)
                normals = traj.get_normals(event.frame)
                faces = traj.get_faces()
                
                # Get material properties
                density = event.density
                
                # Compute fragment properties
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                mesh.density = density
                
                fragment_data = FragmentData(
                    obj_idx=fragment_idx,
                    obj_name=f"fragment_{fragment_idx}",
                    vertices=vertices,
                    normals=normals,
                    faces=faces,
                    mass=mesh.mass,
                    volume=mesh.volume,
                    center_of_mass=mesh.center_mass,
                    inertia_tensor=mesh.moment_inertia,
                    parent_obj_idx=event.original_obj_idx,
                    is_shard=True,
                    fracture_frame=event.frame
                )
                
                event.fragment_data.append(fragment_data)
                return fragment_data
        
        return None
    
    def _modify_frequencies(self, original_freqs: np.ndarray, fragment: FragmentData, event: FractureEvent) -> np.ndarray:
        """
        Modify modal frequencies based on fragment size and shape.
        
        According to fracture sound theory, frequencies shift proportionally
        to 1/size for thin shells, and more complex for 3D objects.
        """
        # Get original object size (approximate radius)
        original_size = self._get_object_size(event.original_obj_idx, event.frame)
        fragment_size = self._get_fragment_size(fragment)
        
        # Frequency scaling factor (inverse of size ratio)
        if fragment_size > 0 and original_size > 0:
            size_ratio = original_size / fragment_size
            freq_scale = size_ratio
        else:
            freq_scale = 1.0
        
        # Add stochastic variation based on fracture type
        if event.fracture_type.value == 'shatter':
            # Shatter causes more random frequency shifts
            stochastic_factor = 1.0 + 0.15 * np.random.randn(len(original_freqs))
        elif event.fracture_type.value == 'crack':
            # Crack causes systematic shift
            stochastic_factor = 1.0 + 0.05 * np.random.randn(len(original_freqs))
        else:  # snap
            stochastic_factor = 1.0 + 0.08 * np.random.randn(len(original_freqs))
        
        modified_freqs = original_freqs * freq_scale * stochastic_factor
        
        # Ensure frequencies are within reasonable range
        modified_freqs = np.clip(modified_freqs, 5, 22000)
        
        return modified_freqs
    
    def _modify_damping(self, original_t60s: np.ndarray, 
                        fragment: FragmentData, 
                        event: FractureEvent) -> np.ndarray:
        """
        Modify damping (T60) based on new boundaries and radiation.
        
        Fracture creates new boundaries which increase damping due to
        energy radiation from the crack.
        """
        # Get size ratio
        original_size = self._get_object_size(event.original_obj_idx, event.frame)
        fragment_size = self._get_fragment_size(fragment)
        
        if fragment_size > 0 and original_size > 0:
            size_ratio = fragment_size / original_size
            # Smaller fragments have shorter T60 (more damping)
            damping_factor = size_ratio ** 0.5
        else:
            damping_factor = 1.0
        
        # Fracture type affects damping
        if event.fracture_type.value == 'shatter':
            damping_factor *= 0.6  # More damping for shatter
        elif event.fracture_type.value == 'crack':
            damping_factor *= 0.8  # Moderate damping increase
        else:  # snap
            damping_factor *= 0.7  # Significant damping for snap
        
        modified_t60s = original_t60s * damping_factor
        
        # Clamp to reasonable range
        modified_t60s = np.clip(modified_t60s, 0.001, 10.0)
        
        return modified_t60s
    
    def _modify_gains(self, original_gains: List[np.ndarray], 
                      fragment: FragmentData, 
                      event: FractureEvent) -> List[np.ndarray]:
        """
        Modify modal gains based on new mode shapes.
        
        This is a simplified approach - full mode shape recomputation would
        require solving the eigenproblem for the new geometry.
        """
        modified_gains = []
        
        # Get size scaling
        original_size = self._get_object_size(event.original_obj_idx, event.frame)
        fragment_size = self._get_fragment_size(fragment)
        
        if fragment_size > 0 and original_size > 0:
            # Smaller fragments generally have lower amplitude
            size_scale = (fragment_size / original_size) ** 1.5
        else:
            size_scale = 0.5
        
        # Fracture type scaling
        if event.fracture_type.value == 'shatter':
            type_scale = 0.7
        elif event.fracture_type.value == 'crack':
            type_scale = 0.9
        else:  # snap
            type_scale = 0.8
        
        for gains in original_gains:
            # Add stochastic variation
            stochastic = 1.0 + 0.1 * np.random.randn(len(gains))
            modified_gain = gains * size_scale * type_scale * stochastic
            modified_gains.append(modified_gain)
        
        return modified_gains
    
    def _get_object_size(self, obj_idx: int, frame: float) -> float:
        """Get characteristic size of an object at a given frame."""
        config = self.entity_manager.get('config')
        trajectories = self.entity_manager.get('trajectories')
        
        # Find trajectory for this object
        for traj in trajectories.values():
            if hasattr(traj, 'obj_idx') and traj.obj_idx == obj_idx:
                vertices = traj.get_vertices(frame)
                if len(vertices) > 0:
                    # Compute bounding sphere radius
                    center = np.mean(vertices, axis=0)
                    distances = np.linalg.norm(vertices - center, axis=1)
                    return np.max(distances)
        
        # Fallback: get from config
        for obj in config.objects:
            if obj.idx == obj_idx:
                # Use bounding box size
                vertices, _, _ = _load_mesh(obj, int(frame))
                if len(vertices) > 0:
                    min_coords = np.min(vertices, axis=0)
                    max_coords = np.max(vertices, axis=0)
                    return np.linalg.norm(max_coords - min_coords) / 2
        
        return 0.1  # Default
    
    def _get_fragment_size(self, fragment: FragmentData) -> float:
        """Get characteristic size of a fragment."""
        if len(fragment.vertices) > 0:
            # Compute bounding sphere radius
            center = fragment.center_of_mass
            distances = np.linalg.norm(fragment.vertices - center, axis=1)
            return np.max(distances)
        return 0.05  # Default
    
    def _write_fracture_lib(self, filename: str, obj_name: str, 
                           frequencies: np.ndarray, t60s: np.ndarray, 
                           gains: List[np.ndarray]):
        """
        Write fracture modal model in Faust .lib format.
        """
        n_modes = len(frequencies)
        n_vertices = len(gains[0]) if gains else 1
        
        # Flatten gains
        flat_gains = []
        for gain_set in gains:
            flat_gains.extend(gain_set)
        
        with open(filename, 'w') as f:
            f.write(f'''// ------------------------------------------------------------
// Fracture modal model for {obj_name}
// Generated by FractureModalModel
// Modes: {n_modes}, Vertices: {n_vertices}
// ------------------------------------------------------------

declare name        "{obj_name}_fracture";
declare version     "0.1";
declare author      "FractureSound";
declare license     "GPL";

import("stdfaust.lib");

// Modal parameters
nModes = {n_modes};
nExPos = {n_vertices};

// Mode frequencies (Hz)
modeFreqsUnscaled = ba.take(nModes, ({", ".join([f"{f:.6f}" for f in frequencies])}));

// T60 decay times (seconds)
modesT60s = t60Scale : ba.take(nModes, ({", ".join([f"{t:.6f}" for t in t60s])}));

// Mode gains (nModes x nExPos)
modesGains = waveform{{{", ".join([f"{g:.10f}" for g in flat_gains])}}};

// Frequency scaling factor
freqScale = 1.0;

// T60 scaling factor
t60Scale = 1.0;

// Process function
process = no.process;
''')
