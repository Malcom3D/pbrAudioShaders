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
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from dask import delayed, compute

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.sound2modal import Sound2Modal, ModalParameters

@dataclass
class SamplesEngine:
    """
    Main engine for SamplesSound module.
    
    Orchestrates:
    1. Loading audio samples for each object
    2. Extracting modal parameters from audio
    3. Generating modal models for use in physicsSolver
    4. Integration with existing modal synthesis pipeline
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        self.sample_rate = config.system.sample_rate
        self.cache_path = config.system.cache_path
        self.dsp_path = f"{self.cache_path}/dsp"
        
        # Ensure output directory exists
        os.makedirs(self.dsp_path, exist_ok=True)
        
        # Initialize Sound2Modal
        self.sound2modal = Sound2Modal(entity_manager=self.entity_manager)
        
        # Map audio files to objects
        self.audio_files = {}
        self._map_audio_files()
    
    def _map_audio_files(self):
        """
               Map audio files to objects based on configuration.
        
        Audio files can be specified in:
        - Object configuration (audio_file field)
        - Convention: {object_name}.wav in audio directory
        - Custom mapping in config
        """
        config = self.entity_manager.get('config')
        
        # Check for audio directory in config
        audio_dir = f"{self.cache_path}/audio_samples"
        if hasattr(config.system, 'audio_samples_path'):
            audio_dir = config.system.audio_samples_path
        
        # Create audio directory if it doesn't exist
        os.makedirs(audio_dir, exist_ok=True)
        
        # Map audio files to objects
        for config_obj in config.objects:
            # Check if audio file is specified in config
            if hasattr(config_obj, 'audio_file') and config_obj.audio_file:
                audio_path = config_obj.audio_file
                if os.path.exists(audio_path):
                    self.audio_files[config_obj.idx] = audio_path
                    debug_print(f"Mapped audio file {audio_path} to object {config_obj.name}")
                    continue
            
            # Try convention: {object_name}.wav in audio directory
            conventional_path = f"{audio_dir}/{config_obj.name}.wav"
            if os.path.exists(conventional_path):
                self.audio_files[config_obj.idx] = conventional_path
                debug_print(f"Mapped audio file {conventional_path} to object {config_obj.name}")
                continue
            
            # Try other extensions
            for ext in ['.flac', '.ogg', '.mp3', '.aiff']:
                alt_path = f"{audio_dir}/{config_obj.name}{ext}"
                if os.path.exists(alt_path):
                    self.audio_files[config_obj.idx] = alt_path
                    debug_print(f"Mapped audio file {alt_path} to object {config_obj.name}")
                    break
    
    def bake(self) -> List[int]:
        """
        Process all objects with audio samples and generate modal models.
        
        Returns:
        --------
        List of object indices that were processed successfully
        """
        config = self.entity_manager.get('config')
        
        processed_objects = []
        
        # Process objects with audio files
        for obj_idx, audio_file in self.audio_files.items():
            debug_print(f"Processing audio for object {obj_idx}: {audio_file}")
            
            # Get object config
            config_obj = None
            for obj in config.objects:
                if obj.idx == obj_idx:
                    config_obj = obj
                    break
            
            if config_obj is None:
                debug_print(f"Object {obj_idx} not found in config")
                continue
            
            # Load mesh for position-specific gains
            vertices, faces = self._load_mesh_for_object(config_obj)
            
            # Extract modal parameters
            try:
                modal_params = self.sound2modal.compute(
                    audio_file=audio_file,
                    obj_idx=obj_idx,
                    vertices=vertices,
                    faces=faces,
                    output_name=config_obj.name
                )
                
                # Generate and save Faust .lib file
                lib_output = f"{self.dsp_path}/{config_obj.name}.lib"
                self.sound2modal.save_to_file(
                    modal_params=modal_params,
                    output_path=lib_output,
                    output_name=config_obj.name
                )
                
                # Generate resonance model if needed
                if config_obj.resonance:
                    self._generate_resonance_model(config_obj, modal_params)
                
                processed_objects.append(obj_idx)
                
            except Exception as e:
                debug_print(f"Error processing audio for object {config_obj.name}: {e}")
                continue
        
        debug_print(f"Processed {len(processed_objects)} objects with audio samples")
        return processed_objects
    
    def _load_mesh_for_object(self, config_obj: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load mesh for an object to compute position-specific gains.
        
        Returns:
        --------
        Tuple of (vertices, faces) or (None, None) if mesh cannot be loaded
        """
        try:
            from pbrAudioCommon import _load_mesh
            vertices, _, faces = _load_mesh(config_obj, 0, use_proxy_path=False)
            return vertices, faces
        except Exception as e:
            debug_print(f"Could not load mesh for {config_obj.name}: {e}")
            return None, None
    
    def _generate_resonance_model(self, config_obj: Any, modal_params: ModalParameters):
        """
        Generate a resonance model from the extracted modal parameters.
        
        Resonance models use a subset of modes with extended decay times.
        """
        # Create resonance modal parameters
        resonance_params = ModalParameters(
            frequencies=modal_params.frequencies[:config_obj.resonance_modes],
            damping_ratios=modal_params.damping_ratios[:config_obj.resonance_modes],
            t60s=modal_params.t60s[:config_obj.resonance_modes] * 1.2,  # Longer decay
            gains=modal_params.gains[:config_obj.resonance_modes, :],
            quality_factors=modal_params.quality_factors[:config_obj.resonance_modes],
            mode_shapes=modal_params.mode_shapes[:config_obj.resonance_modes, :] if modal_params.mode_shapes is not None else None,
            metadata=modal_params.metadata
        )
        
        # Save resonance model
        lib_output = f"{self.dsp_path}/{config_obj.name}_resonance.lib"
        self.sound2modal.save_to_file(
            modal_params=resonance_params,
            output_path=lib_output,
            output_name=f"{config_obj.name}_resonance"
        )
        
        debug_print(f"Generated resonance model for {config_obj.name}")
    
    def get_audio_files(self) -> Dict[int, str]:
        """Get the mapping of object indices to audio files."""
        return self.audio_files.copy()
    
    def has_audio_for_object(self, obj_idx: int) -> bool:
        """Check if an object has an associated audio file."""
        return obj_idx in self.audio_files
    
    def process_single_object(self, obj_idx: int) -> Optional[str]:
        """
        Process a single object and return the path to the generated .lib file.
        
        Parameters:
        -----------
        obj_idx : int
            Object index to process
            
        Returns:
        --------
        Optional[str]
            Path to the generated .lib file, or None if processing failed
        """
        config = self.entity_manager.get('config')
        
        if obj_idx not in self.audio_files:
            debug_print(f"No audio file found for object {obj_idx}")
            return None
        
        # Get object config
        config_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                config_obj = obj
                break
        
        if config_obj is None:
            debug_print(f"Object {obj_idx} not found in config")
            return None
        
        # Load mesh
        vertices, faces = self._load_mesh_for_object(config_obj)
        
        # Extract modal parameters
        try:
            modal_params = self.sound2modal.compute(
                audio_file=self.audio_files[obj_idx],
                obj_idx=obj_idx,
                vertices=vertices,
                faces=faces,
                output_name=config_obj.name
            )
            
            # Generate and save Faust .lib file
            lib_output = f"{self.dsp_path}/{config_obj.name}.lib"
            self.sound2modal.save_to_file(
                modal_params=modal_params,
                output_path=lib_output,
                output_name=config_obj.name
            )
            
            # Generate resonance model if needed
            if config_obj.resonance:
                self._generate_resonance_model(config_obj, modal_params)
            
            return lib_output
            
        except Exception as e:
            debug_print(f"Error processing object {config_obj.name}: {e}")
            return None

