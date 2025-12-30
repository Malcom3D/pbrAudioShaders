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

from dataclasses import dataclass
import os
import numpy as np
import trimesh

from ..core.optimize_obj import OptimizeObj
from ..core.contact_analyzer import ContactAnalyzer
from ..core.contact_audio_generator import ContactAudioGenerator
from ..core.acceleration_noise import AccelerationNoiseGenerator
from ..core.fast_acceleration_noise import FastAccelerationNoiseGenerator
from ..core.impact_manager import ImpactManager
from ..core.synth_writer import SynthWriter
from ..core.modal_synth import ModalSynth

from ..lib.trajectory_loader import load_trajectories_from_obj_files, interpolate_trajectories


@dataclass
class ImpactEngine:
    impact_manager: ImpactManager

    def __post_init__(self):
        print('ImpactEngine.post_init')
        self.optimize_obj = OptimizeObj(self.impact_manager)
        self.contact_analyzer = ContactAnalyzer(self.impact_manager)
        self.contact_audio_generator = ContactAudioGenerator()
        self.acceleration_noise_generator = AccelerationNoiseGenerator()
        self.fast_acceleration_noise_generator = FastAccelerationNoiseGenerator()
        self.synth_writer = SynthWriter(self.impact_manager)
        self.modal_synth = ModalSynth(self.impact_manager)
        
        # Store trajectories and meshes for acceleration noise
        self.trajectories = {}
        self.meshes = {}

    def compute(self):
        print('ImpactEngine.compute')
        config = self.impact_manager.get('config')
        
        # Step 1: Optimize object meshes
        for obj in config.objects:
            self.optimize_obj.compute(obj.idx)

        # Step 2: Load trajectories and meshes for acceleration noise
        self._load_trajectories_and_meshes()
        
        # Step 3: Analyze contacts using physically-based models
        self.contact_analyzer.compute()
        
        # Step 4: Generate contact audio for each object in each event
        self._generate_contact_audio()
        
        # Step 5: Generate acceleration noise
        self._generate_acceleration_noise()
        
        # Step 6: Generate Faust synthesizers
        self.synth_writer.compute()
        
        # Step 7: Render audio
        self.modal_synth.compute()
    
    def _load_trajectories_and_meshes(self):
        """Load trajectories and meshes for acceleration noise generation"""
        config = self.impact_manager.get('config')
        
        print("Loading trajectories and meshes for acceleration noise...")
        
        # Load meshes
        self.meshes = {}
        for config_obj in config.objects:
            optimized_obj_path = os.path.join(config_obj.obj_path, f"optimized_{config_obj.name}.obj")
            if os.path.exists(optimized_obj_path):
                try:
                    mesh = trimesh.load(optimized_obj_path, force='mesh')
                    self.meshes[config_obj.idx] = {
                        'mesh': mesh,
                        'volume': mesh.volume,
                        'bounds': mesh.bounds,
                        'surface_area': mesh.area,
                        'center_mass': mesh.center_mass
                    }
                    print(f"  Loaded mesh for object {config_obj.idx}: {len(mesh.vertices)} vertices")
                except Exception as e:
                    print(f"  Warning: Could not load mesh for object {config_obj.idx}: {e}")
        
        # Try to get trajectories from ContactAnalyzer first
        if hasattr(self.contact_analyzer, 'interpolated_trajectories'):
            self.trajectories = self.contact_analyzer.interpolated_trajectories
            print(f"  Using trajectories from ContactAnalyzer: {len(self.trajectories)} objects")
        else:
            # Fall back to loading trajectories directly
            self.trajectories = {}
            for config_obj in config.objects:
                if config_obj.idx in self.meshes:
                    try:
                        # Load raw trajectory data
                        raw_trajectory = load_trajectories_from_obj_files(
                            config_obj.obj_path,
                            config_obj.name,
                            config.system.fps
                        )
                        
                        if raw_trajectory:
                            # Interpolate to audio sample rate
                            interpolated_trajectory = interpolate_trajectories(
                                raw_trajectory,
                                config.system.sample_rate
                            )
                            
                            self.trajectories[config_obj.idx] = interpolated_trajectory

                            print(f"  Loaded trajectory for object {config_obj.idx}: ")
                            print(f"{interpolated_trajectory['num_samples']} samples at {config.system.sample_rate}Hz")
                        else:
                            print(f"  Warning: No trajectory data for object {config_obj.idx}")
                            
                    except Exception as e:
                        print(f"  Warning: Could not load trajectory for object {config_obj.idx}: {e}")

    def _generate_contact_audio(self):
        """Generate audio for all contact events"""
        config = self.impact_manager.get('config')
        impacts = self.impact_manager.get('impacts')
        
        # Configure audio generator
        self.contact_audio_generator.sample_rate = config.system.sample_rate
        
        # Create output directory for contact audio
        output_dir = f"{config.system.output_path}/contact_audio"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate audio for each event and object
        for event_idx, event in impacts.items():
            # Update event audio parameters
            event.audio_duration = 3.0
            event.sample_rate = config.system.sample_rate
            
            # Generate audio for each object in the event
            for obj_contact in event.object_contacts:
                audio = self.contact_audio_generator.generate_contact_audio(
                    event, obj_contact.obj_idx
                )
                
                # Save audio file
                audio_file = self.contact_audio_generator.save_contact_audio(
                    audio, event, obj_contact.obj_idx, output_dir
                )
                
                # Store audio file path in object contact for later use
                obj_contact.audio_file = audio_file
                
                print(f"Generated {event.dominant_contact_type.value} audio for "
                      f"object {obj_contact.obj_idx} in event {event.idx}: "
                      f"{len(audio)/config.system.sample_rate:.3f}s")
    
    def _generate_acceleration_noise(self):
        """Generate acceleration noise for all objects in all events"""
        config = self.impact_manager.get('config')
        impacts = self.impact_manager.get('impacts')
        
        if not self.trajectories:
            print("Warning: No trajectory data available for acceleration noise generation")
            return
        
        # Configure acceleration noise generators
        self.acceleration_noise_generator.sample_rate = config.system.sample_rate
        self.fast_acceleration_noise_generator.sample_rate = config.system.sample_rate
        
        # Create output directory for acceleration noise
        output_dir = f"{config.system.output_path}/acceleration_noise"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating acceleration noise...")
        
        # Generate acceleration noise for each event and object
        for event_idx, event in impacts.items():
            print(f"  Processing event {event.idx}...")
            
            for obj_contact in event.object_contacts:
                obj_idx = obj_contact.obj_idx
                
                # Skip if no trajectory data for this object
                if obj_idx not in self.trajectories:
                    print(f"    Warning: No trajectory data for object {obj_idx}")
                    continue
                
                # Check if object is debris
                is_debris = False
                if obj_idx in self.meshes:
                    # Get object mass from configuration
                    mass = 1.0  # Default
                    for config_obj in config.objects:
                        if config_obj.idx == obj_idx and config_obj.density:
                            if 'mesh' in self.meshes[obj_idx]:
                                volume = self.meshes[obj_idx]['mesh'].volume
                                mass = config_obj.density * volume
                    
                    is_debris = self.fast_acceleration_noise_generator.is_debris(
                        self.meshes[obj_idx], mass
                    )
                
                # Generate appropriate acceleration noise
                try:
                    if is_debris:
                        print(f"    Generating fast acceleration noise for debris object {obj_idx}")
                        audio = self.fast_acceleration_noise_generator.generate.generate_fast_acceleration_noise(
                            event, obj_idx, self.trajectories, self.meshes, is_debris=True
                        )
                    else:
                        print(f"    Generating acceleration noise for object {obj_idx}")
                        audio = self.acceleration_noise_generator.generate_acceleration_noise(
                            event, obj_idx, self.trajectories, self.meshes
                        )
                    
                    # Save acceleration noise audio
                    import soundfile as sf
                    filename = f"accel_noise_event{event.idx:04d}_obj{obj_idx}.wav"
                    filepath = os.path.join(output_dir, filename)
                    sf.write(filepath, audio, config.system.sample_rate, subtype='FLOAT')
                    
                    # Store acceleration noise file path
                    if not hasattr(obj_contact, 'acceleration_noise_files'):
                        obj_contact.acceleration_noise_files = []
                    obj_contact.acceleration_noise_files.append(filepath)
                    
                    print(f"    Generated acceleration noise for object {obj_idx}: {len(audio)/config.system.sample_rate:.3f}s")
                    
                except Exception as e:
                    print(f"    Error generating acceleration noise for object {obj_idx}: {e}")

