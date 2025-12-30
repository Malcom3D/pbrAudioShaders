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

import os, sys
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from dask import delayed, compute

from ..core.impact_manager import ImpactManager
from ..core.mesh2modal import Mesh2Modal

@dataclass
class SynthWriter:
    impact_manager: ImpactManager

    def __post_init__(self):
        self.mesh2modal = Mesh2Modal(self.impact_manager)

    def compute(self):
        """
        Generate Faust DSP code for each impact event.
        
        For each impact:
        11. Ensure modal models exist for all colliding objects
        2. Generate coupled system Faust code
        3. Save the DSP file for rendering
        """
        config = self.impact_manager.get('config')
        output_path = config.system.output_path
        impacts = self.impact_manager.get('impacts')
        
        # Dictionary to track which objects already have modal models
        processed_objects = {}
        
        # Process each impact
        for impact_idx in range(len(impacts)):
            impact = impacts[impact_idx]
            print(f"Processing impact {impact_idx} at time {impact.time:.3f}s")
            
            # Step 1: Ensure modal models exist for all colliding objects
            for collision_idx in range(len(impact.collisions)):
                collision = impact.collisions[collision_idx]
                if collision.obj_idx not in processed_objects:
                    # Find the object configuration
                    config_obj = None
                    gen_lib = []
                    for config_obj in config.objects:
                        if config_obj.idx == collision.obj_idx:
                            # Check if modal model already exists
                            lib_name = f"{output_path}/dsp/{config_obj.name}.lib"
                            if not os.path.exists(lib_name):
                                gen_lib.append(config_obj.idx)
                        processed_objects[collision.obj_idx] = config_obj
                    tasks = [self.mesh2modal.compute(_obj_idx) for _obj_idx in gen_lib]
                    compute(*tasks)
            
            # Step 2: Generate Faust DSP code for this impact
            dsp_code = self._generate_coupled_system_dsp(impact)
            
            # Step 3: Save the DSP file
            dsp_filename = self._save_dsp_file(impact, dsp_code)
            
            # Store the DSP file path in the impact data for later rendering
            impact.dsp_file = dsp_filename
            print(f"  Generated DSP file: {dsp_filename}")

    def _generate_coupled_system_dsp(self, impact) -> str:
        """
        Generate Faust DSP code for a coupled system with contact audio excitation.
        """
        # Get all colliding objects in this impact
        colliding_objects = []
        for obj_contact in impact.object_contacts:
            colliding_objects.append(obj_contact.obj_idx)
        
        if len(colliding_objects) < 2:
            raise ValueError(f"Impact {impact.idx} has less than 2 colliding objects")
        
        # Generate imports
        imports = self._generate_imports(colliding_objects)
        
        # Generate contact audio excitation
        excitations = self._generate_contact_excitations(impact, colliding_objects)
        
        # Generate coupled modal systems
        coupled_systems = self._generate_coupled_systems(impact, colliding_objects)
        
        # Generate output mixing
        outputs = self._generate_outputs(colliding_objects)
        
        # Combine all parts
        dsp_code = f"""// Impact {impact.idx} - Coupled Modal Synthesis with Contact Audio
// Time: {impact.start_time:.3f}s - {impact.end_time:.3f}s
// Duration: {impact.duration:.3f}s
// Contact type: {impact.dominant_contact_type.value}
// Coordinates: {impact.coord}
// Objects: {colliding_objects}

{imports}

// ============================================
// Contact Audio Excitation
// ============================================

{excitations}

// ============================================
// Coupled Modal Systems
// ============================================

{coupled_systems}

// ============================================
// Output Mixing
// ============================================

{outputs}
"""
        return dsp_code
    
    def _generate_contact_excitations(self, impact, colliding_objects: List[int]) -> str:
        """Generate excitation using pre-rendered contact audio and acceleration noise"""
        config = self.impact_manager.get('config')
        excitations = []
        
        for obj_idx in colliding_objects:
            obj_contact = impact.get_object_contact(obj_idx)
            if not obj_contact:
                # Fallback to impulse excitation
                excitations.append(f"""
// Excitation for object {obj_idx} (fallback impulse)
excitation_{obj_idx} = 1.0 : ba.impulsify * 0.5;
""")
                continue
            
            # Get audio file paths
            contact_audio_file = getattr(obj_contact, 'audio_file', None)
            accel_noise_files = getattr(obj_contact, 'acceleration_noise_files', [])
            
            # Find object name for modal model
            obj_name = ""
            for config_obj in config.objects:
                if config_obj.idx == obj_idx:
                    obj_name = config_obj.name
                    break
            
            # Get vertex IDs for excitation
            vertex_ids = impact.get_all_vertices(obj_idx)
            if not vertex_ids:
                vertex_ids = [0]  # Default to first vertex
            
            # Create excitation combining contact audio and acceleration noise
            excitation_components = []
            
            # Add contact audio if available
            if contact_audio_file:
                rel_path = os.path.relpath(contact_audio_file, os.getcwd())
                excitation_components.append(f'si.buffer(so.waveform("{rel_path}"))')
            
            # Add acceleration noise if available
            for accel_file in accel_noise_files:
                rel_path = os.path.relpath(accel_file, os.getcwd())
                excitation_components.append(f'si.buffer(so.waveform("{rel_path}")) * 0.3')
            
            if excitation_components:
                # Combine all excitation sources
                if len(excitation_components) == 1:
                    excitation_signal = excitation_components[0]
                else:
                    excitation_signal = " + ".join(excitation_components)
                
                excitation_code = f"""
// Combined excitation for object {obj_idx} ({obj_name})
// Contact type: {obj_contact.get_contact_types()}
excitation_{obj_idx} = {excitation_signal};
"""
            else:
                # Fallback to impulse
                excitation_code = f"""
// Impulse excitation for object {obj_idx} ({obj_name})
excitation_{obj_idx} = 1.0 : ba.impulsify * 0.5;
"""
            
            excitations.append(excitation_code)
        
        return "\n".join(excitations)

    def _generate_coupled_systems(self, impact, colliding_objects: List[int]) -> str:
        """Generate coupled modal systems excited by contact audio"""
        config = self.impact_manager.get('config')
        systems = []
        
        for obj_idx in colliding_objects:
            obj_contact = impact.get_object_contact(obj_idx)
            if not obj_contact:
                continue
            
            # Find object configuration
            obj_config = None
            obj_name = ""
            for config_obj in config.objects:
                if config_obj.idx == obj_idx:
                    obj_config = config_obj

                    obj_name = config_obj.name
                    break
            
            if not obj_config:
                continue
            
            # Get vertex IDs for this object
            vertex_ids = impact.get_all_vertices(obj_idx)
            if not vertex_ids:
                vertex_ids = [0]
            
            # Create modal system with multiple excitation points
            # For multiple contact points, we need to sum contributions
            if len(vertex_ids) == 1:
                # Single excitation point
                system_code = f"""
// Modal system for object {obj_idx} ({obj_name})
// Excitation at vertex {vertex_ids[0]}
modal_system_{obj_idx} = excitation_{obj_idx} : {obj_name}({vertex_ids[0]}, 1.0);
"""
            else:
                # Multiple excitation points - sum contributions
                modal_terms = []
                for i, vertex_id in enumerate(vertex_ids):
                    modal_terms.append(f"excitation_{obj_idx} : {obj_name}({vertex_id}, 1.0)")
                
                sum_terms = " + ".join(modal_terms)
                gain = 1.0 / len(vertex_ids)  # Normalize
                
                system_code = f"""
// Modal system for object {obj_idx} ({obj_name})
// Multiple excitation points: {vertex_ids}
modal_system_{obj_idx} = ({sum_terms}) * {gain:.3f};
"""
            
            systems.append(system_code)
        
        return "\n".join(systems)

    def _generate_imports(self, colliding_objects: List[int]) -> str:
        """
        Generate import statements for all modal models.
        """
        config = self.impact_manager.get('config')
        output_path = config.system.output_path
        imports = [self._stdfaust_imports()]
        
        for obj_idx in colliding_objects:
            # Find object configuration
            config_obj = None
            for obj in config.objects:
                if obj.idx == obj_idx:
                    config_obj = obj
                    lib_path = f"{output_path}/dsp/{config_obj.name}.lib"
                    # Use relative path from current directory
                    rel_path = os.path.relpath(lib_path, os.getcwd())
                    imports.append(f'import("{rel_path}");')
        
        return "\n".join(imports)

    def _generate_hammer(self, impact, colliding_objects: List[int]) -> str:
        """
        Generate hammer model to get its characteristic impact sound.
        Process the hammer excitation through modalModel of first object to get the hammer's vibration signature
        """
        config = self.impact_manager.get('config')
        hammer = []

        obj_idx = colliding_objects[0]
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:

                # Get vertex IDs where this object was hit
                vertex_ids = self.impact_manager.get_expos(obj_idx)
                vertex_id = impact.get_expos(obj_idx)[0]
                expos = vertex_ids.index(vertex_id)

                # The modal model function expects: modalModel(expos, t60Scale)
                hammer_code = f"""
// We excite the hammer model to get its characteristic impact sound
hammerExcitation = 1.0 : ba.impulsify;

// // Process the hammer excitation through object {obj_idx} ({config_obj.name})
// This gives us the hammer's vibration signature
hammerVibration = hammerExcitation : {config_obj.name}({expos}, 0.1);
"""
                hammer.append(hammer_code)

        return "\n".join(hammer)

    def _generate_excitations(self, impact, colliding_objects: List[int]) -> str:
        """
        Generate excitation signals for each object based on impact forces.
        """
        forces_mag = self.impact_manager.get_force_mag()
        excitations = []

        for obj_idx in colliding_objects:
            # Get collisions for this object
            obj_collisions = []
            for collision_idx in range(len(impact.collisions)):
                collision = impact.collisions[collision_idx]
                obj_collisions.append(collision)
            
            # Calculate total force magnitude for this object
            total_force = 0.0
            for collision in obj_collisions:
                force_mag = np.linalg.norm(collision.collision.force_vector)
                total_force += force_mag
            
            # Normalize force to reasonable excitation level (0-1 range)
            # using order of magnitude
            normalized_force = total_force / forces_mag
            
            # Generate excitation signal
            # Using a short impulse with amplitude based on force
            excitation_code = f"""
// Excitation for object {obj_idx}
collision_force_{obj_idx} = {normalized_force:.6f};
excitation_{obj_idx} = hammerVibration * collision_force_{obj_idx};
"""
            excitations.append(excitation_code)
        
        return "\n".join(excitations)

    def _generate_outputs(self, colliding_objects: List[int]) -> str:
        """
        Generate output mixing for all modal systems.
        """
        # Sum all modal systems
        system_outputs = []
        for obj_idx in colliding_objects:
            system_outputs.append(f"modal_system_{obj_idx}")
        
        if not system_outputs:
            return "process = 0;"
        
        # Sum all outputs with optional gain adjustments
        sum_output = " + ".join(system_outputs)
        
        # Apply overall gain control and output
        output_code = f"""
// Mix all modal systems
mixed_output = ({sum_output}) * 0.5;

// Output to raw pcm wav float32 samples
process = mixed_output;
"""
        return output_code

    def _save_dsp_file(self, impact, dsp_code: str) -> str:
        """
        Save the generated DSP code to a file.
        
        Returns:
            Path to the saved DSP file
        """
        # Create output directory if it doesn't exist
        config = self.impact_manager.get('config')
        output_path = f"{config.system.output_path}/dsp"
        os.makedirs(output_path, exist_ok=True)
        
        # Generate filename based on impact index and time
        filename = f"impact_{impact.idx:04d}_t{impact.time:.3f}s.dsp"
        filepath = os.path.join(output_path, filename)
        
        # Save the DSP code
        with open(filepath, 'w') as f:
            f.write(dsp_code)
        
        return filepath

    def _stdfaust_imports(self) -> str:
        faustlib = f"{os.path.dirname(os.path.abspath(sys.modules[ImpactManager.__module__].__file__))}/../faustlib"
        stdfaust_lib = f"""
aa = library("{faustlib}/aanl.lib");
sf = library("{faustlib}/all.lib");
an = library("{faustlib}/analyzers.lib");
ba = library("{faustlib}/basics.lib");
co = library("{faustlib}/compressors.lib");
de = library("{faustlib}/delays.lib");
dm = library("{faustlib}/demos.lib");
dx = library("{faustlib}/dx7.lib");
en = library("{faustlib}/envelopes.lib");
fd = library("{faustlib}/fds.lib");
fi = library("{faustlib}/filters.lib");
ho = library("{faustlib}/hoa.lib");
it = library("{faustlib}/interpolators.lib");
ma = library("{faustlib}/maths.lib");
mi = library("{faustlib}/mi.lib");
ef = library("{faustlib}/misceffects.lib");
os = library("{faustlib}/oscillators.lib");
no = library("{faustlib}/noises.lib");
pf = library("{faustlib}/phaflangers.lib");
pl = library("{faustlib}/platform.lib");
pm = library("{faustlib}/physmodels.lib");
qu = library("{faustlib}/quantizers.lib");
rm = library("{faustlib}/reducemaps.lib");
re = library("{faustlib}/reverbs.lib");
ro = library("{faustlib}/routes.lib");
sp = library("{faustlib}/spats.lib");
si = library("{faustlib}/signals.lib");
so = library("{faustlib}/soundfiles.lib");
sy = library("{faustlib}/synths.lib");
ve = library("{faustlib}/vaeffects.lib");
vl = library("{faustlib}/version.lib");
wa = library("{faustlib}/webaudio.lib");
wd = library("{faustlib}/wdmodels.lib");
"""
        return stdfaust_lib
