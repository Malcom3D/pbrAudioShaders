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
import re
import stat
import numpy as np
import shutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _mesh_to_obj, _compute_rayleigh_damping

from ..lib.primitive_geometry import PrimitiveGeometry
from ..lib.shape_properties import ShapeType, ShapeProperties
from ..lib.approx2faust import Approx2Faust, ModalParameters

@dataclass
class Pym2f:
    entity_manager: EntityManager
    mesh2faust: str = "mesh2faust"
    
    # Fallback parameters
    use_fallback: bool = True
    fallback_min_confidence: float = 0.3  # Minimum confidence to use fallback
    max_fallback_attempts: int = 2

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        bin_dir = f"{os.path.dirname(os.path.abspath(sys.modules[Pym2f.__module__].__file__))}/../bin"
        os.environ['LD_LIBRARY_PATH'] = bin_dir
        self.mesh2faust = f"{bin_dir}/{self.mesh2faust}"
        if not os.access(self.mesh2faust, os.X_OK):
            os.chmod(self.mesh2faust, stat.S_IXUSR)
        self.cache_path = f"{self.config.system.cache_path}"
        os.makedirs(f"{self.cache_path}/obj", exist_ok=True)
        os.makedirs(f"{self.cache_path}/dsp", exist_ok=True)
        
        # Initialize fallback components
        self.primitive_geometry = PrimitiveGeometry()
        self.approx2faust = Approx2Faust()

    def compute(self, obj_idx: int, expos: List[int] = None) -> None:
        """
        Compute modal model with fallback mechanism.
        
        Tries mesh2faust first, then falls back to approximate modal model
        if the result is invalid or fails.
        """
        # Extract object configuration
        config_obj = None
        vertices = None
        normals = None
        faces = None
        
        for config_obj_item in self.config.objects:
            if config_obj_item.idx == obj_idx:
                config_obj = config_obj_item
                vertices, normals, faces = _load_mesh(config_obj, 0)
                items = os.listdir(config_obj.obj_path)
                filenames = sorted(items)
                filename = filenames[0]
                if filename.endswith('.npz'):
                    obj_file = f"{self.cache_path}/obj/{filename.removesuffix('npz') + 'obj'}"
                    mesh_obj = _mesh_to_obj(vertices, normals, faces, obj_file, config_obj.resonance)

                young_modulus = config_obj.acoustic_shader.young_modulus
                poisson_ratio = config_obj.acoustic_shader.poisson_ratio
                density = config_obj.acoustic_shader.density
                damping = config_obj.acoustic_shader.damping if not config_obj.acoustic_shader.damping == None else 0.02
                minmode = config_obj.acoustic_shader.low_frequency
                maxmode = config_obj.acoustic_shader.high_frequency
                output_name = f"{config_obj.name}"

        # Try mesh2faust first
        success = False
        file_names = []
        
        for attempt in range(self.max_fallback_attempts + 1):
            if attempt == 0:
                # First attempt: use mesh2faust
                success, file_names = self._try_mesh2faust(config_obj, vertices, normals, faces, obj_file, young_modulus, poisson_ratio, density, damping, minmode, maxmode, expos, output_name)
            if attempt == 1 and not config_obj.static:
                # Second attempt: use mesh2faust with obj from random frame
                rand_frame = np.random.randint(1,int(re.findall(r'\d+', filenames[-2])[-1]))
                rand_vertices, rand_normals, rand_faces = _load_mesh(config_obj, rand_frame)
                mesh_obj = _mesh_to_obj(rand_vertices, rand_normals, rand_faces, obj_file, config_obj.resonance)
                success, file_names = self._try_mesh2faust(config_obj, rand_vertices, rand_normals, rand_faces, obj_file, young_modulus, poisson_ratio, density, damping, minmode, maxmode, expos, output_name)
            else:
                # Fallback: use approximate model
                if self.use_fallback:
                    print(f"Pym2f: Attempting fallback for {config_obj.name} (attempt {attempt})")
                    success, file_names = self._try_fallback(config_obj, vertices, faces, obj_file, young_modulus, poisson_ratio, density, damping, minmode, maxmode, expos, output_name)
            
            if success:
                break
        
        if not success:
            raise RuntimeError(f"Pym2f: Failed to generate modal model for {config_obj.name} "
                              f"after {self.max_fallback_attempts + 1} attempts")
        
        # Post-process generated files
        self._post_process_files(file_names, output_name, config_obj)

    def _try_mesh2faust(self, config_obj, vertices, normals, faces, obj_file, young_modulus, poisson_ratio, density, damping, minmode, maxmode, expos, output_name) -> Tuple[bool, List[str]]:
        """
        Try to generate modal model using mesh2faust.
        
        Returns:
            (success, file_names)
        """
        cmd = f"{self.mesh2faust} "
        
        if not young_modulus == None and not poisson_ratio == None and not density == None:
            alpha_rayleigh, beta_rayleigh = _compute_rayleigh_damping(minmode, maxmode, damping)
            cmd += f"--material {young_modulus} {poisson_ratio} {density} {alpha_rayleigh} {beta_rayleigh} "
        
        if not minmode == None:
            cmd += f"--minmode {minmode} "
        if not maxmode == None:
            cmd += f"--maxmode {maxmode} "
        if not expos == None:
            verts = ''
            for pos in expos:
                verts += f"{pos} "
            cmd += f"--expos {verts} "

        cmd += f"--showfreqs"
        exit_code = os.system(f"{cmd} --name {output_name} --nsynthmodes {self.config.system.modal_modes} --infile {obj_file}")
        
        file_names = []
        if exit_code == 0:
            file_names.append(f"{output_name}.lib")
            
            # Validate the generated lib file
            if self._validate_lib_file(f"{output_name}.lib"):
                # Handle resonance
                if config_obj.resonance:
                    resonance_obj_file = f"{obj_file.removesuffix('.obj')}_resonance.obj"
                    if os.path.exists(resonance_obj_file):
                        exit_code = os.system(f"{cmd} --name {output_name}_resonance --nsynthmodes {config_obj.resonance_modes} --infile {resonance_obj_file}")
                    if not exit_code == 0 or not os.path.exists(resonance_obj_file):
                        exit_code = os.system(f"{cmd} --name {output_name}_resonance --nsynthmodes {config_obj.resonance_modes} --infile {obj_file}")
                    if exit_code == 0:
                        file_names.append(f"{output_name}_resonance.lib")
                    else:
                        print(f"Pym2f: Warning - - resonance model generation failed for {outputoutput_name}")
                
                return True, file_names
        
        # Clean up failed output
        for fn in [f"{output_name}.lib", f"{output_name}_resonance.lib"]:
            if os.path.exists(fn):
                os.remove(fn)
        
        return False, []

    def _try_fallback(self, config_obj, vertices, faces, obj_file, young_modulus, poisson_ratio, density, damping, minmode, maxmode, expos, output_name) -> Tuple[bool, List[str]]:
        """
        Generate approximate modal model using primitive geometry approximation.
        
        Returns:
            (success, file_names)
        """
        # Classify shape
        shape_props = self.primitive_geometry.classify(vertices, faces)
        
        print(f"Pym2f fallback: Classified {config_obj.name} as {shape_props.shape_type.value} "
              f"(confidence: {shape_props.confidence:.2f})")
        
        # Check if classification is good enough
        if shape_props.confidence < self.fallback_min_confidence:
            print(f"Pym2f fallback: Low confidence ({shape_props.confidence:.2f}), "
                  f"using irregular shape approximation")
        
        # Compute modal parameters
        n_modes = self.config.system.modal_modes
        
        modal_params = self.approx2faust.compute(
            vertices=vertices,
            faces=faces,
            young_modulus=young_modulus if young_modulus else 1e9,
            poisson_ratio=poisson_ratio if poisson_ratio else 0.3,
            density=density if density else 1000.0,
            damping=damping if damping else 0.02,
            min_freq=minmode if minmode else 20.0,
            max_freq=maxmode if maxmode else 20000.0,
            n_modes=n_modes
        )
        
        # Generate Faust .lib file
        lib_content = self.approx2faust.to_faust_lib(
            modal_params=modal_params,
            output_name=output_name,
            min_freq=minmode if minmode else 20.0,
            max_freq=maxmode if maxmode else 20000.0
        )
        
        # Save the .lib file
        lib_file = f"{output_name}.lib"
        with open(lib_file, 'w') as f:
            f.write(lib_content)
        
        file_names = [lib_file]
        
        # Validate the generated file
        if not self._validate_lib_file(lib_file):
            print(f"Pym2f fallback: Generated lib file validation failed for {output_name}")
            os.remove(lib_file)
            return False, []
        
        # Handle resonance if requested
        if config_obj.resonance:
            resonance_params = self.approx2faust.compute(
                vertices=vertices,
                faces=faces,
                young_modulus=young_modulus if young_modulus else 1e9,
                poisson_ratio=poisson_ratio if poisson_ratio else 0.3,
                density=density if density else 1000.0,
                damping=damping if damping else 0.02,
                min_freq=minmode if minmode else 20.0,
                max_freq=maxmode if maxmode else 20000.0,
                n_modes=config_obj.resonance_modes
            )
            
            resonance_content = self.approx2faust.to_faust_lib(
                modal_params=resonance_params,
                output_name=f"{output_name}_resonance",
                min_freq=minmode if minmode else 5.0,
                max_freq=maxmode if maxmode else 24000.0
            )
            
            resonance_file = f"{output_name}_resonance.lib"
            with open(resonance_file, 'w') as f:
                f.write(resonance_content)
            
            file_names.append(resonance_file)
        
        print(f"Pym2f fallback: Successfully generated approximate modal model for {config_obj.name}")
        return True, file_names

    def _validate_lib_file(self, lib_file: str) -> bool:
        """
        Validate that a generated .lib file contains valid modal data.
        
        Checks:
        - File exists and is non-empty
        - Contains modeFreqsUnscaled with valid values
        - Contains modesGains waveform with valid values
        - No NaN or Inf values
        - Reasonable frequency range
        """
        if not os.path.exists(lib_file):
            print(f"Pym2f validation: File {lib_file} does not exist")
            return False
        
        file_size = os.path.getsize(lib_file)
        if file_size == 0:
            print(f"Pym2f validation: File {lib_file} is empty")
            return False
        
        try:
            with open(lib_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Pym2f validation: Cannot read {lib_file}: {e}")
            return False
        
        # Check for essential content
        if 'modeFreqsUnscaled' not in content:
            print(f"Pym2f validation: {lib_file} missing modeFreqsUnscaled")
            return False
        
        if 'modesT60s' not in content:
            print(f"Pym2f validation: {lib_file} missing modesT60s")
            return False
        
        if 'modesGains' not in content:
            print(f"Pym2f validation: {lib_file} missing modesGains")
            return False
        
        # Check for NaN or Inf in the content
        if 'nan' in content.lower() or '-nan' in content.lower() or 'inf' in content.lower():
            print(f"Pym2f validation: {lib_file} contains NaN or Inf values")
            return False
        
        # Extract and validate frequencies

        freq_pattern = r'modeFreqsUnscaled.*?=.*?ba\.take.*?$$(.*?)$$'
        t60_pattern = r'modesT60s.*?=.*?t60Scale.*?ba\.take.*?$$(.*?)$$'
        gain_pattern = r'modesGains.*?=.*?waveform\{(.*?)\}'
        parentesis_match = r'\(([^()]+)\)'
        tuple_match = r'\d+\.?\d+'

        freq_validated, gains_validated = (False for _ in range(2))
        with open(lib_file, 'r') as file:
            lines = file.readlines()
            frequencies, t60s, gains = ([] for _ in range(3))
            for line in lines:
                line = line.replace('-nan','1')
                # Extract frequencies from modeFreqsUnscaled
                freq_match = re.search(freq_pattern, line, re.DOTALL)
                if not freq_match == None:
                    freq_tuple_match = re.findall(tuple_match, freq_match.group())
                    if not len(freq_tuple_match) == 0:
                        # Check for reasonable frequency range
                        freqs = [float(f) for f in freq_tuple_match]
                        if max(freqs) < 1.0 or min(freqs) < 0:
                            print(f"Pym2f validation: {lib_file} has unreasonable frequencies")
                            freq_validated = False
                        else:
                            freq_validated = True

            for line in lines:
                # Extract gains - this is complex due to the large waveform
                gain_match = re.search(gain_pattern, line, re.DOTALL)
                if not gain_match == None:
                    gain_tuple_match = re.findall(gain_pattern, gain_match.group())
                    gain_tuple_match = re.sub("'", "", gain_tuple_match[0])
                    if not gain_tuple_match == None:
                        try:
                            gains = [float(f) for f in gain_tuple_match.split(",")]
                            gains_validated = True
                        except:
                            print(f"Pym2f validation: {lib_file} has unreasonable gains")
                            gains_validated = False
        
        return (freq_validated and gains_validated)

    def _post_process_files(self, file_names: List[str], output_name: str, config_obj) -> None:
        """
        Post-process generated .lib files.
        
        - quality filtering and fixing mode frequencies and T60 value
        - Remove import(stdfaust.lib) statements
        - Move files to cache directory
        """
        for file_name in file_names:
            if not os.path.exists(file_name):
                print(f"Pym2f: Warning - {file_name} not found for post-processing")
                continue
            
            with open(file_name, 'r') as file:
                data = file.read()
            
            # Remove import statements to avoid dependency issues
            data = data.replace('import(', '//import(')
            
            with open(file_name, 'w') as file:
                file.write(data)
            
            # Move to cache directory
            dest_path = f"{self.cache_path}/dsp/{file_name}"
            shutil.move(file_name, dest_path)
            print(f"Pym2f: Saved {file_name} to {dest_path}")

