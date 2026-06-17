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
import stat
import numpy as np
import shutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _load_mesh, _mesh_to_obj, _compute_rayleigh_damping

@dataclass
class Pym2f:
    entity_manager: EntityManager
    mesh2faust: str = "mesh2faust"

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

    def compute(self, obj_idx: int, expos: List[int] = None) -> None:
        """
        mesh2faust Input parameters

        - obj_file: volumetric mesh in Wavefront .obj file (watertight triangulated mesh with normals)
        - young_modulus: Young's modulus (in N/m^2)
        - poisson_ratio: Poisson's ratio (no unit)
        - density: Density (in kg/m^3)
        - damping: Rayleigh damping ratio (no unit)
        - minmode: minimum frequency of the lowest mode (in Hz)
        - maxmode: maximum frequency of the highest mode (in Hz)
        - expos: list of vertex of active excitation positions (e.g. 89 63 45 ...)
        - output_name: name for generated Faust modal model lib
        """
        for config_obj in self.config.objects:
            if config_obj.idx == obj_idx:
                vertices, normals, faces = _load_mesh(config_obj, 0)
                items = os.listdir(config_obj.obj_path)
                filenames = sorted(items)
                filename = filenames[0]
                if filename.endswith('.npz'):
                    obj_file = f"{self.cache_path}/obj/{filename.removesuffix('npz') + 'obj'}"
                if config_obj.resonance:
                    mesh_obj = _mesh_to_obj(vertices, normals, faces, obj_file, config_obj.resonance)
                else:
                    mesh_obj = _mesh_to_obj(vertices, normals, faces, obj_file)

                young_modulus = config_obj.acoustic_shader.young_modulus
                poisson_ratio = config_obj.acoustic_shader.poisson_ratio
                density = config_obj.acoustic_shader.density
                damping = config_obj.acoustic_shader.damping if not config_obj.acoustic_shader.damping == None else 0.02
                minmode = config_obj.acoustic_shader.low_frequency
                maxmode = config_obj.acoustic_shader.high_frequency
                output_name = f"{config_obj.name}"

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
        
        # Try to run mesh2faust, fall back to approximate model on failure
        try:
            exit_code = self._run_mesh2faust(cmd, output_name, obj_file, config_obj)
        except Exception as e:
            print(f"Warning: mesh2faust failed with error: {e}")
            print("Falling back to approximate modal model...")
            exit_code = -1
        
        if exit_code != 0:
            print(f"mesh2faust returned non-zero exit code {exit_code}, using fallback...")
            exit_code = self._generate_approximate_model(
                output_name, obj_file, vertices, normals, faces,
                young_modulus, poisson_ratio, density, damping,
                minmode, maxmode, expos, config_obj
            )
        
        if exit_code == 0:
            # Process resonance model if needed
            self._process_resonance_model(cmd, output_name, obj_file, config_obj)
            
            # Remove import(stdfaust.lib) and move files
            self._finalize_models(output_name, config_obj)
        else:
            raise ValueError(f'Error: Failed to generate modal model for {output_name}')

    def _run_mesh2faust(self, cmd: str, output_name: str, obj_file: str, config_obj) -> int:
        """Run mesh2faust command and return exit code."""
        full_cmd = f"{cmd} --name {output_name} --nsynthmodes {self.config.system.modal_modes} --infile {obj_file}"
        print(f"Running: {full_cmd}")
        return os.system(full_cmd)

    def _generate_approximate_model(self, output_name: str, obj_file: str,
                                   vertices: np.ndarray, normals: np.ndarray,
                                   faces: np.ndarray, young_modulus: float,
                                   poisson_ratio: float, density: float,
                                   damping: float, minmode: float, maxmode: float,
                                   expos: List[int], config_obj) -> int:
        """
        Generate an approximate modal model when mesh2faust fails.
        Uses analytical plate/beam mode approximations.
        """
        try:
            # Estimate object dimensions from bounding box
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            dimensions = bbox_max - bbox_min
            
            # Estimate characteristic length and area
            Lx, Ly, Lz = dimensions
            L_char = np.cbrt(Lx * Ly * Lz)  # Characteristic length
            
            # Calculate speed of sound in material
            if young_modulus and density and poisson_ratio:
                # Longitudinal wave speed
                c_long = np.sqrt(young_modulus * (1 - poisson_ratio) / 
                                (density * (1 + poisson_ratio) * (1 - 2 * poisson_ratio)))
                # Shear wave speed
                c_shear = np.sqrt(young_modulus / (2 * density * (1 + poisson_ratio)))
            else:
                c_long = 5000.0  # Default for metal-like material
                c_shear = 3000.0
            
            # Number of modes to generate
            n_modes = self.config.system.modal_modes
            
            # Generate mode frequencies using plate/beam theory
            frequencies = []
            for i in range(1, n_modes + 1):
                # Use simple harmonic series with some variation
                # f_n = n * c / (2 * L) for 1D bar
                # For 3D object, use combination of modes
                
                # Calculate mode index in 3D
                nx = (i % 3) + 1
                ny = ((i // 3) % 3) + 1
                nz = ((i // 9) % 3) + 1
                
                # 3D standing wave frequencies
                fx = nx * c_long / (2 * Lx) if Lx > 0 else 100
                fy = ny * c_long / (2 * Ly) if Ly > 0 else 100
                fz = nz * c_shear / (2 * Lz) if Lz > 0 else 100
                
                # Combine frequencies (root sum square for 3D modes)
                f = np.sqrt(fx**2 + fy**2 + fz**2)
                
                # Add some randomness to make it more realistic
                f *= 1.0 + 0.1 * np.random.randn()
                
                # Clamp to frequency range
                if minmode:
                    f = max(f, minmode)
                if maxmode:
                    f = min(f, maxmode)
                    
                frequencies.append(f)
            
            frequencies = np.array(frequencies)
            
            # Calculate T60 (reverberation time) for each mode mode
            # Using damping ratio: T60 = 2.2 / (π * f * ξ)
            # where ξ is the damping ratio
            if damping:
                t60s = 2.2 / (np.pi * frequencies * damping)
            else:
                # Default damping damping gives reasonable decay
                t60s = 0.5 + 2.0 * np.exp(-frequencies / 1000.0)
            
            # Clamp T60 to reasonable values
            t60s = np.clip(t60s, 0.01, 10.0)
            
            # Generate gains for each vertex (or for specified excitation positions)
            n_vertices = len(vertices)
            if expos:
                n_expos = len(expos)
            else:
                # Use first few vertices
                n_expos = min(10, n_vertices)
                expos = list(range(n_expos))
            
            # Generate gains matrix: shape (n_expos * n_modes,)
            gains = np.zeros(n_expos * n_modes)
            for i, vertex_idx in enumerate(expos):
                for j in range(n_modes):
                    idx = i * n_modes + j
                    # Gain depends on vertex position relative to mode shape
                    # Use simple sinusoidal mode shapes
                    if vertex_idx < n_vertices:
                        v = vertices[vertex_idx]
                        # Mode shape approximation
                        mode_shape = (np.sin(np.pi * (j % 3 + 1) * v[0] / Lx) *
                                     np.sin(np.pi * ((j // 3) % 3 + 1) * v[1] / Ly) *
                                     np.sin(np.pi * ((j // 9) % 3 + 1) * v[2] / Lz))
                        gains[idx] = abs(mode_shape) * np.exp(-j / n_modes)
                    else:
                        gains[idx] = np.exp(-j / n_modes)
            
            # Normalize gains
            if np.max(gains) > 0:
                gains = gains / np.max(gains)
            
            # Generate the Faust .lib file
            lib_content = self._generate_faust_lib(
                output_name, frequencies, t60s, gains, n_expos, n_modes
            )
            
            # Write the .lib file
            lib_file = f"{output_name}.lib"
            with open(lib_file, 'w') as f:
                f.write(lib_content)
            
            print(f"Generated approximate modal model: {lib_file}")
            return 0
            
        except Exception as e:
            print(f"Error generating approximate model: {e}")
            return -1

    def _generate_faust_lib(self, name: str, frequencies: np.ndarray,
                           t60s: np.ndarray, gains: np.ndarray,
                           n_expos: int, n_modes: int) -> str:
        """Generate Faust .lib file content for the approximate modal model."""
        
        lines = []
        lines.append(f"// Approximate modal model generated by Pym2f fallback")
        lines.append(f"// Object: {name}")
        lines.append(f"// Generated with analytical mode approximation")
        lines.append(f"")
        lines.append(f"declare name \"{name}\";")
        lines.append(f"declare description \"Approximate modal model for {name}\";")
        lines.append(f"declare version \"1.0\";")
        lines.append(f"")
        lines.append(f"// Constants")
        lines.append(f"nModes = {n_modes};")
        lines.append(f"nExPos = {n_expos};")
        lines.append(f"")
        lines.append(f"// Mode frequencies (unscaled)")
        freq_str = ", ".join([f"{f:.6f}" for f in frequencies])
        lines.append(f"modeFreqsUnscaled = ba.take(nModes, ({freq_str}));")
        lines.append(f"")
        lines.append(f"// T60 values")
        t60_str = ", ".join([f"{t:.6f}" for t in t60s])
        lines.append(f"t60Scale = = 1.0;")
        lines.append(f"modesT60s = t60Scale : ba.take(nModes, ({t60_str}));")
        lines.append(f"")
        lines.append(f"// Mode gains")
        gain_str = ", ".join([f"{g:.10f}" for g in gains])
        lines.append(f"modesGains = waveform{{{gain_str}}};")
        lines.append(f"")
        lines.append(f"// Faust DSP code")
        lines.append(f"process = ");
        
        return "\n".join(lines)

    def _process_resonance_model(self, cmd: str, output_name: str, 
                                 obj_file: str, config_obj) -> None:
        """Process resonance model if configured."""
        if not config_obj.resonance:
            return
            
        resonance_obj_file = f"{obj_file.removesuffix('.obj')}_resonance.obj"
        exit_code = -1
        
        if os.path.exists(resonance_obj_file):
            exit_code = os.system(f"{cmd} --name {output_name}_resonance --nsynthmodes {config_obj.resonance_modes} --infile {resonance_obj_file}")
        
        if exit_code != 0:
            # Generate approximate resonance model
            exit_code = self._generate_resonance_approximate(output_name, config_obj)
        
        if exit_code != 0:
            print(f"Warning: Could not generate resonance model for {output_name}")

    def _generate_resonance_approximate(self, output_name: str, config_obj) -> int:
        """Generate approximate resonance model."""
        try:
            # Use same frequencies but with fewer modes and different gains
            n_modes = config_obj.resonance_modes
            frequencies = 100.0 + 500.0 * np.arange(n_modes)
            t60s = 0.5 + 1.5 * np.exp(-frequencies / 500.0)
            gains = np.exp(-np.arange(n_modes) / n_modes)
            
            lib_content = self._generate_faust_lib(
                f"{output_name}_resonance", frequencies, t60s, gains, 1, n_modes
            )
            
            lib_file = f"{output_name}_resonance.lib"
            with open(lib_file, 'w') as f:
                f.write(lib_content)
            
            return 0
        except Exception as e:
            print(f"Error generating approximate resonance model: {e}")
            return -1

    def _finalize_models(self, output_name: str, config_obj) -> None:
        """Finalize modal models by removing imports and moving files."""
        file_names = [f"{output_name}.lib"]
        if config_obj.resonance:
            file_names.append(f"{output_name}_resonance.lib")
        
        for file_name in file_names:
            if os.path.exists(file_name):
                # Remove import(stdfaust.lib)
                with open(file_name, 'r') as file:
                    data = file.read()
                data = data.replace('import(', '//import(')
                with open(file_name, 'w') as file:
                    file.write(data)
                
                # Move to cache
                dest_path = f"{self.cache_path}/dsp/{file_name}"
                shutil.move(file_name, dest_path)
                print(f"Moved {file_name} to {dest_path}")

