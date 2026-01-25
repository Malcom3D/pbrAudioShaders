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
import shutil
from typing import List, Tuple
from dataclasses import dataclass

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_mesh, _mesh_to_obj

@dataclass
class Pym2f:
    entity_manager: EntityManager
    mesh2faust: str = "mesh2faust"

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        bin_dir = f"{os.path.dirname(os.path.abspath(sys.modules[Pym2f.__module__].__file__))}/../bin"
        os.environ['LD_LIBRARY_PATH'] = bin_dir
        self.mesh2faust = f"{bin_dir}/{self.mesh2faust}"
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
                fixed_obj_path = f"{config_obj.obj_path}/fixed_obj"
                if os.path.isdir(fixed_obj_path):
                    items = os.listdir(fixed_obj_path)
                    filenames = sorted(items)
                    filename = filenames[0]
                    if filename.endswith('.obj'):
                        obj_file = f"{fixed_obj_path}/{filename}"
                else: 
                    vertices, normals, faces = _load_mesh(config_obj, 0)
                    items = os.listdir(config_obj.obj_path)
                    filenames = sorted(items)
                    filename = filenames[0]
                    if filename.endswith('.npz'):
                        obj_file = f"{self.cache_path}/obj/{filename.removesuffix('npz') + 'obj'}"
                    mesh_obj = _mesh_to_obj(vertices, normals, faces, obj_file)

                young_modulus = config_obj.acoustic_shader.young_modulus
                poisson_ratio = config_obj.acoustic_shader.poisson_ratio
                density = config_obj.acoustic_shader.density
                damping = config_obj.acoustic_shader.damping if not config_obj.acoustic_shader.damping == None else 0.02 # conservative value
                minmode = config_obj.acoustic_shader.low_frequency
                maxmode = config_obj.acoustic_shader.high_frequency
                output_name = f"{config_obj.name}"

        cmd = f"{self.mesh2faust} "
        if not young_modulus == None and not poisson_ratio == None and not density == None:
            """
            compute alpha and beta Rayleigh damping coefficient from low and high frequency:
            alpha: stiffness-proportional damping coefficient (in seconds)
            beta: mass-proportional damping coefficient (in 1/seconds)
            """
            alpha_rayleigh, beta_rayleigh = self._compute_rayleigh_damping(minmode, maxmode, damping)
            cmd += f"--material {young_modulus} {poisson_ratio} {density} {alpha_rayleigh} {beta_rayleigh} "
        cmd += f"--infile {obj_file} "
        if not minmode == None:
            cmd += f"--minmode {minmode} "
        if not maxmode == None:
            cmd += f"--maxmode {maxmode} "
        if not expos == None:
            verts = ''
            for pos in expos:
                verts += f"{pos} "
            cmd += f"--expos {verts} "
        if not output_name == None:
            cmd += f"--name {output_name} "
        if not config_obj.connected == False:
            cmd += f"--freqcontrol "

        cmd += f"--showfreqs"
        exit_code = os.system(cmd)
        if not exit_code == 0:
            raise ValueError(f'Error: {cmd}')
        file_name = f"{output_name}.lib"

        # remove import(stdfaust.lib)
        with open(file_name, 'r') as file:
            data = file.read()
        data = data.replace('import(', '//import(')
        with open(file_name, 'w') as file:
            file.write(data)

        dest_path = f"{self.cache_path}/dsp/{output_name}.lib"
        shutil.move(file_name, dest_path)

    def _compute_rayleigh_damping(self, f1: float, f2: float, xi1: float, xi2: float = None) -> Tuple[float, float]:
        """
        Compute Rayleigh damping coefficients α and β.
    
        Parameters:
        -----------
        f1 : float
            First frequency (Hz)
        f2 : float
            Second frequency (Hz)
        xi1 : float
            Damping ratio at f11 (dimensionless, e.g., 0.05 for 5%)
        xi2 : float
            Damping ratio at f2 (dimensionless, e.g., 0.05 for 5%)
    
        Returns:
        --------
        tuple : (alpha, beta)
            Mass-proportional coefficient α (1/s)
            Stiffness-proportional coefficient β (s)
    
        Notes:
        ------
        Rayleigh damping: C = αM + βK
        Damping ratio at frequency ω: ξ = α/(2ω) + βω/2
        """
        xi2 = xi1 if xi2 == None else xi2

        # Convert frequencies to angular frequencies (rad/s)
        omega1 = 2 * np.pi * f1
        omega2 = 2 * np.pi * f2
    
        # Solve the system of equations:
        # ξ1 = α/(2ω1) + βω1/2
        # ξ2 = α/(2ω2) + βω2/2
    
        # Create the coefficient matrix
        A = np.array([
            [1/(2*omega1), omega1/2],
            [1/(2*omega2), omega2/2]
        ])
        
        # Create the right-hand side vector
        b = np.array([xi1, xi2])
    
        # Solve for α and β
        try:
            alpha, beta = np.linalg.solve(A, b)
            return alpha, beta
        except np.linalg.LinAlgError:
            raise ValueError("The two frequencies must be different to compute unique Rayleigh damping coefficients.")
