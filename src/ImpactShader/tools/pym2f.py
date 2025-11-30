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
from typing import List
from dataclasses import dataclass

@dataclass
class Pym2f:
    mesh2faust: str = "mesh2faust"
    low_freq: float = 1.0
    high_freq: float = 24000

    def convert(self, obj_file: str, young_modulus: float = None, poisson_ration: float = None, density: float = None, damping: float = None, nfemmodes: int = None, minmode: float = None, maxmode: float = None, expos: List = None, output_name: str = None):
        """
            Input parameters

            - obj_file: volumetric mesh in Wavefront .obj file (triangulated mesh with normals)

            - young_modulus: Young's modulus (in N/m^2)
            - poisson_ration: Poisson's ratio (no unit)
            - density: Density (in kg/m^3)
            - damping: Rayleigh damping ratio (no unit)
            - nfemmodes: number of modes to be computed for the finite element analysis
            - minmode: minimum frequency of the lowest mode (in Hz)
            - maxmode: maximum frequency of the highest mode (in Hz)
            - expos: list of vertex of active excitation positions (e.g. 89 63 45 ...)
            - output_name: name for generated Faust modal model lib
        """
        bin_dir = f"{os.path.dirname(os.path.abspath(sys.modules[Pym2f.__module__].__file__))}/../bin"
        os.environ['LD_LIBRARY_PATH'] = bin_dir
        self.mesh2faust = f"{bin_dir}/{self.mesh2faust}"
        cmd = f"{self.mesh2faust} --debug "
        if not young_modulus == None and poisson_ration == None and density == None:
            damping = damping if not damping == None else 0.02 # conservative value
                """
                compute alpha and beta Rayleigh damping coefficient from low and high frequency:
                alpha: stiffness-proportional damping coefficient (in seconds)
                beta: mass-proportional damping coefficient (in 1/seconds)
                """
                omega_1 = 2 * np.pi * self.low_freq 
                omega_2 = 2 * np.pi * self.high_freq
                alpha_rayleigh = damping * 2 * omega_1 + omega_1 * omega_2 ** 2
                beta_rayleigh = omega_1 + omega_2 * damping
            cmd += f"--material {young_modulus} {poisson_ration} {density} {alpha_rayleigh} {beta_rayleigh} "
        cmd += f"--infile {obj_file} "
        if not nfemmodes == None:
            cmd += f"--nfemmodes {nfemmodes} "
        if not minmode == None:
            cmd += f"--minmode {minmode} "
        else:
            cmd += f"--minmode {self.low_freq} "
        if not maxmode == None:
            cmd += f"--maxmode {maxmode} "
        else:
            cmd += f"--maxmode {self.high_freq} "
        if not expos == None:
            verts = ''
            for pos in expos:
                verts += f"{pos} "
            cmd += f"--expos {verts} "
        if not output_name == None:
            cmd += f"--name {output_name} "

        exit_code = os.system(cmd)
        if not exit_code == 0:
            print('Error')
