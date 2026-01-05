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

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..lib.acoustic_shader import AcousticShader, AcousticProperties, AcousticCoefficients

@dataclass
class SystemConfig:
    sample_rate: int = 48000
    bit_depth: int = 32
    fps: int = 24 # video fps
    fps_base: int = 1
    subframes: int = 1 # video subframes
    collision_margin: float = 0.05
    min_vertex: int = None
    cache_path: str = "./pbrAudioCache/"

@dataclass
class ObjectConfig:
    idx: int
    name: str
    obj_path: str
    pose_path: str
    static: bool
    tiny_edge: float = None
    acoustic_shader: Optional[AcousticShader] = None

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.data = json.load(f)

        self.system = SystemConfig(**self.data.get('system', {}))

        # Handle objects with nested acoustic_shader
        self.objects = []
        for o in self.data.get('objects', []):
            acoustic_shader_data = o.get('acoustic_shader', {})

            object_config = ObjectConfig(
                **{k: v for k, v in o.items() if k != 'acoustic_shader'},
                acoustic_shader=self._create_acoustic_shader(acoustic_shader_data) if acoustic_shader_data else None
            )
            self.objects.append(object_config)

    def _create_acoustic_shader(self, shader_data: Dict[str, Any]) -> AcousticShader:
        """Create AcousticShader instance from dictionary data"""
        acoustic_props_data = shader_data.get('acoustic_properties', {})

        # Create AcousticCoefficients for each property
        acoustic_properties = AcousticProperties()

        if 'absorption' in acoustic_props_data:
            abs_data = acoustic_props_data['absorption']
            acoustic_properties.absorption = AcousticCoefficients(
                frequencies=np.array(abs_data['frequencies']),
                coefficients=np.array(abs_data['coefficients'])
            )

        if 'refraction' in acoustic_props_data:
            refr_data = acoustic_props_data['refraction']
            acoustic_properties.refraction = AcousticCoefficients(
                frequencies=np.array(refr_data['frequencies']),
                coefficients=np.array(refr_data['coefficients'])
            )

        if 'reflection' in acoustic_props_data:
            refl_data = acoustic_props_data['reflection']
            acoustic_properties.reflection = AcousticCoefficients(
                frequencies=np.array(refl_data['frequencies']),
                coefficients=np.array(refl_data['coefficients'])
            )

        if 'scattering' in acoustic_props_data:
            scat_data = acoustic_props_data['scattering']
            acoustic_properties.scattering = AcousticCoefficients(
                frequencies=np.array(scat_data['frequencies']),
                coefficients=np.array(scat_data['coefficients'])
            )

        # Create AcousticShader
        return AcousticShader(
            sound_speed=shader_data.get('sound_speed', 343.0),
            young_modulus=shader_data.get('young_modulus', []),
            poisson_ratio=shader_data.get('poisson_ratio', []),
            density=shader_data.get('density', 1.225),
            damping=shader_data.get('damping', []),
            friction=shader_data.get('friction', []),
            low_frequency=shader_data.get('low_frequency', 1.0),
            high_frequency=shader_data.get('high_frequency', 24000.0),
            acoustic_properties=acoustic_properties
        )
