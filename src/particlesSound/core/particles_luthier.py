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
import pickle
import numpy as np
from typing import Any, Dict, List
from dataclasses import dataclass

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.voxels_modal_ir_convolver import VoxelsModalIRConvolver

@dataclass
class ParticlesLuthier:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.dsp_path = f"{config.system.cache_path}/dsp"

    def compute(self, particles_obj_idx: int):
        """
        Initializes the VoxelsModalIRConvolver instances based on the composer's analysis.
        Creates a pool of convolvers for each material involved in the collisions.
        """
        config = self.entity_manager.get('config')
        
        composer_dir = f"{config.system.cache_path}/particles_composer"
        filepath = f"{composer_dir}/convolvers_{particles_obj_idx}.pkl"
        
        if not os.path.exists(filepath):
            debug_print(f"No composer data found for {particles_obj_idx}")
            return

        with open(filepath, 'rb') as f:
            max_concurrent = pickle.load(f)

        # Create a pool of convolvers for each object
        convolver_pool = {}
        for obj_idx, count in max_concurrent.items():
            obj_config = next((o for o in config.objects if o.idx == obj_idx), None)
            if not obj_config:
                debug_print(f"Config for object {obj_idx} not found, skipping.")
                continue
            
            lib_path = f"{self.dsp_path}/{obj_config.name}.lib"
            if not os.path.exists(lib_path):
                debug_print(f"Modal lib for {obj_config.name} not found at {lib_path}, skipping.")
                continue
            
            # Create a pool of VoxelsModalIRConvolver instances for this object
            convolver_pool[obj_idx] = [
                VoxelsModalIRConvolver(
                    sample_rate=config.system.sample_rate,
                    modal_libs={obj_idx: lib_path}
                ) for _ in range(count)
            ]
            debug_print(f"Created {count} convolvers for object {obj_config.name} (idx: {obj_idx})")

        # Save the pool of convolvers
        # We can't pickle the convolvers directly due to their state, so we save the parameters
        # to recreate them in the player.
        luthier_dir = f"{config.system.cache_path}/particles_luthier"
        os.makedirs(luthier_dir, exist_ok=True)
        luthier_filepath = f"{luthier_dir}/pool_{particles_obj_idx}.pkl"
        
        # We will save the mapping of obj_idx to the number of convolvers needed
        with open(luthier_filepath, 'wb') as f:
            pickle.dump(max_concurrent, f)

        debug_print(f"Luthier finished for particles object {particles_obj_idx}. Convolver pools are ready to be instantiated.")

