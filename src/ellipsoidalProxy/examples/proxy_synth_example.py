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

"""
Example usage of the lightweight proxy synthesizer.
"""

import os
import numpy as np
from physicsSolver import EntityManager, physicsEngine
from ellipsoidalProxy import ProxyEngine

# Initialize entity manager
entity_manager = EntityManager("config.json")

# Run physics solver first
physics_engine = physicsEngine(entity_manager)
physics_engine.bake()

# Initialize proxy engine
proxy_engine = ProxyEngine(entity_manager)

# Process all proxy objects
proxy_engine.compute()

print("Proxy synthesis completed!")

