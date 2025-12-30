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
from dask import delayed, compute
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..core.mesh2modal import Mesh2Modal

@dataclass
class PreBaking:
    entity_manager: EntityManager

    def compute(self):
        config = self.entity_manager.get('config')
        mesh2modal = Mesh2Modal(self.entity_manager)

        tasks = [mesh2modal.compute(obj.idx) for obj in config.objects]
        results = compute(*tasks)
