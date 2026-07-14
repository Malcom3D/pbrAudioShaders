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

import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class ScoreEvent:
    """Represents the score for a single collision"""
    coll_obj: int
    type: np.ndarray # shape(total_samples,1)
    vertex_ids: np.ndarray  # shape(total_samples,n_vertices)
    contact_area: np.ndarray = None # shape(total_samples,1)
    force: np.ndarray = None # shape(total_samples,1) Excitation force magnitude
    coupling_data: np.ndarray = None # shape(total_samples,1) Array of [other_obj_idx, coupling_strength] pairs

@dataclass
class ScoreTrack:
    """Represents a score track for a single object."""
    obj_idx: int
    obj_name: str
    is_final: bool = False
    events: List[ScoreEvent] = field(default_factory=list)

    def add_event(self, event: ScoreEvent) -> None:
        """Add an event to the track."""
        self.events.append(event)

