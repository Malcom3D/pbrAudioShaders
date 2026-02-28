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
import json
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ModalVertices:
    obj_idx: int
    vertices: np.ndarray
    connected_area: float = None # only for static contact

    def add_vertices(self, vertices: np.ndarray):
        self.vertices = np.unique(np.append(self.vertices, vertices))

    def get_vertices(self):
        return self.vertices

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ModalVertices to a serializable dictionary.

        Returns:
            Dictionary representation of the modal vertices data.
        """
        data_dict = {}

        # Handle Enum serialization
        data_dict['obj_idx'] = self.obj_idx
        data_dict['connected_area'] = self.connected_area

        # Handle numpy array serialization
        data_dict['vertices'] = self.vertices.tolist()

        return data_dict

    def save(self, filepath: str, indent: int = 2) -> None:
        """
        Save modal vertices data to a JSON file.

        Args:
            filepath: Path to save the JSON file
            indent: JSON indentation level
        """
        data_dict = self.to_dict()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=indent)

    @classmethod
    def load(cls, filepath: str) -> 'ModalVertices':
        """
        Load modal vertices data from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ModalVertices instance
        """
        with open(filepath, 'r') as f:
            data_dict = json.load(f)

        # Convert list back to numpy array
        data_dict['vertices'] = np.array(data_dict['vertices'])

        return cls(**data_dict)
