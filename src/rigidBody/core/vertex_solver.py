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

import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_pose, _load_mesh

@dataclass
class VertexSolver:
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        tmp_trajectories = self.entity_manager.get('trajectories')
        for index in tmp_trajectories:
            if 'tmpTrajectoryData' in str(type(tmp_trajectories[index])) and tmp_trajectories[index].obj_idx == obj_idx:
                tmp_trajectory = tmp_trajectories[index]
                frame = tmp_trajectory.frame
                impact_position = tmp_trajectory.position
                impact_rotation = tmp_trajectory.rotation
                # Find the object config
                config_obj = None
                for obj in config.objects:
                    if obj.idx == obj_idx:
                        config_obj = obj
                        break
                
                if config_obj:
                    # Load pose data
                    positions, rotations = _load_pose(config_obj)
                    
                    # Get vertices at impact time using interpolation
                    vertices_impact = self._impact_vertices(config_obj, positions, rotations, impact_position, impact_rotation, frame)
                    tmp_trajectory.add_data('vertices', vertices_impact)

    def _impact_vertices(self, config_obj, positions: np.ndarray, rotations: np.ndarray, impact_position: np.ndarray, impact_rotation: np.ndarray, frame: float, n_frames: int = 3) -> np.ndarray:
        """
        Compute vertex positions at impact time by loading and interpolating vertices from multiple surrounding frames for better accuracy.
        
        Parameters:
        -----------
        n_frames : int
            Number of frames to use on each side of impact for averaging
        """
        # Get frame range for averaging
        frame_idx = int(round(frame))
        start_frame = max(0, frame_idx - n_frames)
        end_frame = min(len(positions), frame_idx + n_frames + 1)
        
        all_local_vertices = []
        
        # Collect local vertices from multiple frames
        for i in range(start_frame, end_frame):
            vertices_i, _, _ = _load_mesh(config_obj, i)
            pos_i = positions[i]
            rot_i = Rotation.from_euler('XYZ', rotations[i])
            R_i = rot_i.as_matrix()
            
            # Convert to local coordinates
            local_vertices_i = (R_i.T @ (vertices_i - pos_i).T).T
            all_local_vertices.append(local_vertices_i)
        
        # Average local coordinates
        avg_local_vertices = np.mean(all_local_vertices, axis=0)
        
        # Apply impact transformation
        R_impact = Rotation.from_euler('XYZ', impact_rotation).as_matrix()
        vertices_impact = (R_impact @ avg_local_vertices.T).T + impact_position
        
        return vertices_impact
