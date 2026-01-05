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
import numpy as np
import trimesh
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..core.impact_manager import ImpactManager

#from core.impact_manager import ImpactManager

@dataclass
class OptimizeObj:
    impact_manager: ImpactManager
    
    def compute(self, obj_idx: int) -> None:
        print('OptimizeObj.compute')
        config = self.impact_manager.get('config')
        
        # Find the object configuration
        config_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                config_obj = obj
                break
        
        if config_obj is None:
            raise ValueError(f"Object with idx {obj_idx} not found in configuration")
        
        obj_path = config_obj.obj_path
        
        # Find the first .obj file in the directory
        obj_files = []
        if os.path.isdir(obj_path):
            # Look for files with obj extension
            for filename in os.listdir(obj_path):
                if filename.endswith('.obj'):
                    input_obj = f"{obj_path}/{filename}"
                    break
        
        # Load mesh with trimesh
        mesh = trimesh.load(input_obj, force='mesh')
        
        # 1. Fill holes if mesh is not watertight
        if not mesh.is_watertight:
            mesh.fill_holes()
        
        # 2. Remove duplicate vertices and faces
        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
        
        # 3. Fix winding and normals
        mesh.fix_normals()
        
        # 4. Fix degenerate faces
        mesh.remove_degenerate_faces()
        
        # 5. Convert quads to triangles if needed
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are quads (4 vertices per face)
            if mesh.faces.shape[1] >= 4:
                mesh = mesh.triangulate()
        
        # 6. Apply uniform mesh sampling
        min_vertex = config.system.min_vertex
        if config_obj.optimize and min_vertex is not None and min_vertex > 0:
            points, face_indices = trimesh.sample.sample_surface_even(mesh, count=min_vertex)
            mesh = mesh.submesh([face_indices])[0]
        
            # Remove duplicate vertices and faces
            mesh.merge_vertices()
            mesh.remove_duplicate_faces()
    
            # Fix winding and normals
            mesh.fix_normals()
    
            # Fix degenerate faces
            mesh.remove_degenerate_faces()

        # 7. Final validation and fixes
        # Ensure mesh is still valid
        if not mesh.is_watertight:
            # Try to fill holes again
            mesh.fill_holes()
        
        # Remove any remaining degenerate faces
        mesh.remove_degenerate_faces()
        
        # Ensure consistent normals
        mesh.fix_normals()
        
        # 8. Export the fixed mesh
        optimized_obj_path = os.path.join(obj_path, f"optimized_{config_obj.name}.obj")
        
        # Export with vertex normals
        mesh.export(optimized_obj_path, include_normals=True)
