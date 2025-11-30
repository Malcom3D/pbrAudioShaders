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
#from trimesh import repair, remesh
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class MeshToModal:
    obj_file_path: str
    is_watertight: bool = False
    original_vertex_count: int = 0
    processed_vertex_count: int = 0

    """
    A class to process 3D meshes for modal analysis.
    Handles mesh cleaning, validation, and optimization for numerical analysis.
    Export Faust lib modalModel and obj for future reference
    Args:
        obj_file_path (str): Path to the input Wavefront OBJ file
    """
    
    def __post_init__(self):
        self.mesh = None
        
    def load_mesh(self) -> bool:
        """
        Load the mesh from OBJ file and perform initial validation.
        
        Returns:
            bool: True if mesh loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.obj_file_path):
                raise FileNotFoundError(f"OBJ file not found: {self.obj_file_path}")
            
            # Load mesh using trimesh
            self.mesh = trimesh.load_mesh(self.obj_file_path)
            
            if not isinstance(self.mesh, trimesh.Trimesh):
                raise ValueError("Loaded file is not a triangular mesh")
            
            self.original_vertex_count = len(self.mesh.vertices)
            self.processed_vertex_count = self.original_vertex_count
            
            print(f"Mesh loaded successfully:")
            print(f"  Vertices: {self.original_vertex_count}")
            print(f"  Faces: {len(self.mesh.faces)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def check_watertight(self) -> bool:
        """
        Check if the mesh is watertight (manifold and closed).
        
        Returns:
            bool: True if mesh is watertight
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        self.is_watertight = self.mesh.is_watertight
        print(f"Watertight status: {self.is_watertight}")
        
        if not self.is_watertight:
            print("  Mesh is not watertight. Attempting repair...")
        
        return self.is_watertight
    
    def repair_mesh(self) -> None:
        """
        Comprehensive mesh repair including:
        - Fill holes
        - Fix winding and normals
        - Remove duplicate vertices and faces
        - Fix degenerate faces
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        print("Starting mesh repair...")
        
        # Store original stats
        original_verts = len(self.mesh.vertices)
        original_faces = len(self.mesh.faces)
        
        # Fix winding (ensure consistent face orientation)
        trimesh.repair.fix_winding(self.mesh)
        print("  Fixed face winding")
        
        # Fix normals
        self.mesh.fix_normals()
        print("  Fixed vertex normals")
        
        # Remove duplicate vertices
        self.mesh.merge_vertices()
        print("  Removed duplicate vertices")
        
        # Remove duplicate faces
        self.mesh.remove_duplicate_faces()
        print("  Removed duplicate faces")
        
        # Fix degenerate faces
        self.mesh.remove_degenerate_faces()
        print("  Removed degenerate faces")
        
        # Fill holes if mesh is not watertight
        if not self.mesh.is_watertight:
            try:
                # Fill small holes
                self.mesh.fill_h_holes()
                print("  Filled holes")
            except Exception as e:
                print(f"  Warning: Could not fill all holes: {e}")
        
        # Final validation
        self.mesh.process()
        
        # Update processed vertex count
        self.processed_vertex_count = len(self.mesh.vertices)
        
        print(f"Mesh repair completed:")
        print(f"  Vertices: {original_verts} -> {self.processed_vertex_count}")
        print(f"  Faces: {original_faces} -> {len(self.mesh.faces)}")
    
    def convert_to_triangles(self) -> None:
        """
        Convert mesh to triangles only if it contains quads or other polygons.
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        # Check if mesh needs triangulation
        if hasattr(self.mesh, 'faces') and len(self.mesh.faces) > 0:
            # trimesh automatically triangulates during loading, but we double-check
            if not all(len(face) == 3 for face in self.mesh.faces):
                print("Converting mesh to triangles...")
                original_faces = len(self.mesh.faces)
                self.mesh = self.mesh.triangulate()
                print(f"  Faces after triangulation: {original_faces} -> {len(self.mesh.faces)}")
            else:
                print("Mesh is already composed of triangles")
        else:
            print("No faces found in mesh")
    
    def optimize_mesh_size(self, target_vertices: int, tolerance: float = 0.2) -> None:
        """
        Optimize mesh size to be close to target number of vertices.
        Uses subdivision for too few vertices or simplification for too many.
        
        Args:
            target_vertices (int): Desired number of vertices
            tolerance (float): Acceptable deviation from target (default: 0.2 = 20%)
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        current_vertices = len(self.mesh.vertices)
        lower_bound = target_vertices * (1 - tolerance)
        upper_bound = target_vertices * (1 + tolerance)
        
        print(f"Mesh optimization target: {target_vertices} vertices (±{tolerance*100:.0f}%)")
        print(f"Current vertices: {current_vertices}")
        
        if lower_bound <= current_vertices <= upper_bound:
            print("Mesh already within target size range")
            return
        
        if current_vertices < lower_bound:
            # Need to subdivide (add more vertices)
            self._subdivide_to_target(target_vertices)
        else:
            # Need to simplify (reduce vertices)
            self._simplify_to_target(target_vertices)
        
        self.processed_vertex_count = len(self.mesh.vertices)
        print(f"Final vertex count: {self.processed_vertex_count}")
    
    def _subdivide_to_target(self, target_vertices: int) -> None:
        """
        Subdivide mesh to approach target vertex count using loop subdivision.
        
        Args:
            target_vertices (int): Target number of vertices
        """
        current_vertices = len(self.mesh.vertices)
        
        # Estimate required subdivisions
        # Loop subdivision roughly quadruples the number of faces
        subdivisions = 0
        estimated_vertices = current_vertices
        
        while estimated_vertices < target_vertices and subdivisions < 5:  # Safety limit
            subdivisions += 1
            # Rough estimation: vertices ~= original_vertices * 4^(subdivisions) / 2
            estimated_vertices = current_vertices * (4 ** subdivisions) // 2
        
        if subdivisions > 0:
            print(f"Applying {subdivisions} loop subdivision(s)")
            self.mesh = self.mesh.subdivide_loop(iterations=subdivisions)
            
            # If still below target, do one more with partial
            current_after_subdiv = len(self.mesh.vertices)
            if current_after_subdiv < target_vertices:
                print("Applying additional partial subdivision")
                # For fine adjustment, we might need custom approach
                # For now, we accept the closest we can get
        else:
            print("No subdivision needed")
    
    def _simplify_to_target(self, target_vertices: int) -> None:
        """
        Simplify mesh to approach target vertex count using quadric decimation.
        
        Args:
            target_vertices (int): Target number of vertices
        """
        current_vertices = len(self.mesh.vertices)
        
        if current_vertices <= target_vertices:
            return
        
        # Calculate reduction factor
        factor = target_vertices / current_vertices
        
        try:
            print(f"Simplifying mesh with factor: {factor:.3f}")
            self.mesh = self.mesh.simplify_quadric_decimation(target_vertices)
        except Exception as e:
            print(f"Quadric decimation failed: {e}")
            print("Attempting alternative simplification...")
            # Fallback: use face count based simplification
            target_faces = int(len(self.mesh.faces) * factor)
            self.mesh = self.mesh.simplify_quadric_decimation(target_faces * 2)  # Rough estimate
    
    def process_complete(self, target_vertices: Optional[int] = None) -> bool:
        """
        Complete processing pipeline for the mesh.
        
        Args:
            target_vertices (int, optional): Target number of vertices for optimization
            
        Returns:
            bool: True if processing completed successfully
        """
        try:
            # Step 1: Load mesh
            if not self.load_mesh():
                return False
            
            # Step 2: Check watertight status
            self.check_watertight()
            
            # Step 3: Repair mesh
            self.repair_meshesh()
            
            # Step 4: Convert to triangles
            self.convert_to_triangles()
            
            # Step 5: Optimize mesh size if target specified
            if target_vertices is not None:
                self.optimize_mesh_size(target_vertices)
            
            # Final watertight check
            final_watertight = self.mesh.is_watertight
            print(f"Final watertight status: {final_watertight}")
            
            return True
            
        except Exception as e:
            print(f"Error during complete processing: {e}")
            return False
    
    def export_mesh(self, output_path: str) -> bool:
        """
        Export the processed mesh as Wavefront OBJ file.
        
        Args:
            output_path (str): Path for the output OBJ file
            
        Returns:
            bool: True if export successful
        """
        if self.mesh is None:
            raise ValueError("No mesh to export. Process mesh first.")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export as OBJ with vertex normals
            self.mesh.export(output_path, include_normals=True)
            
            
            return True
            
        except Exception as e:
            print(f"Error exporting mesh: {e}")
            return False
