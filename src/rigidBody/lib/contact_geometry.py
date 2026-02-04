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

import numpy as np
from scipy.spatial import KDTree
import itertools

class ContactGeometry:
    def __init__(self, mesh1_vertices, mesh1_faces, mesh1_center, mesh2_vertices, mesh2_faces, mesh2_center):
        """
        Initialize collision detector with two meshes.
        
        Parameters:
        -----------
        mesh1_vertices, mesh2_vertices: numpy array of shape (n, 3)
            Vertex positions in world coordinates (meters)
        mesh1_faces, mesh2_faces: numpy array of shape (m, 3)
            Face indices (triangles)
        mesh1_center, mesh2_center: numpy array of shape (3,)
            Center positions of meshes
        """
        self.mesh1 = {
            'vertices': mesh1_vertices,
            'faces': mesh1_faces,
            'center': mesh1_center,
            'normals': self._compute_face_normals(mesh1_vertices, mesh1_faces)
        }
        
        self.mesh2 = {
            'vertices': mesh2_vertices,
            'faces': mesh2_faces,
            'center': mesh2_center,
            'normals': self._compute_face_normals(mesh2_vertices, mesh2_faces)
        }
        
        # Build KD-trees for efficient nearest neighbor search
        self.kdtree1 = KDTree(mesh1_vertices)
        self.kdtree2 = KDTree(mesh2_vertices)
        
    def _find_contact_points(self):
        """Find contact points between two meshes."""
        contact_points = []
        
        # Find vertices of mesh1 inside mesh2
        for i, vertex in enumerate(self.mesh1['vertices']):
            # Simple proximity check (can be enhanced with more sophisticated methods)
            dist, idx = self.kdtree2.query(vertex)
            if dist < 0.01:  # Threshold for contact (1 cm)
                contact_points.append(vertex)
        
        # Find vertices of mesh2 inside mesh1
        for i, vertex in enumerate(self.mesh2['vertices']):
            dist, idx = self.kdtree1.query(vertex)
            if dist < 0.01:  # Threshold for contact (1 cm)
                contact_points.append(vertex)
        
        # If no vertex contacts found, find closest points between meshes
        if len(contact_points) == 0:
            # Find closest points between the two two meshes
            distances, indices = self.kdtree1.query(self.mesh2['vertices'])
            min_idx = np.argmin(distances)
            closest_point1 = self.mesh1['vertices'][indices[min_idx]]
            closest_point2 = self.mesh2['vertices'][min_idx]
            
            # Use midpoint as contact point
            contact_point = (closest_point1 + closest_point2) / 2
            contact_points.append(contact_point)
        
        return np.array(contact_points)
    
    def _compute_contact_normal(self, contact_points):
        """Compute contact normal from contact points."""
        if len(contact_points) == 0:
            return np.array([0, 0, 1])
        
        # Use direction between mesh centers as initial guess
        center_vec = self.mesh2['center'] - self.mesh1['center']
        if np.linalg.norm(center_vec) > 1e-10:
            normal_guess = center_vec / np.linalg.norm(center_vec)
        else:
            normal_guess = np.array([0, 0, 1])
        
        # Refine normal using contact point distribution
        if len(contact_points) > 1:
            # Fit a plane to contact points
            centroid = np.mean(contact_points, axis=0)
            centered = contact_points - centroid
            
            # Use PCA to find normal (smallest eigenvector)
            if len(centered) >= 3:
                cov = centered.T @ centered
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # Smallest eigenvalue
                
                # Ensure normal points from mesh1 to mesh2
                if np.dot(normal, normal_guess) < 0:
                    normal = -normal
                return normal
        
        return normal_guess

    def _compute_face_normals(self, vertices, faces):
        """Compute face normals for all triangles."""
        normals = []
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            normals.append(normal)
        return np.array(normals)

    def get_contact_normal(self):
        # Find contact points
        contact_points = self._find_contact_points()

        if len(contact_points) == 0: 
            return None 

        # Compute contact normal
        return self._compute_contact_normal(contact_points)

