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
    
    def _get_edges_from_faces(self, faces):
        """Extract unique edges from faces."""
        edges = set()
        for face in faces:
            # Add all edges of the triangle
            edges.add(tuple(sorted([face[0], face[1]])))
            edges.add(tuple(sorted([face[1], face[2]])))
            edges.add(tuple(sorted([face[2], face[0]])))
        return list(edges)
    
    def _get_support_points(self, vertices, direction):
        """Find extreme points in a given direction."""
        projections = vertices @ direction
        max_idx = np.argmax(projections)
        min_idx = np.argmin(projections)
        return vertices[max_idx], vertices[min_idx]
    
    def _project_vertices(self, vertices, axis):
        """Project vertices onto an axis."""
        return vertices @ axis
    
    def _check_separating_axis(self, axis):
        """Check if axis is a separating axis between two meshes."""
        # Project both meshes onto the axis
        proj1 = self._project_vertices(self.mesh1['vertices'], axis)
        proj2 = self._project_vertices(self.mesh2['vertices'], axis)
        
        
        # Check for overlap
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return True, 0  # Separating axis found
        
        # Calculate penetration depth
        overlap = min(max1, max2) - max(min1, min2)
        return False, overlap
    
    def detect_collision_sat(self):
        """
        Detect collision using Separating Axis Theorem.
        Returns collision info if meshes intersect.
        """
        # Test face normals of mesh1
        for normal in self.mesh1['normals']:
            is_separating, overlap = self._check_separating_axis(normal)
            if is_separating:
                return None  # No collision
        
        # Test face normals of mesh2
        for normal in self.mesh2['normals']:
            is_separating, overlap = self._check_separating_axis(normal)
            if is_separating:
                return None  # No collision
        
        # Test cross products of edges (for 3D SAT)
        edges1 = self._get_edges_from_faces(self.mesh1['faces'])
        edges2 = self._get_edges_from_faces(self.mesh2['faces'])
        
        # Get unique edge directions
        edge_dirs1 = []
        for edge in edges1[:15]:  # Limit to avoid too many tests
            v0, v1 = self.mesh1['vertices'][edge[0]], self.mesh1['vertices'][edge[1]]
            edge_dirs1.append(v1 - v0)
        
        edge_dirs2 = []
        for edge in edges2[:15]:
            v0, v1 = self.mesh2['vertices'][edge[0]], self.mesh2['vertices'][edge[1]]
            edge_dirs2.append(v1 - v0)
        
        # Test cross products of edge pairs
        for dir1 in edge_dirs1[:5]:  # Further limit for performance
            for dir2 in edge_dirs2[:5]:
                axis = np.cross(dir1, dir2)
                if np.linalg.norm(axis) < 1e-10:
                    continue
                axis = axis / np.linalg.norm(axis)
                
                is_separating, overlap = self._check_separating_axis(axis)
                if is_separating:
                    return None  # No collision
        
        # If we get here, meshes are colliding
        return self._compute_contact_info()
    
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
    
    def _compute_penetration_depth(self, contact_normal):
        """Compute penetration depth along contact normal."""
        # Project both meshes onto contact normal
        proj1 = self._project_vertices(self.mesh1['vertices'], contact_normal)
        proj2 = self._project_vertices(self.mesh2['vertices'], contact_normal)
        
        # Calculate overlap
        max1, min1 = np.max(proj1), np.min(proj1)
        max2, min2 = np.max(proj2), np.min(proj2)
        
        penetration = min(max1, max2) - max(min1, min2)
        return max(0, penetration)  # Ensure non-negative
    
    def _compute_collision_area(self, contact_points, contact_normal):
        """Estimate collision area from contact points."""
        if len(contact_points) < 3:
            # If few contact points, estimate area based on penetration
            penetration = self._compute_penetration_depth(contact_normal)
            # Rough estimate: area ~ (penetration)^2
            return max(0.001, penetration ** 2)
        
        # Project contact points onto contact plane
        centroid = np.mean(contact_points, axis=0)
        
        # Create basis for contact plane
        if abs(contact_normal[0]) > abs(contact_normal[1]):
            tangent1 = np.array([contact_normal[2], 0, -contact_normal[0]])
        else:
            tangent1 = np.array([0, contact_normal[2], -contact_normal[1]])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(contact_normal, tangent1)
        
        # Project points onto plane
        projected = []
        for point in contact_points:
            vec = point - centroid
            u = np.dot(vec, tangent1)
            v = np.dot(vec, tangent2)
            projected.append([u, v])
        
        projected = np.array(projected)
        
        # Compute convex hull area (simplified)
        if len(projected) >= 3:
            # Sort points by angle
            angles = np.arctan2(projected[:, 1], projected[:, 0])
            sorted_idx = np.argsort(angles)
            hull_points = projected[sorted_idx]
            
            # Compute area using shoelace formula
            area = 0
            for i in range(len(hull_points)):
                j = (i + 1) % len(hull_points)
                area += hull_points[i, 0] * hull_points[j, 1]
                area -= hull_points[j, 0] * hull_points[i, 1]
            
            area = abs(area) / 2
            return max(0.001, area)
        
        return 0.001  # Minimum area
 
    def get_contact_normal(self):
        # Find contact points
        contact_points = self._find_contact_points()

        if len(contact_points) == 0:
            return None

        # Compute contact normal
        return self._compute_contact_normal(contact_points)

    def _compute_contact_info(self):
        """Compute comprehensive contact information."""
        # Find contact points
        contact_points = self._find_contact_points()
        
        if len(contact_points) == 0:
            return None
        
        # Compute contact normal
        contact_normal = self._compute_contact_normal(contact_points)
        
        # Compute penetration depth
        penetration_depth = self._compute_penetration_depth(contact_normal)
        
        # Compute collision area
        collision_area = self._compute_collision_area(contact_points, contact_normal)
        
        # Compute average contact point
        avg_contact_point = np.mean(contact_points, axis=0)
        
        return {
            'contact_point': avg_contact_point,
            'contact_normal': contact_normal,
            'penetration_depth': penetration_depth,
            'collision_area': collision_area,
            'contact_points': contact_points,
            'num_contact_points': len(contact_points)
        }
    
    def detect_collision(self):
        """
        Main collision detection function.
        Returns collision information dictionary or None if no collision.
        """
        return self.detect_collision_sat()
