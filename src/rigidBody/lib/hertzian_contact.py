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
import trimesh
from scipy.spatial import ConvexHull
from typing import List, Tuple, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager

@dataclass
class HertzianContact:
    entity_manager: EntityManager
    
    def compute(self, obj_idx: int, other_obj_idx: int, frame_idx: float) -> Tuple[float, float, float]:
        """
        Analyzes impact between two rigidbody meshes using Hertzian contact theory.
        For non-spherical meshes, compute effective contact radius from geometry.
 
        Parameters:
        -----------
        obj_idx: int
            Object id
        other_obj_idx: int
            Object id of 
        sample_idx: float
            Samples index
        """
        config = self.entity_manager.get('config')
        trajectories = self.entity_manager.get('trajectories')
        forces = self.entity_manager.get('forces')

        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
            elif conf_obj.idx == other_obj_idx:
                other_config_obj = conf_obj
                other_young_modulus = other_config_obj.acoustic_shader.young_modulus
                other_poisson_ratio = other_config_obj.acoustic_shader.poisson_ratio
                other_density = other_config_obj.acoustic_shader.density

        for f_idx in forces.keys():
            if forces[f_idx].obj_idx == obj_idx and other_obj_idx in forces[f_idx].other_obj_idx:
                force = forces[f_idx]

        for t_idx in trajectories.keys():
            if trajectories[t_idx].obj_idx == obj_idx:
                trajectory = trajectories[t_idx]
            elif trajectories[t_idx].obj_idx == other_obj_idx:
                other_trajectory = trajectories[t_idx]

        self.mesh1_vertices = trajectory.get_vertices(frame_idx)
        self.mesh1_normals = trajectory.get_normals(frame_idx)
        self.mesh1_faces = trajectory.get_faces()
        
        self.mesh2_vertices = other_trajectory.get_vertices(frame_idx)
        self.mesh2_normals = other_trajectory.get_normals(frame_idx)
        self.mesh2_faces = other_trajectory.get_faces()
        
        # Precompute mesh properties
        self.mesh1_center = self._compute_center_of_mass(self.mesh1_vertices, self.mesh1_faces)
        self.mesh2_center = self._compute_center_of_mass(self.mesh2_vertices, self.mesh2_faces)

        approach_velocity = np.linalg.norm(force.get_relative_velocity(frame_idx))
        R_eff, E_star, impact_duration, results = self._hertzian_impact_duration(approach_velocity, config_obj, other_config_obj)

        return R_eff, E_star, impact_duration
        
    def _compute_center_of_mass(self, vertices, faces):
        """Compute center of mass assuming uniform density."""
        # For convex hull approximation
        hull = ConvexHull(vertices)
        return np.mean(vertices[hull.vertices], axis=0)
    
    def _compute_effective_radius_from_curvature(self, vertices, normals, contact_point, contact_normal):
        """
        Compute effective radius at contact point based on local curvature.
        
        For non-spherical objects, the effective radius R* is computed from
        principal curvatures at the contact point.
        """
        # Find nearest vertices to contact point
        distances = np.linalg.norm(vertices - contact_point, axis=1)
        nearest_indices = np.argsort(distances)[:10]  # Use 10 nearest vertices
        
        # Get normals at nearest points
        nearest_normals = normals[nearest_indices]
        
        # Project vertices onto plane perpendicular to contact normal
        plane_basis = self._find_orthonormal_basis(contact_normal)
        projected_points = []
        
        for idx in nearest_indices:
            vec = vertices[idx] - contact_point
            # Project onto plane
            proj = vec - np.dot(vec, contact_normal) * contact_normal
            projected_points.append(proj)
        
        if len(projected_points) < 3:
            # Fallback: use bounding sphere radius
            return self._compute_bounding_sphere_radius(vertices)
        
        projected_points = np.array(projected_points)
        
        # Fit quadratic surface to approximate local curvature
        # z = a*x^2 + b*y^2 + c*xy (where z is along contact_normal)
        x_coords = np.dot(projected_points, plane_basis[0])
        y_coords = np.dot(projected_points, plane_basis[1])
        
        # Use distances along normal as z values
        z_coords = np.array([np.dot(vertices[idx] - contact_point, contact_normal) for idx in nearest_indices])
        
        # Fit quadratic surface: z = A*x^2 + B*y^2 + C*x*y
        X = np.column_stack([x_coords**2, y_coords**2, x_coords*y_coords])
        coeffs, _, _, _ = np.linalg.lstsq(X, z_coords, rcond=None)
        
        A, B, C = coeffs
        
        # Principal curvatures are related to second derivatives
        # k1 = 2A, k2 = 2B (ignoring cross term for approximation)
        k1 = 2 * A
        k2 = 2 * B
        
        # Effective radius: R* = 1 / sqrt(k1 * k2) if both curvatures positive
        if k1 * k2 > 0:
            R_eff = 1 / np.sqrt(abs(k1 * k2))
        else:
            # Use mean curvature if mixed signs
            R_eff = 2 / (abs(k1) + abs(k2))
        
        return max(R_eff, 0.001)  # Minimum radius to avoid division by zero
    
    def _find_orthonormal_basis(self, normal):
        """Find orthonormal basis for plane perpendicular to given normal."""
        # Find first vector not parallel to normal
        if abs(normal[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        return v1, v2
    
    def _compute_bounding_sphere_radius(self, vertices):
        """Compute radius of bounding sphere as fallback."""
        center = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - center, axis=1)
        return np.max(distances)
    
    def _find_contact_point(self, approach_direction):
        """
        Find approximate contact point between two meshes.
        
        Parameters:
        -----------
        approach_direction: numpy array (3,)
            Direction from mesh1 to mesh mesh2 at impact
            
        Returns:
        --------
        contact_point1, contact_point2: contact points on each mesh
        contact_normal: normal direction at contact
        """
        # Transform approach direction to mesh2's frame relative to mesh1
        approach_dir_norm = approach_direction / np.linalg.norm(approach_direction)
        
        # Find extreme points along approach direction for each mesh
        # For mesh1, find point farthest in -approach direction
        proj1 = np.dot(self.mesh1_vertices - self.mesh1_center, -approach_dir_norm)
        idx1 = np.argmax(proj1)
        point1 = self.mesh1_vertices[idx1]
        normal1 = self.mesh1_normals[idx1]
        
        # For mesh2, find point farthest in +approach direction
        proj2 = np.dot(self.mesh2_vertices - self.mesh2_center, approach_dir_norm)
        idx2 = np.argmax(proj2)
        point2 = self.mesh2_vertices[idx2]
        normal2 = self.mesh2_normals[idx2]
        
        # Contact point is midpoint between extreme points
        contact_point = (point1 + point2) / 2
        
        # Contact normal is average of surface normals (pointing from mesh1 to mesh2)
        contact_normal = normal2 - normal1
        if np.linalg.norm(contact_normal) > 0:
            contact_normal = contact_normal / np.linalg.norm(contact_normal)
        else:
            contact_normal = approach_dir_norm
        
        return point1, point2, contact_normal
    
    def _compute_effective_contact_radius(self, contact_point1, contact_point2, contact_normal, approach_direction):
        """
        Compute combined effective radius for Hertzian contact.
        
        For two bodies with radii R1 and R2, the effective radius is:
        1/R* = 1/R1 + 1/R2
        """
        # Compute effective radius for each mesh at contact point
        R1 = self._compute_effective_radius_from_curvature(self.mesh1_vertices, self.mesh1_normals, contact_point1, -contact_normal)
        
        R2 = self._compute_effective_radius_from_curvature(self.mesh2_vertices, self.mesh2_normals, contact_point2, contact_normal)
        
        # Combined effective radius
        if R1 > 0 and R2 > 0:
            R_star = 1 / (1/R1 + 1/R2)
        else:
            R_star = max(R1, R2)
        
        return R_star, R1, R2
    
    def _hertzian_impact_duration(self, approach_velocity: float, config_obj: Any, other_config_obj: Any):
        """
        Calculate impact duration using Hertzian theory.
        
        Parameters:
        -----------
        approach_velocity: float
            Relative approach velocity at impact (m/s)
        material_properties: dict
            Dictionary containing:
            - E1, E2: Young's moduli (Pa)
            - nu1, nu2: Poisson's ratios
            - m1, m2: masses (kg)
            - Optional: 'method' for duration calculation
            
        Returns:
        --------
        duration: impact duration in seconds
        R_star: effective radius
        E_start: effective_modulus
        results: dictionary with all computed values
        """
        # Extract material properties
        E1 = config_obj.acoustic_shader.young_modulus
        E2 = other_config_obj.acoustic_shader.young_modulus
        nu1 = config_obj.acoustic_shader.poisson_ratio
        nu2 = other_config_obj.acoustic_shader.poisson_ratio

        # Calculate masses
        mesh1 = trimesh.Trimesh(vertices=self.mesh1_vertices, vertex_normals=self.mesh1_normals, faces=self.mesh1_faces)
        mesh2 = trimesh.Trimesh(vertices=self.mesh2_vertices, vertex_normals=self.mesh2_normals, faces=self.mesh2_faces)
        mesh1.density = config_obj.acoustic_shader.density
        mesh2.density = other_config_obj.acoustic_shader.density
        m1 = mesh1.mass
        m2 = mesh2.mass
        
        # Approach direction (from mesh1 center to mesh2 center)
        approach_direction = self.mesh2_center - self.mesh1_center
        if np.linalg.norm(approach_direction) == 0:
            approach_direction = np.array([1, 0, 0])  # Default direction
        
        # Find contact point and normal
        contact_point1, contact_point2, contact_normal = self._find_contact_point(approach_direction)

        # Compute effective contact radius
        R_star, R1, R2 = self._compute_effective_contact_radius(contact_point1, contact_point2, contact_normal, approach_direction)
        
        # Compute effective elastic modulus
        E_star = 1 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2)
        
        # Compute reduced mass
        m_reduced = (m1 * m2) / (m1 + m2)
        
        # Hertzian contact parameters
        # Maximum compression (delta_max)
        # From energy conservation: 1/2 * m_reduced * v^2 = 2/5 * E_star * sqrt(R_star) * delta_max^(5/2)
        
        # Solve for delta_max
        v = approach_velocity
        numerator = (15/8 * m_reduced * v**2 / (E_star * np.sqrt(R_star)))**(2/5)
        delta_max = numerator
        
        # Impact duration (approximate)
        # Method 1: Simple harmonic approximation
        # Contact stiffness: k = (4/3) * E_star * sqrt(R_star * delta_max)
        k = (4/3) * E_star * np.sqrt(R_star * delta_max)
        
        # Natural frequency
        omega = np.sqrt(k / m_reduced)
        
        # Duration for half-period of oscillation
        duration_simple = np.pi / omega
        
        # Method 2: More accurate formula from Hertzian theory
        # T = 2.94 * (m_reduced^2 / (E_star^2 * R_star * v))^(1/5)
        duration_hertz = 2.94 * (m_reduced**2 / (E_star**2 * R_star * v))**(1/5)
        
        # Use the more accurate formula
        duration = duration_hertz
        
        # Maximum contact force
        F_max = (4/3) * E_star * np.sqrt(R_star) * delta_max**(3/2)
        
        # Contact area radius at maximum compression
        a_max = np.sqrt(R_star * delta_max)
        
        results = {
            'duration': duration,
            'duration_simple': duration_simple,
            'duration_hertz': duration_hertz,
            'effective_radius': R_star,
            'radius_mesh1': R1,
            'radius_mesh2': R2,
            'effective_modulus': E_star,
            'reduced_mass': m_reduced,
            'max_compression': delta_max,
            'max_force': F_max,
            'contact_radius': a_max,
            'contact_point_mesh1': contact_point1,
            'contact_point_mesh2': contact_point2,
            'contact_normal': contact_normal,
            'stiffness': k,
            'natural_frequency': omega
        }
        
        return R_star, E_star, duration, results
