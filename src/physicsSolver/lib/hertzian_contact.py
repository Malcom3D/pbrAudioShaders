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
import math
import trimesh
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial import ConvexHull

from ..core.entity_manager import EntityManager
from ..lib.force_data import ContactType

@dataclass
class HertzianContact:
    """Hertzian contact mechanics calculator for rigid body collisions."""
    entity_manager: EntityManager

    def compute_impact(self, obj1_idx: int, obj2_idx: int, sample_idx: float) -> Dict[str, Any]:
        """
        Compute Hertzian impact parameters for an impact event.
        
        Parameters:
        -----------
        obj1_idx, obj2_idx : int
            Object indices
        sample_idx : float
            Sample index at impact
            
        Returns:
        --------
        Dict containing:
            - effective_radius: Effective radius of curvature (m)
            - effective_modulus: Effective Young's modulus (Pa)
            - duration: Impact duration (s)
            - contact_radius: Maximum contact radius (m)
            - max_force: Maximum contact force (N)
            - mass1, mass2: Object masses (kg)
            - penetration_depth: Maximum penetration depth (m)
            - contact_pressure: Maximum contact pressure (Pa)
            - coupling_strength: Normalized coupling strength (0-1)
        """
        # Get object configurations and trajectories
        config = self.entity_manager.get('config')
        
        for obj in config.objects:
            if obj.idx == obj1_idx:
                config_obj1 = obj
            if obj.idx == obj2_idx:
                config_obj2 = obj

        trajectories = self.entity_manager.get('trajectories')
        for traj in trajectories.values():
            if hasattr(traj, 'obj_idx'):
                if traj.obj_idx == obj1_idx:
                    trajectory1 = traj
                if traj.obj_idx == obj2_idx:
                    trajectory2 = traj
        
        # Get material properties
        E1 = config_obj1.acoustic_shader.young_modulus
        E2 = config_obj2.acoustic_shader.young_modulus
        nu1 = config_obj1.acoustic_shader.poisson_ratio
        nu2 = config_obj2.acoustic_shader.poisson_ratio
        density1 = config_obj1.acoustic_shader.density
        density2 = config_obj2.acoustic_shader.density
        
        # Get velocities at impact
        v1 = trajectory1.get_velocity(sample_idx)
        v2 = trajectory2.get_velocity(sample_idx)
        relative_velocity = np.linalg.norm(v1 - v2)
#        relative_velocity = abs(np.linalg.norm(v1 - v2))
#        relative_velocity = 1e-6 if 0 < relative_velocity < 1e-6 else relative_velocity
        
        # Get geometries at impact
        vertices1 = trajectory1.get_vertices(sample_idx)
        vertices2 = trajectory2.get_vertices(sample_idx)
        normals1 = trajectory1.get_normals(sample_idx)
        normals2 = trajectory2.get_normals(sample_idx)
        faces1 = trajectory1.get_faces()
        faces2 = trajectory2.get_faces()
        
        # Compute effective radius using convex hull approximation
        R1 = self._compute_effective_radius(vertices1)
        R2 = self._compute_effective_radius(vertices2)

        # Effective radius for two spheres in contact
        if R1 is not None and R2 is not None:
            R_eff = (R1 * R2) / (R1 + R2)
        elif R1 is not None:
            R_eff = R1  # Object 2 is flat
        elif R2 is not None:
            R_eff = R2  # Object 1 is flat
        else:
            R_eff = 0.01  # Default small radius
        
        # Effective Young's modulus
        E_star = 1 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2) if E1 and E2 and nu1 and nu2 else 1e9
        
        # Compute masses from volumes and densities
        mesh1 = trimesh.Trimesh(vertices=vertices1, vertex_normals=normals1, faces=faces1)
        mesh2 = trimesh.Trimesh(vertices=vertices2, vertex_normals=normals2, faces=faces2)
        mesh1.density = density1
        mesh2.density = density2
        mass1 = mesh1.mass
        mass2 = mesh2.mass
        # verify and limit too small masses to 0.1gr
#        if mesh1.is_volume:
#            mass1 = mesh1.mass if mesh1.mass < 9e-5 else 0.0001
#        else:
#            print(f"Warning: HertzianContact {obj1_idx} is not totally watertight. Using convex hull mass")
#            mass1 = mesh1.convex_hull.mass if mesh1.mass < 9e-5 else 0.0001
#        if mesh2.is_volume:
#            mass2 = mesh2.mass if mesh1.mass < 9e-5 else 0.0001
#        else:
#            print(f"Warning: HertzianContact {obj2_idx} is not totally watertight. Using convex hull mass")
#            mass2 = mesh2.convex_hull.mass if mesh1.mass < 9e-5 else 0.0001
        
        # Reduced mass
        reduced_mass = (mass1 * mass2) / (mass1 + mass2)
        
        # Hertzian impact parameters
        # Maximum penetration depth (δ_max)
#        print(sample_idx, obj1_idx, obj2_idx, reduced_mass, relative_velocity, E_star, np.sqrt(R_eff))
        delta_max = ( (5 * reduced_mass * relative_velocity**2) / (4 * E_star * np.sqrt(R_eff)) )**(2/5)
        
        # Maximum contact radius
        a_max = np.sqrt(R_eff * delta_max)
        
        # Maximum contact force
        F_max = (4/3) * E_star * np.sqrt(R_eff) * delta_max**(3/2)
        
        # Impact duration (τ)
        # From Hertz theory: τ ≈ 2.94 * δ_max / v_impact
        if relative_velocity > 0:
            duration = 2.94 * delta_max / relative_velocity
        else:
            duration = 0.01  # Default short duration

        # Get contact normal
        contact_normal = self._get_contact_normal(trajectory1, trajectory2, sample_idx)
        
        # Find vertices near contact
        contact_point = self._estimate_contact_point(vertices1, vertices2, contact_normal)
        
        # Maximum contact pressure (Hertz pressure)
        p_max = (3 * F_max) / (2 * np.pi * a_max**2)

        # Normalized coupling strength
        # Based on force magnitude, contact area, and material properties
        coupling_strength = self._compute_coupling_strength(config_obj1, config_obj2, F_max, a_max, relative_velocity, is_continuous=False, contact_type=ContactType.IMPACT)

        return {
            'effective_radius': R_eff,
            'effective_modulus': E_star,
            'duration': duration,
            'contact_point': contact_point,
            'contact_radius': a_max,
            'max_force': F_max,
            'mass1': mass1,
            'mass2': mass2,
            'penetration_depth': delta_max,
            'contact_pressure': p_max,
            'coupling_strength': coupling_strength,
            'contact_type': ContactType.IMPACT
        }

    def _compute_effective_radius(self, vertices: np.ndarray) -> Optional[float]:
        """
        Compute effective radius of curvature using convex hull approximation.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex positions (N, 3)
            
        Returns:
        --------
        Effective radius (m) or None if cannot compute
        """
        if len(vertices) < 4:
            return None
        
        # Compute convex hull
        hull = ConvexHull(vertices)
            
        # Get hull vertices
        hull_vertices = vertices[hull.vertices]
            
        # Compute centroid
        centroid = np.mean(hull_vertices, axis=0)
            
        # Compute distances from centroid to hull vertices
        distances = np.linalg.norm(hull_vertices - centroid, axis=1)
            
        # Effective radius as average distance
        R_eff = np.mean(distances)
            
        # Alternative: Use bounding sphere approximation
        # max_distance = np.max(distances)
        # R_eff = max_distance
            
        return float(R_eff)

    def compute_continuous_contact(self, force_data: Any, obj1_idx: int, obj2_idx: int, sample_idx: float, frame_range: int) -> Dict[str, Any]:
        """
        Compute Hertzian parameters for continuous contact (scraping, sliding, rolling).
        
        Parameters:
        -----------
        obj1_idx, obj2_idx : int
            Object indices
        sample_idx : float
            Starting sample index
        frame_range : int
            Duration of contact in samples
            
        Returns:
        --------
        Dict containing contact parameters and type classification
        """
        # Get object configurations and trajectories
        config = self.entity_manager.get('config')
        
        for obj in config.objects:
            if obj.idx == obj1_idx:
                config_obj1 = obj
            if obj.idx == obj2_idx:
                config_obj2 = obj
        
        trajectories = self.entity_manager.get('trajectories')
        for traj in trajectories.values():
            if hasattr(traj, 'obj_idx'):
                if traj.obj_idx == obj1_idx:
                    trajectory1 = traj
                if traj.obj_idx == obj2_idx:
                    trajectory2 = traj
        
        # Get material properties
        E1 = config_obj1.acoustic_shader.young_modulus
        E2 = config_obj2.acoustic_shader.young_modulus
        nu1 = config_obj1.acoustic_shader.poisson_ratio
        nu2 = config_obj2.acoustic_shader.poisson_ratio
        friction1 = config_obj1.acoustic_shader.friction
        friction2 = config_obj2.acoustic_shader.friction
        roughness1 = config_obj1.acoustic_shader.roughness
        roughness2 = config_obj2.acoustic_shader.roughness
        
        # Get velocities and forces
        v1 = trajectory1.get_velocity(sample_idx)
        v2 = trajectory2.get_velocity(sample_idx)
        omega1 = trajectory1.get_angular_velocity(sample_idx)
        omega2 = trajectory2.get_angular_velocity(sample_idx)
        
#        relative_velocity = abs(np.linalg.norm(v1 - v2))
#        relative_velocity = 1e-6 if 0 < relative_velocity < 1e-6 else relative_velocity
        relative_velocity = np.linalg.norm(v1 - v2)
        tangential_velocity = np.linalg.norm(v1 - v2 - np.dot(v1 - v2, self._get_contact_normal(trajectory1, trajectory2, sample_idx)) * self._get_contact_normal(trajectory1, trajectory2, sample_idx))
        
        # Get normal force from force data or estimate
        normal_force_mag = force_data.normal_force_magnitude
        tangential_force_mag = force_data.tangential_force_magnitude
        
        # Get geometries
        vertices1 = trajectory1.get_vertices(sample_idx)
        vertices2 = trajectory2.get_vertices(sample_idx)
        
        # Compute effective radius
        R1 = self._compute_effective_radius(vertices1)
        R2 = self._compute_effective_radius(vertices2)
        
        if R1 is not None and R2 is not None:
            R_eff = (R1 * R2) / (R1 + R2)
        elif R1 is not None:
            R_eff = R1
        elif R2 is not None:
            R_eff = R2
        else:
            R_eff = 0.01
        
        # Effective Young's modulus
        E_star = 1 / ((1 - nu1**2)/E1 + (1 - nu2**2)/E2) if E1 and E2 and nu1 and nu2 else 1e9
        
        # Contact radius for continuous contact
        a_contact = ( (3 * normal_force_mag * R_eff) / (4 * E_star) )**(1/3)
        
        # Penetration depth
        delta = a_contact**2 / R_eff
        
        # Contact pressure
        p_contact = 0
        if not a_contact == 0:
            p_contact = (3 * normal_force_mag) / (2 * np.pi * a_contact**2)
        
        # Get contact normal
        contact_normal = self._get_contact_normal(trajectory1, trajectory2, sample_idx)
        
        # Find vertices near contact
        contact_point = self._estimate_contact_point(vertices1, vertices2, contact_normal)
        
        # Rolling radius (for rolling contact)
        rolling_radius = self._compute_rolling_radius(vertices1, vertices2, trajectory1, trajectory2, contact_normal, contact_point, sample_idx)
        
        # Classify contact type
        contact_type = self._classify_contact_type(relative_velocity, tangential_velocity, normal_force_mag, tangential_force_mag, omega1, omega2, roughness1, roughness2, friction1, friction2, R1, R2)
        
        # Compute coupling strength for continuous contact
        coupling_strength = self._compute_coupling_strength(config_obj1, config_obj2, normal_force_mag, a_contact, relative_velocity, is_continuous=True, contact_type=contact_type)
        
        return {
            'effective_radius': R_eff,
            'effective_modulus': E_star,
            'contact_point': contact_point,
            'contact_radius': a_contact,
            'rolling_radius': rolling_radius,
            'normal_force': normal_force_mag,
            'tangential_force': tangential_force_mag,
            'penetration_depth': delta,
            'contact_pressure': p_contact,
            'contact_type': contact_type,
            'coupling_strength': coupling_strength,
            'relative_velocity': relative_velocity,
            'tangential_velocity': tangential_velocity
        }

    def _compute_rolling_radius(self, vertices1: np.ndarray, vertices2: np.ndarray, trajectory1: Any, trajectory2: Any, contact_normal: np.ndarray, contact_point: np.ndarray, sample_idx: float) -> float:
        """
        Compute effective rolling radius for rolling contact.
        
        Parameters:
        -----------
        vertices1, vertices2 : np.ndarray
            Vertex positions
        trajectory1, trajectory2 : Any
            Trajectory objects
        sample_idx : float
            Sample index
            
        Returns:
        --------
        Effective rolling radius (m)
        """
        # For rolling, we need the radius at the contact point
        # This is a simplified approximation
        
        # For object 1: distance from center to contact point along contact normal
        center1 = np.mean(vertices1, axis=0)
        R1_rolling = np.abs(np.dot(contact_point - center1, contact_normal))
        
        # For object 2: distance from center to contact point along contact normal
        center2 = np.mean(vertices2, axis=0)
        R2_rolling = np.abs(np.dot(contact_point - center2, -contact_normal))
        
        # Effective rolling radius
        if R1_rolling > 0 and R2_rolling > 0:
            R_rolling = (R1_rolling * R2_rolling) / (R1_rolling + R2_rolling)
        else:
            R_rolling = max(R1_rolling, R2_rolling)
        
        return float(R_rolling)
    
    def _get_contact_normal(self, trajectory1: Any, trajectory2: Any, sample_idx: float) -> np.ndarray:
        """
        Estimate contact normal between two objects.
        
        Parameters:
        -----------
        trajectory1, trajectory2 : Any
            Trajectory objects
        sample_idx : float
            Sample index
            
        Returns:
        --------
        Contact normal vector (unit length)
        """
        # Get positions
        pos1 = trajectory1.get_position(sample_idx)
        pos2 = trajectory2.get_position(sample_idx)
        
        # Direction from object 1 to object 2
        direction = pos2 - pos1
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        else:
            return np.array([0, 1, 0])  # Default upward normal

    def _estimate_contact_point(self, vertices1: np.ndarray, vertices2: np.ndarray, contact_normal: np.ndarray) -> np.ndarray:
        """
        Estimate contact point between two objects.
        
        Parameters:
        -----------
        vertices1, vertices2 : np.ndarray
            Vertex positions
        contact_normal : np.ndarray
            Contact normal vector
            
        Returns:
        --------
        Estimated contact point (3,)
        """
        # Simple approximation: midpoint between closest points
        
        # Find vertex on object 1 most aligned with contact normal
        proj1 = vertices1 @ contact_normal
        idx1 = np.argmax(proj1)
        point1 = vertices1[idx1]
        
        # Find vertex on object 2 most aligned with opposite normal
        proj2 = vertices2 @ (-contact_normal)
        idx2 = np.argmax(proj2)
        point2 = vertices2[idx2]
        
        # Midpoint
        return (point1 + point2) / 2

    def _classify_contact_type(self, relative_velocity: float, tangential_velocity: float, normal_force: float, tangential_force: float, omega1: np.ndarray, omega2: np.ndarray, roughness1: float, roughness2: float, friction1: float, friction2: float, R1: float, R2: float) -> ContactType:
        """
        Enhanced contact type classification with better thresholds and physics-based logic.
        
        Parameters:
        -----------
        relative_velocity : float
            Magnitude of relative velocity (m/s)
        tangential_velocity : float
            Magnitude of tangential velocity (m/s)
        normal_force : float
            Normal force magnitude (N)
        tangential_force : float
            Tangential force magnitude (N)
        omega1, omega2 : np.ndarray
            Angular velocity vectors (rad/s)
        roughness1, roughness2 : float
            Surface roughness (Ra in meters)
        friction1, friction2 : float
            Static friction coefficients
        R1, R2 : float
            Effective radii of curvature (m)
            
        Returns:
        --------
        ContactType enum
        """
        # Constants
        VELOCITY_THRESHOLD = 0.001  # 1 mm/s threshold for "static"
        ANGULAR_VELOCITY_THRESHOLD = 0.01  # 0.01 rad/s threshold
        
        # Calculate derived parameters
        angular_speed1 = np.linalg.norm(omega1)
        angular_speed2 = np.linalg.norm(omega2)
        avg_angular_speed = (angular_speed1 + angular_speed2) / 2
        
        # Average material properties
        avg_friction = (friction1 + friction2) / 2 if friction1 and friction2 else 0.3
        avg_roughness = (roughness1 + roughness2) / 2 if roughness1 and roughness2 else 0.5
        
        # Check for static contact (no significant movement)
        if (relative_velocity < VELOCITY_THRESHOLD and 
            angular_speed1 < ANGULAR_VELOCITY_THRESHOLD and 
            angular_speed2 < ANGULAR_VELOCITY_THRESHOLD):
            return ContactType.STATIC
        
        # Calculate theoretical rolling velocities
        # For rolling without slip: v = ω × R
        rolling_velocity1 = R1 * angular_speed1 if R1 else 0
        rolling_velocity2 = R2 * angular_speed2 if R2 else 0
        
        # Check if either object is rolling
        is_rolling1 = (R1 > 0 and angular_speed1 > ANGULAR_VELOCITY_THRESHOLD and 
                       abs(rolling_velocity1 - tangential_velocity) / max(rolling_velocity1, 0.001) < 0.2)
        is_rolling2 = (R2 > 0 and angular_speed2 > ANGULAR_VELOCITY_THRESHOLD and 
                       abs(rolling_velocity2 - tangential_velocity) / max(rolling_velocity2, 0.001) < 0.2)
        
        # Pure rolling condition
        if (is_rolling1 or is_rolling2) and tangential_velocity > VELOCITY_THRESHOLD:
            # Check if it it's pure rolling (minimal slip)
            slip_ratio = 0
            if is_rolling1 and rolling_velocity1 > 0:
                slip_ratio = abs(tangential_velocity - rolling_velocity1) / rolling_velocity1
            elif is_rolling2 and rolling_velocity2 > 0:
                slip_ratio = abs(tangential_velocity - rolling_velocity2) / rolling_velocity2
            
            if slip_ratio < 0.1:  # Less than 10% slip
                return ContactType.ROLLING
        
        # Check for sliding vs scraping
        if tangential_velocity > VELOCITY_THRESHOLD:
            # Calculate friction utilization
            max_friction_force = avg_friction * normal_force if normal_force > 0 else 0
            friction_utilization = (tangential_force / max_friction_force 
                                   if max_friction_force > 0 else 1.0)
            
            # Determine if it's scraping based on multiple factors
            scraping_score = 0.0
            
            # Factor 1: High friction utilization
            if friction_utilization > 0.7:
                scraping_score += 0.4
            elif friction_utilization > 0.4:
                scraping_score += 0.2
            
            # Factor 2: High roughness
            if avg_roughness > 0.0001:  # 0.1 mm roughness
                roughness_factor = min(avg_roughness * 10000, 1.0)  # Scale to 0-1
                scraping_score += 0.3 * roughness_factor
            
            # Factor 3: High velocity with low angular speed (pure sliding)
            if tangential_velocity > 0.1 and avg_angular_speed < 0.1:
                scraping_score += 0.3
            
            # Factor 4: Intermittent contact (check if this is available from context)
            # This would require additional data about contact stability
            
            if scraping_score >= 0.6:
                return ContactType.SCRAPING
            else:
                return ContactType.SLIDING
        
        # Mixed contact conditions
        # Condition 1: Some rolling with some sliding/scraping
        if ((is_rolling1 or is_rolling2) and 
            tangential_velocity > VELOCITY_THRESHOLD and
            (angular_speed1 > ANGULAR_VELOCITY_THRESHOLD or 
             angular_speed2 > ANGULAR_VELOCITY_THRESHOLD)):
            
            # Calculate slip percentage
            if is_rolling1 and rolling_velocity1 > 0:
                slip_percentage = abs(tangential_velocity - rolling_velocity1) / rolling_velocity1
            elif is_rolling2 and rolling_velocity2 > 0:
                slip_percentage = abs(tangential_velocity - rolling_velocity2) / rolling_velocity2
            else:
                slip_percentage = 1.0
            
            # If significant slip (10-90%), it's mixed
            if 0.1 <= slip_percentage <= 0.9:
                return ContactType.MIXED
        
        # Condition 2: Static with intermittent micro-movements
        if (relative_velocity < VELOCITY_THRESHOLD * 10 and  # Very slow
            (angular_speed1 > 0 or angular_speed2 > 0) and  # Some rotation
            tangential_force > 0):  # Some tangential force
            return ContactType.MIXED
        
        # Default to sliding if there's tangential motion but no other classification fits
        if tangential_velocity > VELOCITY_THRESHOLD:
            return ContactType.SLIDING
        
        # If we get here and there's some motion but not classified, default to mixed
        if relative_velocity > VELOCITY_THRESHOLD or avg_angular_speed > ANGULAR_VELOCITY_THRESHOLD:
            return ContactType.MIXED
        
        return ContactType.STATIC
    
    
    def get_mixed_factor(self, relative_velocity: float, tangential_velocity: float, normal_force: float, tangential_force: float, omega1: np.ndarray, omega2: np.ndarray, roughness1: float, roughness2: float, friction1: float, friction2: float, vertices1: np.ndarray, vertices2: np.ndarray) -> Dict[str, float]:
        """
        Compute detailed mixed contact factors for simultaneous contact types.
        
        Returns normalized factors (0-1) indicating the proportion of each contact type
        occurring simultaneously.
        
        Returns:
        --------
        Dict containing:
            - rolling_factor: Proportion of pure rolling (0-1)
            - sliding_factor: Proportion of sliding (0-1)
            - scraping_factor: Proportion of scraping (0-1)
            - static_factor: Proportion of static contact (0-1)
            - impact_factor: Proportion of impact (0-1, usually 0 for continuous contact)
        """
        # Calculate effective radii
        R1 = self._compute_effective_radius(vertices1)
        R2 = self._compute_effective_radius(vertices2)
        
        # Initialize factors
        factors = {
            'rolling_factor': 0.0,
            'sliding_factor': 0.0,
            'scraping_factor': 0.0,
            'static_factor': 0.0,
            'impact_factor': 0.0
        }
        
        # Constants
        VELOCITY_THRESHOLD = 0.001
        ANGULAR_VELOCITY_THRESHOLD = 0.01
        
        # Calculate angular speeds
        angular_speed1 = np.linalg.norm(omega1)
        angular_speed2 = np.linalg.norm(omega2)
        
        # Average properties
        avg_friction = (friction1 + friction2) / 2 if friction1 and friction2 else 0.3
        avg_roughness = (roughness1 + roughness2) / 2 if roughness1 and roughness2 else 0.5
        
        # Calculate theoretical rolling velocities
        rolling_velocity1 = R1 * angular_speed1 if R1 else 0
        rolling_velocity2 = R2 * angular_speed2 if R2 else 0
        max_rolling_velocity = max(rolling_velocity1, rolling_velocity2)
        
        # 1. Static factor (no movement)
        if (relative_velocity < VELOCITY_THRESHOLD and 
            angular_speed1 < ANGULAR_VELOCITY_THRESHOLD and 
            angular_speed2 < ANGULAR_VELOCITY_THRESHOLD):
            factors['static_factor'] = 1.0
            return factors
        
        # 2. Rolling factor calculation
        if max_rolling_velocity > 0 and tangential_velocity > 0:
            # Calculate slip ratio
            if rolling_velocity1 > 0:
                slip_ratio1 = abs(tangential_velocity - rolling_velocity1) / rolling_velocity1
            else:
                slip_ratio1 = 1.0
                
            if rolling_velocity2 > 0:
                slip_ratio2 = abs(tangential_velocity - rolling_velocity2) / rolling_velocity2
            else:
                slip_ratio2 = 1.0
                
            # Use minimum slip ratio (best rolling condition)
            min_slip_ratio = min(slip_ratio1, slip_ratio2)
            
            # Rolling factor is high when slip is low
            factors['rolling_factor'] = max(0.0, 1.0 - min_slip_ratio)
        
        # 3. Sliding/Scraping factors
        if tangential_velocity > VELOCITY_THRESHOLD:
            # Calculate friction utilization
            max_friction_force = avg_friction * normal_force if normal_force > 0 else 0
            friction_utilization = (tangential_force / max max_friction_force 
                                   if max_friction_force > 0 else 1.0)
            
            # Base sliding factor (inversely related to rolling)
            base_sliding = 1.0 - factors['rolling_factor']
            
            # Adjust for scraping based on multiple criteria
            scraping_probability = 0.0
            
            # Criterion 1: High friction utilization
            if friction_utilization > 0.8:
                scraping_probability += 0.5
            elif friction_utilization > 0.5:
                scraping_probability += 0.3
            
            # Criterion 2: Surface roughness
            if avg_roughness > 0.00005:  # 50 micron roughness
                roughness_contribution = min(avg_roughness * 20000, 1.0)  # Scale
                scraping_probability += 0.3 * roughness_contribution
            
            # Criterion 3: Velocity (scraping often at moderate speeds)
            if 0.01 < tangential_velocity < 1.0:  # 1 cm/s to 1 m/s
                speed_factor = min(tangential_velocity, 1.0)
                scraping_probability += 0.2 * speed_factor
            
            # Criterion 4: Intermittency (simplified - could be enhanced with more data)
            # Assuming some vibration or stick-slip behavior increases scraping likelihood
            if angular_speed1 > 0 or angular_speed2 > 0:
                scraping_probability += 0.1
            
            # Cap scraping probability
            scraping_probability = min(scraping_probability, 1.0)
            
            # Distribute between sliding and scraping
            factors['scraping_factor'] = base_sliding * scraping_probability
            factors['sliding_factor'] = base_sliding * (1.0 - scraping_probability)
        
        # 4. Static factor for very slow movements
        if relative_velocity < VELOCITY_THRESHOLD * 10:
            # Inverse relationship with velocity
            static_component = 1.0 - min(relative_velocity / (VELOCITY_THRESHOLD * 10), 1.0)
            factors['static_factor'] = static_component * 0.3  # Scale down
        
        # 5. Normalize factors to sum to 1.0
        total = sum(factors.values())
        if total > 0:
            for key in factors:
                factors[key] /= total
        
        # 6. Special case: If rolling is dominant but there's also significant
        # tangential force, redistribute some to sliding/scraping
        if (factors['rolling_factor'] > 0.7 and 
            tangential_force > 0 and 
            friction_utilization > 0.3):
            
            # Reduce rolling, increase sliding/scraping
            adjustment = min(factors['rolling_factor'] * 0.3, 0.2)
            factors['rolling_factor'] -= adjustment
            
            # Distribute adjustment between sliding and scraping based on roughness
            if avg_roughness > 0.0001:
                factors['scraping_factor'] += adjustment * 0.7
                factors['sliding_factor'] += adjustment * 0.3
            else:
                factors['sliding_factor'] += adjustment
        
        # Final normalization
        total = sum(factors.values())
        if total > 0:
            for key in factors:
                factors[key] /= total
        
        return factors

    def _compute_coupling_strength(self, config_obj1: Any, config_obj2: Any, force: float, contact_radius: float, velocity: float, is_continuous: bool = False, contact_type: ContactType = None) -> float:
        """
        Compute normalized coupling strength (0-1).
        
        Parameters:
        -----------
        config_obj1, config_obj2: ObjectConfig
            obj configuration class
        force : float
            Contact force (N)
        contact_radius : float
            Contact radius (m)
        velocity : float
            Relative velocity (m/s)
        is_continuous : bool
            Whether this is continuous contact
        contact_type : ContactType
            Type of contact
            
        Returns:
        --------
        Normalized coupling strength (0-1)
        """
        # Young's moduli (Pa)
        E1 = config_obj1.acoustic_shader.young_modulus
        E2 = config_obj2.acoustic_shader.young_modulus

        # Density (Kg/m^3)
        density1 = config_obj1.acoustic_shader.density
        density2 = config_obj2.acoustic_shader.density

        # Rayleigh damping ratio (no unit)
        damping1 = config_obj1.acoustic_shader.damping
        damping2 = config_obj2.acoustic_shader.damping

        # Base coupling from force and contact area
        contact_area = np.pi * contact_radius**2

        # Impedance matching factor
        # Acoustic impedance Z = ρ * c, where c = sqrt(E/ρ) for longitudinal waves
        c1 = np.sqrt(E1 / density1)
        c2 = np.sqrt(E2 / density2)
        Z1 = density1 * c1
        Z2 = density2 * c2
        Z_min = min(Z1, Z2)
        Z_max = max(Z1, Z2)
        impedance_match = 2 * Z_min / (Z_max + Z_min) if (Z_max + Z_min) > 0 else 0.0
        
        # Force transmission factor (pressure)
        force_transmission = 0.5
        E_avg = (E1 + E2) / 2
        if E_avg > 0 and contact_area > 0:
            stress = force / contact_area
            stress_ratio = stress / E_avg
            # Sigmoid mapping for stress ratio
            force_transmission = 1.0 / (1.0 + np.exp(-100.0 * (stress_ratio - 0.01)))
        
        # Material stiffness factor
        stiffness_factor = min(E1, E2) / max(E1, E2)
        
        # Damping factor
        damping_avg = (damping1 + damping2) / 2
        damping_factor = min(1.0, damping_avg * 20.0)  # Scale factor

        # Velocity factor
        velocity_factor = min(velocity / 10.0, 1.0)  # Normalize by 10 m/s
        
        # Contact type factor
        if contact_type:
            if contact_type == ContactType.IMPACT:
                type_factor = 1.0
            elif contact_type == ContactType.SCRAPING:
                type_factor = 0.8
            elif contact_type == ContactType.SLIDING:
                type_factor = 0.6
            elif contact_type == ContactType.MIXED: ##################################### <- To Be Reviewd (more rolling or (scraping or sliding or static)?)
                type_factor = 0.5
            elif contact_type == ContactType.ROLLING:
                type_factor = 0.4
            elif contact_type == ContactType.STATIC:
                type_factor = 0.2
            else:
                type_factor = 0.5
        else:
            type_factor = 0.7 if is_continuous else 1.0
        
        # Combine factors
        coupling = (0.10 * force_transmission +
                   0.20 * impedance_match +
                   0.15 * damping_factor +
                   0.3 * stiffness_factor +
                   0.2 * velocity_factor +
                   0.1 * type_factor)
        
        # Normalize to 0-1 range
        coupling = max(0.0, min(1.0, coupling))
        
        return float(coupling)
