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
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import trimesh

class ImpactForceAnalyzer:
    def __init__(self, fps, young_modulus, poisson_ratio, density, alpha_damping, beta_damping):
        """
        Initialize the impact force analyzer.
        
        Parameters:
        - fps: frames per second
        - young_modulus: Young's modulus in N/m²
        - poisson_ratio: Poisson's ratio
        - density: density in kg/m³
        - alpha_damping, beta_damping: Rayleigh damping coefficients
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.alpha_damping = alpha_damping
        self.beta_damping = beta_damping
        
        # Calculate derived material properties
        self.shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
        self.bulk_modulus = young_modulus / (3 * (1 - 2 * poisson_ratio))
        
        # Storage for analysis results
        self.impact_data = {}
        
    def load_obj_sequence(self, directory):
        """
        Load a sequence of OBJ files from a directory.
        
        Returns:
        - vertices_list: list of vertex arrays per frame
        - faces_list: list of face arrays per frame
        - normals_list: list of normal arrays per frame
        - timestamps: array of timestamps
        """
        obj_files = sorted([f for f in os.listdir(directory) if f.endswith('.obj')])
        
        vertices_list = []
        faces_list = []
        normals_list = []
        timestamps = []
        
        for i, obj_file in enumerate(obj_files):
            filepath = os.path.join(directory, obj_file)
            mesh = trimesh.load_mesh(filepath)
            
            vertices_list.append(mesh.vertices)
            faces_list.append(mesh.faces)
            normals_list.append(mesh.vertex_normals)
            timestamps.append(i * self.dt)
            
        return vertices_list, faces_list, normals_list, np.array(timestamps)
    
    def find_impact_location(self, vertices_list_1, vertices_list_2, threshold=0.01):
        """
        Find the impact location by detecting when objects get close.
        
        Parameters:
        - vertices_list_1, vertices_list_2: vertex sequences for both objects
        - threshold: distance threshold for impact detection in meters
        
        Returns:
        - impact_time: time of impact
        - impact_location: world coordinates of impact
        - vertex_id_1, vertex_id_2: IDs of closest vertices
        """
        min_distances = []
        closest_pairs = []
        
        for frame_idx in range(min(len(vertices_list_1), len(vertices_list_2))):
            vertices1 = vertices_list_1[frame_idx]
            vertices2 = vertices_list_2[frame_idx]
            
            # Use KDTree for efficient nearest neighbor search
            tree1 = KDTree(vertices1)
            tree2 = KDTree(vertices2)
            
            # Find closest points between objects
            distances, indices = tree1.query(vertices2)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            min_distances.append(min_distance)
            closest_pairs.append((indices[min_idx_idx], min_idx, vertices2[min_idx]))
            
            # Check if impact occurred
            if min_distance <= threshold:
                impact_frame = frame_idx
                vertex_id_1 = indices[min_idx]
                vertex_id_2 = min_idx
                impact_location = vertices2[min_idx]
                
                # Interpolate to get more precise impact time
                if frame_idx > 0:
                    prev_dist = min_distances[frame_idx - 1]
                    curr_dist = min_distance
                    if prev_dist > threshold:
                        # Linear interpolation for impact time
                        t_frac = (threshold - prev_dist) / (curr_dist - prev_dist)
                        impact_time = (frame_idx - 1 + t_frac) * self.dt
                    else:
                        impact_time = frame_idx * self.dt
                else:
                    impact_time = frame_idx * self.dt
                
                return impact_time, impact_location, vertex_id_1, vertex_id_2
        
        # If no impact found, return the closest approach
        min_frame = np.argmin(min_distances)
        vertex_id_1, vertex_id_2, impact_location = closest_pairs[min_frame]
        impact_time = min_frame * self.dt
        
        print(f"Warning: No impact detected below threshold. Closest approach: {min(min_distances):.4f}m")
        return impact_time, impact_location, vertex_id_1, vertex_id_2
    
    def interpolate_vertex_motion(self, vertices_list, vertex_id, timestamps):
        """
        Interpolate the motion of a specific vertex over time.
        
        Returns:
        - position_interp: interpolation function for position
        - velocity_interp: interpolation function for velocity
        - acceleration_interp: interpolation function for acceleration
        """
        # Extract vertex positions over time
        vertex_positions = np.array([vertices[vertex_id] for vertices in vertices_list])
        
        # Create interpolation functions
        position_interp = interp1d(timestamps, vertex_positions, axis=0, 
                                 kind='cubic', fill_value='extrapolate')
        
        # Calculate velocity and acceleration
        velocity = np.gradient(vertex_positions, timestamps, axis=0)
        acceleration = np.gradient(velocity, timestamps, axis=0)
        
        velocity_interp = interp1d(timestamps, velocity, axis=0, 
                                 kind='cubic', fill_value='extrapolate')
        acceleration_interp = interp1d(timestamps, acceleration, axis=0, 
                                     kind='cubic', fill_value='extrapolate')
        
        return position_interp, velocity_interp, acceleration_interp
    
    def calculate_contact_force(self, vertices_list_1, vertices_list_2, 
                              vertex_id_1, vertex_id_2, timestamps, 
                              contact_area_estimate=0.001):
        """
        Calculate contact force based on material properties and motion.
        
        Parameters:
        - contact_area_estimate: estimated contact area in m²
        """
        # Interpolate motion of contact vertices
        pos_interp_1, vel_interp_1, acc_interp_1 = self.interpolate_vertex_motion(
            vertices_list_1, vertex_id_1, timestamps)
        pos_interp_2, vel_interp_2, acc_interp_2 = self.interpolate_vertex_motion(
            vertices_list_2, vertex_id_2, timestamps)
        
        # Calculate relative motion at contact point
        time_fine = np.linspace(timestamps[0], timestamps[-1], len(timestamps) * 10)
        
        rel_position = pos_interp_1(time_fine) - pos_interp_2(time_fine)
        rel_velocity = vel_interp_1(time_fine) - vel_interp_2(time_fine)
        rel_acceleration = acc_interp_1(time_fine) - acc_interp_2(time_fine)
        
        # Calculate penetration depth (magnitude of relative position)
        penetration = np.linalg.norm(rel_position, axis=1)
        
        # Calculate contact force using Hertzian contact theory
        # Simplified model - you might want to use a more sophisticated contact model
        effective_young = 1 / ((1 - self.poisson_ratio**2) / self.young_modulus + 
                              (1 - self.poisson_ratio**2) / self.young_modulus)
        
        # Spring force (Hertzian contact)
        spring_force = effective_young * np.sqrt(contact_area_estimate) * penetration
        
        # Damping force (Rayleigh damping)
        damping_force = (self.alpha_damping * self.density * contact_area_estimate * 
                        np.linalg.norm(rel_velocity, axis=1) +
                        self.beta_damping * effective_young * contact_area_estimate * 
                        np.linalg.norm(rel_velocity, axis=1))
        
        # Total contact force
        total_force = spring_force + damping_force
        
        # Create force interpolation function
        force_interp = interp1d(time_fine, total_force, kind='cubic', 
                              fill_value=0.0, bounds_error=False)
        
        return force_interp, time_fine, total_force
    
    def analyze_impact(self, dir_object_1, dir_object_2, impact_threshold=0.01):
        """
        Complete impact analysis between two objects.
        """
        print(f"Analyzing impact between {dir_object_1} and {dir_object_2}")
        
        # Load OBJ sequences
        vertices_1, faces_1, normals_1, timestamps_1 = self.load_obj_sequence(dir_object_1)
        vertices_2, faces_2, normals_2, timestamps_2 = self.load_obj_sequence(dir_object_2)
        
        # Find impact location and time
        impact_time, impact_location, vertex_id_1, vertex_id_2 = self.find_impact_location(
            vertices_1, vertices_2, impact_threshold)
        
        # Calculate contact force
        timestamps = timestamps_1[:min(len(vertices_1), len(vertices_2))]
        force_interp, force_time, force_magnitude = self.calculate_contact_force(
            vertices_1, vertices_2, vertex_id_1, vertex_id_2, timestamps)
        
        # Store results
        impact_id = f"{os.path.basename(dir_object_1)}_{os.path.basename(dir_object_2)}"
        self.impact_data[impact_id] = {
            'impact_time': impact_time,
            'impact_location': impact_location,
            'vertex_id_1': vertex_id_1,
            'vertex_id_2': vertex_id_2,
            'force_interpolator': force_interp,
            'force_time_series': force_time,
            'force_magnitude_series': force_magnitude,
            'max_force': np.max(force_magnitude),
            'impulse': np.trapz(force_magnitude, force_time)
        }
        
        return self.impact_data[impact_id]
    
    def get_impact_force_at_time(self, impact_id, time):
        """Get impact force at specific time for a given impact."""
        if impact_id in self.impact_data:
            return self.impact_data[impact_id]['force_interpolator'](time)
        else:
            raise ValueError(f"Impact ID {impact_id} not found")
    
    def generate_report(self, impact_id):
        """Generate a report for a specific impact analysis."""
        if impact_id not in self.impact_data:
            raise ValueError(f"Impact ID {impact_id} not found")
        
        data = self.impact_data[impact_id]
        
        report = f"""
        IMPACT ANALYSIS REPORT: {impact_id}
        =================================
        Impact Time: {data['impact_time']:.4f} seconds
        Impact Location: {data['impact_location']}
        Closest Vertex IDs: Object 1: {data['vertex_id_1']}, Object 2: {data['vertex_id_2']}
        Maximum Force: {data['max_force']:.2f} N
        Total Impulse: {data['impulse']:.4f} N·s
        =================================
        """
        return report

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with material properties
    analyzer = ImpactForceAnalyzer(
        fps=1000,  # 1000 frames per second
        young_modulus=2e9,  # 2 GPa
        poisson_ratio=0.3,
        density=1200,  # kg/m³
        alpha_damping=0.1,
        beta_damping=0.01
    )
    
    # Analyze impact between two objects
    try:
        impact_results = analyzer.analyze_impact("dir_1", "dir_2", impact_threshold=0.005)
        
        # Generate report
        print(analyzer.generate_report("dir_1_dir_2"))
        
        # Get force at specific time
        force_at_0_1s = analyzer.get_impact_force_at_time("dir_1_dir_2", 0.1)
        print(f"Force at 0.1s: {force_at_0_1s:.2f} N")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

