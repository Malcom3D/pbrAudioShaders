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
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import minimize
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..utils.config import Config, ObjectConfig

from ..lib.functions import _load_pose

@dataclass
class RotationSolver:
    entity_manager: EntityManager
    vertices_local: trimesh.caching.TrackedArray = None
    inertia_tensor: np.ndarray = None
    mass: np.float64 = None
    coefficient_of_restitution: float = None
    friction_coefficient: float = None

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        tmp_trajectories = self.entity_manager.get('trajectories')
        for index in tmp_trajectories:
            tmp_trajectory = tmp_trajectories[index]
            if hasattr(tmp_trajectory, 'add_data') and tmp_trajectory.obj_idx == obj_idx:
                frame = tmp_trajectory.frame
                impact_position = tmp_trajectory.position
                for config_obj in config.objects:
                    if config_obj.idx == obj_idx:
                            positions, rotations, landmarks_vertices = _load_pose(config_obj)
                            rotation = self._impact_rotation(config_obj, positions, rotations, impact_position, frame, sfps)
                            tmp_trajectory.add_data('rotation', rotation)

    def _load_obj(self, obj_idx: int, frame_idx: int):
        config = self.entity_manager.get('config')
        # Load mesh
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                if config_obj.static == True:
                    for filename in os.listdir(config_obj.obj_path):
                        if filename.endswith('.obj'):
                            obj_file = f"{config_obj.obj_path}/{filename}"
                            return trimesh.load_mesh(obj_file)
                elif config_obj.static == False:
                    items = os.listdir(config_obj.obj_path)
                    obj_filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
                    obj_file = os.path.join(config_obj.obj_path, obj_filenames[frame_idx])
                    if not os.path.exists(obj_file):
                        raise FileNotFoundError(f"OBJ file not found for {obj_name}: {obj_file}")
                    return trimesh.load_mesh(obj_file)

    def _impact_rotation(self, config_obj: ObjectConfig, positions: np.ndarray, rotations: np.ndarray, impact_position: np.ndarray, frame: float, sfps: int, max_iterations: int = 100, tolerance: float = 1e-6) -> Rotation:
        """
        Physics-based estimation of rotation at impact moment.
    
        This function uses collision physics to estimate the rotation at the exact
        moment of impact, accounting for the collision response.
    
        Parameters:
        -----------
    
        Returns:
        --------
        Rotation object representing estimated rotation at impact
        """

        def objective_function(rot_params: np.ndarray, use_post_impact: bool = False) -> float:
            """
            Objective function for optimization.
            Minimizes the difference between predicted and actual post-impact state.
            """
            # Convert parameters to rotation
            rot_impact = Rotation.from_rotvec(rot_params)
        
            # 1. Integrate from pre-impact to impact
            time_to_impact = impact_time * dt
            rot_pre_to_impact = self._integrate_to_impact(pre_impact_rot, pre_impact_ang_vel, time_to_impact)
        
            # Error between integrated rotation and proposed proposed impact rotation
            error_consistency = Rotation.magnitude(rot_impact * rot_pre_to_impact.inv())
        
            if not use_post_impact or post_impact_rot is None:
                # If no post-impact data, just ensure consistency
                return error_consistency
        
            # 2. Compute collision response at impact
            # Estimate linear velocity at impact (simplified)
            if post_impact_pos is not None:
                lin_vel_impact = (post_impact_pos - pre_impact_pos) / dt
            else:
                lin_vel_impact = np.zeros(3)
        
            # Compute angular velocity change due to collision
            _, delta_ang_vel = self._compute_collision_response(rot_impact, impact_pos, pre_impact_ang_vel, lin_vel_impact)
        
            ang_vel_post_impact = pre_impact_ang_vel + delta_ang_vel
        
            # 3. Integrate from impact to post-impact
            time_from_impact = (1 - impact_time) * dt
            rot_impact_to_post = self._integrate_to_impact(rot_impact, ang_vel_post_impact, time_from_impact)
        
            # 4. Compare with actual post-impact rotation
            error_prediction = Rotation.magnitude(rot_impact_to_post * post_impact_rot.inv())
        
            # Combine errors (weighted)
            total_error = 0.3 * error_consistency + 0.7 * error_prediction
        
            return total_error

        # Time information
        dt = 1.0/sfps
    
        # Pre-impact data
        frame_before = math.floor(frame)
        pre_impact_pos = positions[frame_before]
        pre_impact_rot = Rotation.from_euler('XYZ', rotations[frame_before])

        pre_impact_vel = (np.linalg.norm(positions[frame_before -1] - positions[frame_before]))/sfps
        pre_impact_ang_vel = self._angular_velocity(Rotation.from_euler('XYZ', rotations[frame_before -1]), pre_impact_rot, dt)
    
        # Post-impact data (if available)
        frame_after = math.ceil(frame)
        post_impact_pos = positions[frame_after]
        post_impact_rot = Rotation.from_euler('XYZ', rotations[frame_before])

        post_impact_vel = (np.linalg.norm(positions[frame_after] - positions[frame_after +1]))/sfps
        post_impact_ang_vel = self._angular_velocity(post_impact_rot, Rotation.from_euler('XYZ', rotations[frame_after +1]), dt)
    
        # Impact data
        impact_time = frame - frame_before
        impact_pos = impact_position

        # Object properties
        mesh = self._load_obj(config_obj.idx, frame_before)
        faces = mesh.faces
        self.vertices_local = mesh.vertices - pre_impact_pos
        mesh = trimesh.Trimesh(vertices=self.vertices_local, faces=faces)
        volume = mesh.volume
        center_of_mass = mesh.center_mass
        self.inertia_tensor = mesh.moment_inertia
        self.mass = config_obj.acoustic_shader.density * volume

        # Material properties
        self.coefficient_of_restitution = abs(post_impact_vel)/abs(pre_impact_vel)

        self.friction_coefficient = config_obj.acoustic_shader.friction
        youngs_modulus = config_obj.acoustic_shader.young_modulus
        poissons__ratio = config_obj.acoustic_shader.poisson_ratio
        rayleigh_damping = config_obj.acoustic_shader.damping

        # Main estimation logic
        if post_impact_rot is not None:
            # Use optimization with post-impact data as reference
            use_post_impact = True
        
            # Initial guess: interpolate between pre and post impact
            if post_impact_rot is not None:
                # Simple interpolation considering impact time
                rots = Rotation.concatenate([pre_impact_rot, post_impact_rot])
                times = [0,1]
                slerp = Slerp(times, rots)
                rot_interp = slerp(impact_time)
                initial_guess = rot_interp.as_rotvec()
            else:
                # Just integrate from pre-impact
                rot_guess = self._integrate_to_impact(pre_impact_rot, pre_impact_ang_vel, impact_time * dt)
                initial_guess = rot_guess.as_rotvec()
        
            # Optimize
            result = minimize(
                lambda x: objective_function(x, use_post_impact=True),
                initial_guess,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * 3,  # Bound rotation vector components
                options={'maxiter': max_iterations, 'ftol': tolerance, 'disp': False}
            )
        
            optimal_rot = Rotation.from_rotvec(result.x)
        
        else:
            # No post-impact data available
            # Use simplified physics-based estimation
        
            # Integrate to impact time
            optimal_rot = self._integrate_to_impact(pre_impact_rot, pre_impact_ang_vel, impact_time * dt)
        
            # Apply estimated collision effect
            # (This is approximate without post-impact reference)
            if post_impact_pos is not None:
                # Estimate linear velocity
                lin_vel = (post_impact_pos - pre_impact_pos) / dt
            
                # Compute collision response
                _, delta_ang_vel = self._compute_collision_response(optimal_rot, impact_pos, pre_impact_ang_vel, lin_vel)
            
                # Adjust rotation slightly based on collision
                # This is heuristic - adjust the weight based on your needs
                collision_effect_weight = 0.5
                delta_rot_vec = delta_ang_vel * (impact_time * dt) * collision_effect_weight
                delta_rot = Rotation.from_rotvec(delta_rot_vec)
                optimal_rot = optimal_rot * delta_rot
    
        return optimal_rot.as_euler('xyz')

    def _compute_contact_geometry(self, rot: Rotation, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate contact point and normal based on object geometry and position.
        This assumes the object is colliding with a static surface (e.g., floor).
        
        Returns:
        --------
        contact_point : Estimated contact point in world coordinates (3,)
        contact_normal : Estimated contact normal (3,)
        penetration_depth : Estimated penetration depth
        """
        # Transform vertices to world coordinates
        R = rot.as_matrix()
        vertices_world = (R @ self.vertices_local.T).T + pos
        
        # Find lowest point (assuming collision with horizontal surface)
        # You can modify this based on your specific collision scenario
        lowest_idx = np.argmin(vertices_world[:, 1])  # Assuming y is up/down
        contact_point = vertices_world[lowest_idx]
        
        # Estimate contact normal from vertex normal or geometry
        # For simplicity, use vertical normal (adjust based on your scene)
        contact_normal = np.array([0.0, 1.0, 0.0])  # Assuming floor collision
        
        # Estimate penetration (assuming object is slightly below surface)
        ground_height = 0.0  # Adjust based on your scene
        penetration_depth = max(0.0, ground_height - contact_point[1])
        
        return contact_point, contact_normal, penetration_depth
    
    def _compute_collision_response(self, rot: Rotation, pos: np.ndarray, ang_vel: np.ndarray, lin_vel: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute collision response using impulse-based method.
        
        Returns:
        --------
        delta_lin_vel : Change in linear velocity (3,)
        delta_ang_vel : Change in angular velocity (3,)
        """
        if lin_vel is None:
            lin_vel = np.zeros(3)
        
        # Estimate contact geometry
        contact_point, contact_normal, penetration = self._compute_contact_geometry(rot, pos)
        
        if penetration <= 0:
            # No collision
            return np.zeros(3), np.zeros(3)
        
        # Contact point relative to center of mass in world coordinates
        R = rot.as_matrix()
        r = contact_point - pos
        
        # Pre-collision velocity at contact point
        v_contact = lin_vel + np.cross(ang_vel, r)
        
        # Normal component of velocity
        v_n = np.dot(v_contact, contact_normal)
        
        # Only process if approaching the surface
        if v_n >= -1e-10:
            return np.zeros(3), np.zeros(3)
        
        # Inertia tensor in world coordinates
        I_world = R @ self.inertia_tensor @ R.T
        I_inv_world = np.linalg.inv(I_world)
        
        # Compute effective mass matrix
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        
        # Effective mass matrix for contact point
        K = np.eye(3) / self.mass - r_cross @ I_inv_world @ r_cross
        K_inv = np.linalg.inv(K)
        
        # Normal impulse (using coefficient of restitution)
        j_n = -(1 + self.coefficient_of_restitution) * v_n / np.dot(contact_normal, K_inv @ contact_normal)
        
        # Tangential impulse (friction)
        v_t = v_contact - v_n * contact_normal
        v_t_mag = np.linalg.norm(v_t)
        
        if v_t_mag > 1e-10:
            tangent_dir = v_t / v_t_mag
            j_t_max = self.friction_coefficient * abs(j_n)
            
            # Compute maximum tangential impulse that can be applied
            j_t_possible = -v_t_mag / np.dot(tangent_dir, K_inv @ tangent_dir)
            j_t = np.clip(j_t_possible, -j_t_max, j_t_max)
            j_t_vec = j_t * tangent_dir
        else:
            j_t_vec = np.zeros(3)
        
        # Total impulse
        j_total = j_n * contact_normal + j_t_vec
        
        # Velocity changes
        delta_lin_vel = j_total / self.mass
        delta_ang_vel = I_inv_world @ np.cross(r, j_total)
        
        return delta_lin_vel, delta_ang_vel
    
    def _integrate_to_impact(self, start_rot: Rotation, start_ang_vel: np.ndarray, delta_time: float) -> Rotation:
        """Integrate rotation forward in time assuming constant angular velocity."""
        # Simple integration: R(t) = R0 * exp(ω * t)
        delta_rot_vec = start_ang_vel * delta_time
        delta_rot = Rotation.from_rotvec(delta_rot_vec)
        return start_rot * delta_rot
    
    def _angular_velocity(self, rot1: Rotation, rot2: Rotation, dt: float) -> np.ndarray:
        """
        Estimate angular velocity between two rotations.
        """
        # Relative rotation
        delta_rot = rot2 * rot1.inv()
    
        # Convert to rotation vector
        rotvec = delta_rot.as_rotvec()
    
        # Angular velocity = rotation vector / time
        return rotvec / dt
