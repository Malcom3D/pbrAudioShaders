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
from scipy.spatial import KDTree
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import math

from ..core.entity_manager import EntityManager
from ..lib.force_data import ForceData
from ..lib.collision_data import CollisionData, CollisionType

@dataclass
class ForceSolver:
    entity_manager: EntityManager

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        collision_margin = config.system.collision_margin
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        collisions, active_collisions = ([] for _ in range(2))
        for config_obj in config.objects:
            if config_obj.idx == obj_idx:
                if config_obj.static:
                    # exit: obj_idx are static
                    return
                collisions_data = self.entity_manager.get('collisions')
                trajectories = self.entity_manager.get('trajectories')
                for t_idx in trajectories.keys():
                    if 'TrajectoryData' in str(type(trajectories[t_idx])):
                        if trajectories[t_idx].obj_idx == config_obj.idx:
                            trajectory = trajectories[t_idx]
                            frames = trajectory.get_x()
                for c_idx in collisions_data.keys():
                    if collisions_data[c_idx].obj1_idx  == config_obj.idx or collisions_data[c_idx].obj2_idx == config_obj.idx:
                        active_collisions.append(collisions_data[c_idx])

                for idx in range(len(frames)):
                    collisions, other_obj_indices, other_trajectories, other_config_objs = ([] for _ in range(4))
                    frame = frames[idx]            
                    for c_idx in range(len(active_collisions)):
                        if (active_collisions[c_idx].frame <= frame <= active_collisions[c_idx].frame + active_collisions[c_idx].frame_range):
                            collisions.append(active_collisions[c_idx])
                            other_obj_indices.append(collisions.obj1_idx if not collisions.obj1_idx == obj_idx else collisions.obj2_idx)
                            for t_idx in range(len(trajectories)):
                                if 'TrajectoryData' in str(type(trajectories[t_idx])):
                                    if trajectories[t_idx].obj_idx in other_obj_indices:
                                        other_trajectories.append(trajectories[t_idx])
                                    for other_config in config.objects:
                                        if other_config.idx in other_obj_indices:
                                            other_config_objs.append(other_config)
                    if collisions == []: 
                        force_data = self._calculate_forces(frame=frame, obj_idx=obj_idx, config_obj=config_obj, trajectory=trajectory, sfps=sfps, sample_rate=sample_rate)
                    else:
                        force_data = self._calculate_collision_forces(frame=frame, obj_idx=obj_idx, other_obj_indices=other_obj_indices, config_obj=config_obj, other_config_objs=other_config_objs, trajectory=trajectory, other_trajectories=other_trajectories, collisions=collisions, sfps=sfps, sample_rate=sample_rate)

                    if not force_data == None:
                        for index in range(len(force_data)):
                            force_idx = len(self.entity_manager.get('forces')) + 1
                            self.entity_manager.register('forces', force_data[index], force_idx)

    def _calculate_forces(self, frame: float, obj_idx: int, config_obj: Any, trajectory: Any, sfps: int, sample_rate: int) -> Optional[List[ForceData]]:
        """
        Calculate forces and velocities for a specific object at frame time.
        Uses finite differences to compute velocities and accelerations.
        """
        # Get current and neighboring frames
        frame_idx = np.searchsorted(trajectory.get_x(), frame)
        frames = trajectory.get_x()
        
        if frame_idx >= len(frames) - 1 or frame_idx == 0:
            # Can't compute derivatives at boundaries
            return None
        
        # Time step
        dt = 1 / sfps
        
        # Get positions at current, previous, and next frames
        pos_current = trajectory.get_position(frame)
        pos_prev = trajectory.get_position(frames[frame_idx - 1])
        pos_next = trajectory.get_position(frames[frame_idx + 1])
        
        # Get rotations
        rot_current = trajectory.get_rotation(frame)
        rot_prev = trajectory.get_rotation(frames[frame_idx - 1])
        rot_next = trajectory.get_rotation(frames[frame_idx + 1])
        
        # Convert Euler angles to rotation matrices
        from scipy.spatial.transform import Rotation
        R_current = Rotation.from_euler('xyz', rot_current).as_matrix()
        R_prev = Rotation.from_euler('xyz', rot_prev).as_matrix()
        R_next = Rotation.from_euler('xyz', rot_next).as_matrix()
        
        # Compute linear velocity (central difference)
        linear_velocity = (pos_next - pos_prev) / (2 * dt)
        
        # Compute linear acceleration (second order central difference)
        linear_acceleration = (pos_next - 2 * pos_current + pos_prev) / (dt ** 2)
        
        # Compute angular velocity
        # Using rotation vector approach
        delta_rot_prev = Rotation.from_matrix(R_prev.T @ R_current).as_rotvec()
        delta_rot_next = Rotation.from_matrix(R_current.T @ R_next).as_rotvec()
        angular_velocity = (delta_rot_prev + delta_rot_next) / (2 * dt)
        
        # Compute angular acceleration
        angular_acceleration = (delta_rot_next - delta_rot_prev) / (dt ** 2)
        
        # For non-collision frames, normal and tangential forces are zero
        # but we compute relative velocity for potential future collisions
        relative_velocity = linear_velocity.copy()
        normal_velocity = np.zeros(3)
        
        # Estimate forces based on acceleration (F = ma)
        # Get object mass from acoustic shader density and volume
        vertices = trajectory.get_vertices(frame)
        faces = trajectory.get_faces()
        
        # Create mesh to compute volume
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        volume = mesh.volume
        mass = config_obj.acoustic_shader.density * volume
        
        # Total force from acceleration
        total_force = mass * linear_acceleration
        
        # Decompose into normal and tangential components
        # For non-collision, assume gravity is the main normal force
        g_0 = -9.80665
        gravity = np.array([0, g_0, 0])  # m/s²
        normal_force = mass * gravity
        normal_force_magnitude = np.linalg.norm(normal_force)
        
        # Tangential force is what's left after subtracting normal force
        tangential_force = total_force - normal_force
        tangential_force_magnitude = np.linalg.norm(tangential_force)
        
        # Apply stochastic variations to forces based on material properties
        stochastic_normal_force, stochastic_tangential_force = self._stochastic_force_model(normal_force=normal_force, tangential_force=tangential_force, config_obj=config_obj)

        force_data = ForceData(
            frame=frame,
            obj1_idx=obj_idx,
            obj2_idx=-1,  # No second object for non-collision
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            angular_acceleration=angular_acceleration,
            relative_velocity=relative_velocity,
            normal_velocity=normal_velocity,
            normal_force=normal_force,
            tangential_force=tangential_force,
            normal_force_magnitude=normal_force_magnitude,
            tangential_force_magnitude=tangential_force_magnitude,
            stochastic_normal_force=stochastic_normal_force,
            stochastic_tangential_force=stochastic_tangential_force
        )

        return [force_data]

    def _calculate_collision_forces(self, frame: float, obj_idx: int, other_obj_indices: List[int], config_obj: Any, other_config_objs: List[Any], trajectory: Any, other_trajectories: List[Any], collisions: List[Any], sfps: float, sample_rate: int) -> Optional[List[ForceData]]:
        """
        Calculate forces and velocities for a specific collision event or multiple simultaneous collision event.
        
        Implements the stochastic physically-based model for impact sound synthesis.
        """
        force_data_list = []

        # Get time step
        dt = 1.0 / sfps

        for idx in range(len(collisions)):
            collision = collisions[idx]
            other_obj_idx = other_obj_indices[idx]
            other_trajectory = other_trajectories[idx]
            other_config_obj = other_config_objs[idx]
        
        
            # Get positions and velocities before and after collision
            frames = trajectory.get_x()
            frame_idx = np.searchsorted(frames, frame)
        
            if frame_idx >= len(frames) - 1 or frame_idx == 0:
                return None
        
            # Pre-collision (just before impact)
            frame_before = frames[frame_idx - 1]
            pos_before = trajectory.get_position(frame_before)
            rot_before = trajectory.get_rotation(frame_before)
        
            # Post-collision (just after impact)
            frame_after = frames[frame_idx + 1]
            pos_after = trajectory.get_position(frame_after)
            rot_after = trajectory.get_rotation(frame_after)
        
            # Current frame (at impact)
            pos_current = trajectory.get_position(frame)
            rot_current = trajectory.get_rotation(frame)
        
            # Other object states
            other_pos_before = other_trajectory.get_position(frame_before)
            other_pos_after = other_trajectory.get_position(frame_after)
            other_pos_current = other_trajectory.get_position(frame)
        
            # Compute velocities using central difference
            linear_velocity_before = (pos_current - pos_before) / dt
            linear_velocity_after = (pos_after - pos_current) / dt
        
            other_linear_velocity_before = (other_pos_current - other_pos_before) / dt
            other_linear_velocity_after = (other_pos_after - other_pos_current) / dt
        
            # Relative velocity at impact
            relative_velocity = linear_velocity_before - other_linear_velocity_before
        
            # Estimate contact normal (simplified - using direction between centers)
            # In a real implementation, this would use the collision_area data
            contact_normal = (other_pos_current - pos_current)
            contact_normal_mag = np.linalg.norm(contact_normal)
            if contact_normal_mag > 0:
                contact_normal = contact_normal / contact_normal_mag
            else:
                contact_normal = np.array([1, 0, 0])  # Default
        
            # Decompose relative velocity into normal and tangential components
            normal_velocity = np.dot(relative_velocity, contact_normal) * contact_normal
            tangential_velocity = relative_velocity - normal_velocity
        
            # Get material properties
            restitution = abs(linear_velocity_after)/abs(linear_velocity_before)
            if hasattr(config_obj.acoustic_shader, 'restitution'):
                restitution = config_obj.acoustic_shader.restitution
        
            friction = 0.3  # Default friction coefficient
            if hasattr(config_obj.acoustic_shader, 'friction'):
                friction = config_obj.acoustic_shader.friction
        
            # Compute masses
            vertices = trajectory.get_vertices(frame)
            faces = trajectory.get_faces()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            volume = mesh.volume
            mass = config_obj.acoustic_shader.density * volume
        
            other_vertices = other_trajectory.get_vertices(frame)
            other_faces = other_trajectory.get_faces()
            other_mesh = trimesh.Trimesh(vertices=other_vertices, faces=other_faces)
            other_volume = other_mesh.volume
            other_mass = other_config_obj.acoustic_shader.density * other_volume
        
            # Reduced mass for collision
            reduced_mass = (mass * other_mass) / (mass + other_mass)
        
            # Normal impulse (from coefficient of restitution)
            normal_relative_speed = np.dot(relative_velocity, contact_normal)
            normal_impulse = reduced_mass * (1 + restitution) * abs(normal_relative_speed)
        
            # Tangential impulse (friction)
            tangential_speed = np.linalg.norm(tangential_velocity)
            if tangential_speed > 0:
                max_friction_impulse = friction * normal_impulse
                tangential_impulse = min(reduced_mass * tangential_speed, max_friction_impulse)
                tangential_direction = tangential_velocity / tangential_speed
                tangential_force = tangential_impulse / dt * tangential_direction
            else:
                tangential_force = np.zeros(3)
        
            # Normal force (from normal impulse)
            normal_force = normal_impulse / dt * contact_normal

            # Compute accelerations from velocity change
            linear_acceleration = (linear_velocity_after - linear_velocity_before) / dt
        
            # Compute angular quantities (simplified)
            from scipy.spatial.transform import Rotation
            R_before = Rotation.from_euler('xyz', rot_before).as_matrix()
            R_after = Rotation.from_euler('xyz', rot_after).as_matrix()
            R_current = Rotation.from_euler('xyz', rot_current).as_matrix()
        
            # Angular velocity
            delta_rot_before = Rotation.from_matrix(R_before.T @ R_current).as_rotvec()
            delta_rot_after = Rotation.from_matrix(R_current.T @ R_after).as_rotvec()
            angular_velocity = (delta_rot_before + delta_rot_after) / (2 * dt)
        
            # Angular acceleration
            angular_acceleration = (delta_rot_afterafter - delta_rot_before) / (dt ** 2)
        
            # Magnitudes
            normal_force_magnitude = np.linalg.norm(normal_force)
            tangential_force_magnitude = np.linalg.norm(tangential_force)

            # Apply stochastic variations to forces based on material properties
            stochastic_normal_force, stochastic_tangential_force = self._stochastic_force_model(normal_force=normal_force, tangential_force=tangential_force, config_obj=config_obj)

            force_data = ForceData(
                frame=frame,
                obj1_idx=obj_idx,
                obj2_idx=other_obj_idx,
                linear_velocity=linear_velocity_before,  # Velocity at impact
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                angular_acceleration=angular_acceleration,
                relative_velocity=relative_velocity,
                normal_velocity=normal_velocity,
                normal_force=normal_force,
                tangential_force=tangential_force,
                normal_force_magnitude=normal_force_magnitude,
                tangential_force_magnitude=tangential_force_magnitude,
                stochastic_normal_force=stochastic_normal_force,
                stochastic_tangential_force=stochastic_tangential_force
            )
            force_data_list.append(force_data)

        return force_data_list

    def _stochastic_force_model(self, normal_force: np.ndarray, tangential_force: np.ndarray, config_obj: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stochastic variations to forces based on material properties.
        Based on the stochastic physically-based model from the referenced paper.
        """
        # Extract material properties
        print('config_obj.acoustic_shader: ', config_obj.acoustic_shader.young_modulus, config_obj.acoustic_shader.roughness)
        youngs_modulus = config_obj.acoustic_shader.young_modulus
        poissons_ratio = config_obj.acoustic_shader.poisson_ratio
        density = config_obj.acoustic_shader.density
        roughness = config_obj.acoustic_shader.roughness
        
        # Stochastic variation factors
        # These would be based on the paper's stochastic model
        normal_variation = 1.0 + np.random.normal(0, 0.1) * roughness
        tangential_variation = 1.0 + np.random.normal(0, 0.15) * roughness
        
        # Apply stochastic variations
        stochastic_normal_force = normal_force * normal_variation
        stochastic_tangential_force = tangential_force * tangential_variation
        
        # Add high-frequency components (simulating surface roughness effects)
        hf_factor = np.random.normal(0, 0.05 * roughness)
        hf_component = np.random.randn(3) * hf_factor * np.linalg.norm(normal_force)
        
        stochastic_normal_force += hf_component
        
        return stochastic_normal_force, stochastic_tangential_force
