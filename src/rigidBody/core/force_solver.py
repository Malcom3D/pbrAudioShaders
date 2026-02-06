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
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import math

from ..core.entity_manager import EntityManager
from ..lib.force_data import ContactType, ForceData, ForceDataSequence
from ..lib.contact_geometry import ContactGeometry
from ..lib.hertzian_contact import HertzianContact
from ..lib.cubicspline_with_nan import CubicSplineWithNaN

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

                forces_frames, other_obj_idx, restitution, relative_velocity, normal_velocity, normal_force, tangential_force, tangential_velocity, normal_force_magnitude, tangential_force_magnitude, stochastic_normal_force, stochastic_tangential_force, contact_type, contact_point, contact_radius, rolling_radius, impact_duration, contact_pressure, penetration_depth, coupling_strength = ([] for _ in range(20))

                for idx in range(len(frames)):
                    collisions, other_obj_indices, other_trajectories, other_config_objs = ([] for _ in range(4))
                    frame = frames[idx]            
                    for c_idx in range(len(active_collisions)):
                        if (active_collisions[c_idx].frame <= frame <= active_collisions[c_idx].frame + active_collisions[c_idx].frame_range):
                            collisions.append(active_collisions[c_idx])
                            other_obj_indices.append(active_collisions[c_idx].obj1_idx if not active_collisions[c_idx].obj1_idx == obj_idx else active_collisions[c_idx].obj2_idx)
                            for t_idx in trajectories.keys():
                                if 'TrajectoryData' in str(type(trajectories[t_idx])):
                                    if trajectories[t_idx].obj_idx in other_obj_indices:
                                        other_trajectories.append(trajectories[t_idx])
                                    for other_config in config.objects:
                                        if other_config.idx in other_obj_indices:
                                            other_config_objs.append(other_config)
                    if collisions == []: 
                        forces_data = self._calculate_forces(frame=frame, obj_idx=obj_idx, config_obj=config_obj, trajectory=trajectory, sfps=sfps, sample_rate=sample_rate)
                    else:
                        forces_data = self._calculate_collision_forces(frame=frame, obj_idx=obj_idx, other_obj_indices=other_obj_indices, config_obj=config_obj, other_config_objs=other_config_objs, trajectory=trajectory, other_trajectories=other_trajectories, collisions=collisions, sfps=sfps, sample_rate=sample_rate)

                    if not forces_data == None:
                        for index in range(len(forces_data)):
                            force_data = forces_data[index]
                            forces_frames.append(force_data.frame)
                            if not force_data.obj2_idx == -1:
                                other_obj_idx.append(force_data.obj2_idx if not force_data.obj2_idx == obj_idx else force_data.obj1_idx)
                            restitution.append(force_data.restitution)
                            relative_velocity.append(force_data.relative_velocity)
                            normal_velocity.append(force_data.normal_velocity)
                            normal_force.append(force_data.normal_force)
                            tangential_force.append(force_data.tangential_force)
                            tangential_velocity.append(force_data.tangential_velocity)
                            normal_force_magnitude.append(force_data.normal_force_magnitude)
                            tangential_force_magnitude.append(force_data.tangential_force_magnitude)
                            stochastic_normal_force.append(force_data.stochastic_normal_force)
                            stochastic_tangential_force.append(force_data.stochastic_tangential_force)
                            print('force_data.contact_type: ', force_data.contact_type)
                            contact_type.append(force_data.contact_type if not force_data.contact_type == None else 0)
                            print('force_data.contact_point: ', type(force_data.contact_point), force_data.contact_point)
                            contact_point.append(force_data.contact_point if isinstance(force_data.contact_point, np.ndarray) else np.array([np.nan,np.nan,np.nan]))
                            print('force_data.contact_radius: ', force_data.contact_radius)
                            contact_radius.append(force_data.contact_radius if not force_data.contact_radius == None else np.nan)
                            print('force_data.rolling_radius: ', force_data.rolling_radius)
                            rolling_radius.append(force_data.rolling_radius if not force_data.rolling_radius == None else np.nan)
                            print('force_data.impact_duration: ', force_data.impact_duration)
                            impact_duration.append(force_data.impact_duration if not force_data.impact_duration == None else np.nan)
                            print('force_data.contact_pressure: ', force_data.contact_pressure)
                            contact_pressure.append(force_data.contact_pressure if not force_data.contact_pressure == None else np.nan)
                            print('force_data.penetration_depth: ', force_data.penetration_depth)
                            penetration_depth.append(force_data.penetration_depth if not force_data.penetration_depth == None else np.nan)
                            print('force_data.coupling_strength: ', force_data.coupling_strength)
                            coupling_strength.append(force_data.coupling_strength if not force_data.coupling_strength == None else 0.0)

                forces_frames = np.unique(np.sort(np.array(forces_frames)))
                other_obj_idx = np.unique(np.sort(np.array(other_obj_idx)))

                restitution = np.array(restitution)
                relative_velocity = np.array(relative_velocity)
                normal_velocity = np.array(normal_velocity)
                normal_force = np.array(normal_force)
                tangential_force = np.array(tangential_force)
                tangential_velocity = np.array(tangential_velocity)
                normal_force_magnitude = np.array(normal_force_magnitude)
                tangential_force_magnitude = np.array(tangential_force_magnitude)
                stochastic_normal_force = np.array(stochastic_normal_force)
                stochastic_tangential_force = np.array(stochastic_tangential_force)
                contact_type = np.array(contact_type)
                contact_point = np.array(contact_point)
                contact_radius = np.array(contact_radius)
                rolling_radius = np.array(rolling_radius)
                impact_duration = np.array(impact_duration)
                contact_pressure = np.array(contact_pressure)
                penetration_depth = np.array(penetration_depth)
                coupling_strength = np.array(coupling_strength)

                # create interpolator
                restitution = CubicSpline(forces_frames, restitution, extrapolate=1)
                relative_velocity = [CubicSpline(forces_frames, relative_velocity[:, i], extrapolate=1) for i in range(relative_velocity.shape[1])]
                normal_velocity = [CubicSpline(forces_frames, normal_velocity[:, i], extrapolate=1) for i in range(normal_velocity.shape[1])]
                normal_force = [CubicSpline(forces_frames, normal_force[:, i], extrapolate=1) for i in range(normal_force.shape[1])]
                tangential_force = [CubicSpline(forces_frames, tangential_force[:, i], extrapolate=1) for i in range(tangential_force.shape[1])]
                tangential_velocity = [CubicSpline(forces_frames, tangential_velocity[:, i], extrapolate=1) for i in range(tangential_velocity.shape[1])]
                normal_force_magnitude = CubicSpline(forces_frames, normal_force_magnitude, extrapolate=1)
                tangential_force_magnitude = CubicSpline(forces_frames, tangential_force_magnitude, extrapolate=1)
                stochastic_normal_force = [CubicSpline(forces_frames, stochastic_normal_force[:, i], extrapolate=1) for i in range(stochastic_normal_force.shape[1])]
                stochastic_tangential_force = [CubicSpline(forces_frames, stochastic_tangential_force[:, i], extrapolate=1) for i in range(stochastic_tangential_force.shape[1])]
                print(contact_point)
                contact_point = [CubicSplineWithNaN(forces_frames, contact_point[:, i], extrapolate=1) for i in range(contact_point.shape[1])]
                contact_radius = CubicSplineWithNaN(forces_frames, contact_radius, extrapolate=1)
                rolling_radius = CubicSplineWithNaN(forces_frames, rolling_radius, extrapolate=1)
                contact_pressure = CubicSplineWithNaN(forces_frames, contact_pressure, extrapolate=1)
                penetration_depth = CubicSplineWithNaN(forces_frames, penetration_depth, extrapolate=1)
                coupling_strength = CubicSpline(forces_frames, coupling_strength, extrapolate=1)

                force_data_sequence = ForceDataSequence(frames=forces_frames, obj_idx=obj_idx, other_obj_idx=other_obj_idx, restitution=restitution, relative_velocity=relative_velocity, normal_velocity=normal_velocity, normal_force=normal_force, tangential_force=tangential_force, tangential_velocity=tangential_velocity, normal_force_magnitude=normal_force_magnitude, tangential_force_magnitude=tangential_force_magnitude, stochastic_normal_force=stochastic_normal_force, stochastic_tangential_force=stochastic_tangential_force, contact_type=contact_type, contact_point=contact_point, contact_radius=contact_radius, rolling_radius=rolling_radius, impact_duration=impact_duration, contact_pressure=contact_pressure, penetration_depth=penetration_depth, coupling_strength=coupling_strength)

                force_idx = len(self.entity_manager.get('forces')) + 1
                self.entity_manager.register('forces', force_data_sequence, force_idx)

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
        
        # Get linear velocity
        linear_velocity = trajectory.get_velocity(frames[frame_idx])
        
        # Get linear acceleration
        linear_acceleration = trajectory.get_acceleration(frames[frame_idx])
        
        # For non-collision frames, normal and tangential forces are zero
        # but we compute relative velocity for potential future collisions
        relative_velocity = linear_velocity.copy()
        normal_velocity = np.zeros(3)
        tangential_velocity = np.zeros(3)
        
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
            restitution=0,
            relative_velocity=relative_velocity,
            normal_velocity=normal_velocity,
            normal_force=normal_force,
            tangential_force=tangential_force,
            tangential_velocity=tangential_velocity,
            normal_force_magnitude=normal_force_magnitude,
            tangential_force_magnitude=tangential_force_magnitude,
            stochastic_normal_force=stochastic_normal_force,
            stochastic_tangential_force=stochastic_tangential_force
        )

        return [force_data]

    def _calculate_collision_forces(self, frame: float, obj_idx: int, other_obj_indices: List[int], config_obj: Any, other_config_objs: List[Any], trajectory: Any, other_trajectories: List[Any], collisions: List[Any], sfps: float, sample_rate: int) -> Optional[List[ForceData]]:
        """
        Calculate forces and velocities for a specific collision event or multiple simultaneous collision events.
        
        Implements the stochastic physically-based model for collisions sound synthesis.
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
        
            # Current frame (at impact)
            pos_current = trajectory.get_position(frame)
            verts_current = trajectory.get_vertices(frame)
            faces_current = trajectory.get_faces()
        
            # Other object states
            other_pos_current = other_trajectory.get_position(frame)
            other_verts_current = other_trajectory.get_vertices(frame)
            other_faces_current = other_trajectory.get_faces()
        
            # Compute velocities using central difference
            linear_velocity_before = trajectory.get_velocity(frame_before)
            linear_velocity_after = trajectory.get_velocity(frame_after)
        
            other_linear_velocity_before = other_trajectory.get_velocity(frame_before)
            other_linear_velocity_after = other_trajectory.get_velocity(frame_after)
        
            # Relative velocity at impact
            relative_velocity = linear_velocity_before - other_linear_velocity_before
        
            # Estimate contact normal using approximated collision_area from ContactGeometry class
            # Create collision detector
            detector = ContactGeometry(verts_current, faces_current, pos_current, other_verts_current, other_faces_current, other_pos_current)
            contact_normal = detector.get_contact_normal()
            contact_normal_mag = np.linalg.norm(contact_normal)
            if contact_normal_mag > 0:
                contact_normal = contact_normal / contact_normal_mag
            else:
                contact_normal = np.array([1, 0, 0])  # Default
        
            # Decompose relative velocity into normal and tangential components
            normal_velocity = np.dot(relative_velocity, contact_normal) * contact_normal
            tangential_velocity = relative_velocity - normal_velocity
        
            # Get material properties
            restitution = abs(np.linalg.norm(other_linear_velocity_after) - np.linalg.norm(linear_velocity_after))/abs(np.linalg.norm(linear_velocity_before) - np.linalg.norm(other_linear_velocity_before))
            if hasattr(config_obj.acoustic_shader, 'restitution'):
                restitution = config_obj.acoustic_shader.restitution
        
            friction = 0.3  # Default friction coefficient
            if hasattr(config_obj.acoustic_shader, 'friction'):
                friction = config_obj.acoustic_shader.friction
        
            # Compute masses
            vertices = trajectory.get_vertices(frame)
            faces = trajectory.get_faces()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.density = config_obj.acoustic_shader.density
            mass = mesh.mass
        
            other_vertices = other_trajectory.get_vertices(frame)
            other_faces = other_trajectory.get_faces()
            other_mesh = trimesh.Trimesh(vertices=other_vertices, faces=other_faces)
            other_mesh.density = other_config_obj.acoustic_shader.density
            other_mass = other_mesh.mass
        
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
            linear_acceleration = trajectory.get_acceleration(frame)
        
            # Magnitudes
            normal_force_magnitude = np.linalg.norm(normal_force)
            tangential_force_magnitude = np.linalg.norm(tangential_force)

            # Apply stochastic variations to forces based on material properties
            stochastic_normal_force, stochastic_tangential_force = self._stochastic_force_model(normal_force=normal_force, tangential_force=tangential_force, config_obj=config_obj)

            # Init the ForceData class
            force_data = ForceData(
                frame=frame,
                obj1_idx=obj_idx,
                obj2_idx=other_obj_idx,
                restitution=restitution,
                relative_velocity=relative_velocity,
                normal_velocity=normal_velocity,
                normal_force=normal_force,
                tangential_force=tangential_force,
                tangential_velocity=tangential_velocity,
                normal_force_magnitude=normal_force_magnitude,
                tangential_force_magnitude=tangential_force_magnitude,
                stochastic_normal_force=stochastic_normal_force,
                stochastic_tangential_force=stochastic_tangential_force
            )

            # Analyzes collision between two rigidbody meshes using Hertzian contact theory.
            hertzian_contact = HertzianContact(self.entity_manager)
            hertz_impact_data, hertz_continuous_data = (None for _ in range(2))
            frame_range = collision.frame_range
            if collision.type.value == 'impact':
                hertz_impact_data = hertzian_contact.compute_impact(obj_idx, other_obj_idx, frame)
            elif collision.type.value == 'contact' and collision.frame == frame: 
                hertz_impact_data = hertzian_contact.compute_impact(obj_idx, other_obj_idx, frame)
                hertz_continuous_data = hertzian_contact.compute_continuous_contact(force_data, obj_idx, other_obj_idx, frame, frame_range)
            else:
                hertz_continuous_data = hertzian_contact.compute_continuous_contact(force_data, obj_idx, other_obj_idx, frame, frame_range)

            if not hertz_impact_data == None and hertz_continuous_data == None:
                contact_type = hertz_impact_data['contact_type'].value
                contact_point = hertz_impact_data['contact_point']
                contact_radius = hertz_impact_data['contact_radius']
                rolling_radius = np.nan
                impact_duration = hertz_impact_data['duration']
                contact_pressure = hertz_impact_data['contact_pressure']
                penetration_depth = hertz_impact_data['penetration_depth']
                coupling_strength = hertz_impact_data['coupling_strength']
            elif not hertz_impact_data == None and not hertz_continuous_data == None:
                contact_type = hertz_continuous_data['contact_type'].value
                contact_point = (hertz_impact_data['contact_point'] + hertz_continuous_data['contact_point']) / 2
                contact_radius = (hertz_impact_data['contact_radius'] + hertz_continuous_data['contact_radius']) / 2
                rolling_radius = hertz_continuous_data['rolling_radius']
                impact_duration = hertz_impact_data['duration']
                contact_pressure = (hertz_impact_data['contact_pressure'] + hertz_continuous_data['contact_pressure']) / 2
                penetration_depth = (hertz_impact_data['penetration_depth'] + hertz_continuous_data['penetration_depth']) / 2
                coupling_strength = (hertz_impact_data['coupling_strength'] + hertz_continuous_data['coupling_strength']) / 2
            elif hertz_impact_data == None and not hertz_continuous_data == None:
                contact_type = hertz_continuous_data['contact_type'].value
                contact_point = hertz_continuous_data['contact_point']
                contact_radius = hertz_continuous_data['contact_radius']
                rolling_radius = hertz_continuous_data['rolling_radius']
                impact_duration = np.nan
                contact_pressure = hertz_continuous_data['contact_pressure']
                penetration_depth = hertz_continuous_data['penetration_depth']
                coupling_strength = hertz_continuous_data['coupling_strength']

            # Rewrite complete ForceData class
            force_data = ForceData(
                frame=frame,
                obj1_idx=obj_idx,
                obj2_idx=other_obj_idx,
                restitution=restitution,
                relative_velocity=relative_velocity,
                normal_velocity=normal_velocity,
                normal_force=normal_force,
                tangential_force=tangential_force,
                tangential_velocity=tangential_velocity,
                normal_force_magnitude=normal_force_magnitude,
                tangential_force_magnitude=tangential_force_magnitude,
                stochastic_normal_force=stochastic_normal_force,
                stochastic_tangential_force=stochastic_tangential_force,
                contact_type=contact_type,
                contact_point=contact_point,
                contact_radius=contact_radius,
                rolling_radius=rolling_radius,
                impact_duration=impact_duration,
                contact_pressure=contact_pressure,
                penetration_depth=penetration_depth,
                coupling_strength=coupling_strength
            )

            force_data_list.append(force_data)

        return force_data_list

    def _stochastic_force_model(self, normal_force: np.ndarray, tangential_force: np.ndarray, config_obj: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stochastic variations to forces based on material properties.
        Based on the stochastic physically-based model from the referenced paper.
        """
        # Extract material properties
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
