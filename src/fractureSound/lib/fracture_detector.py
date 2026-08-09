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
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import TrajectoryData, ForceDataSequence, CollisionData

from .fracture_data import FractureEvent, FractureType, FragmentData

@dataclass
class FractureDetector:
    """
    Detects and classifies fracture events from trajectory and force data.
   
    Implements the fracture detection algorithm described in:
    "Fracture Sound: A physically based approach to the synthesis of fracture sounds"
    """
   
    entity_manager: EntityManager
   
    # Detection parameters
    velocity_threshold: float = 0.01  # Minimum velocity change for fracture detection (m/s)
    energy_threshold: float = 0.01   # Minimum energy release for fracture (J)
    time_window: float = 0.02        # Time window for fracture detection (seconds)
   
    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.sample_rate = config.system.sample_rate
        self.fps = config.system.fps
        self.fps_base = config.system.fps_base
        self.subframes = config.system.subframes
        self.sfps = (self.fps / self.fps_base) * self.subframes

        self.fracture_dir = f"{config.system.cache_path}/fracture"
        os.makedirs(self.fracture_dir, exist_ok=True)
   
    def detect_fracture_events(self, obj_idx: int, fragment_indices: List[int]) -> List[FractureEvent]:
        """
        Detect fracture events by analyzing the transition from original object to fragments.

        Parameters:
        -----------
        obj_idx : int
            Index of the original object before fracture
        fragment_indices : List[int]
            Indices of the fragments after fracture

        Returns:
        --------
        List[FractureEvent]
            Detected fracture events
        """
        config = self.entity_manager.get('config')

        # Get object configurations
        original_obj = None
        fragments = []

        for obj in config.objects:
            if obj.idx == obj_idx:
                original_obj = obj
            elif obj.idx in fragment_indices:
                fragments.append(obj)

        if original_obj is None:
            raise ValueError(f"Object {obj_idx} not found")

        if len(fragments) == 0:
            debug_print(f"No fragments found for object {original_obj.name}")
            return []

        # Get trajectories
        trajectories = self.entity_manager.get('trajectories')
        original_trajectory = None
        fragment_trajectories = {}

        for traj in trajectories.values():
            if isinstance(traj, TrajectoryData):
                if traj.obj_idx == obj_idx:
                    original_trajectory = traj
                elif traj.obj_idx in fragment_indices:
                    fragment_trajectories[traj.obj_idx] = traj

        if original_trajectory is None:
            raise ValueError(f"Trajectory for object {obj_idx} not found")

        # Get fracture frame from config
        fracture_frame = original_obj.fractured
        if fracture_frame is not False:
            # Convert frame to samples
            fracture_sample = fracture_frame * self.sample_rate / self.sfps
        else:
            debug_print(f"No fracture frame specified for {original_obj.name}")
            return []

        # Find the exact fracture moment by analyzing trajectories
        fracture_moment = self._find_fracture_moment(original_trajectory, fragment_trajectories, fracture_sample, fragment_indices)

        if fracture_moment is None:
            debug_print(f"Could not determine fracture moment for {original_obj.name}")
            return []

        # Collect fracture-related data
        fracture_collisions = []
        fracture_forces = []

        # Get collision and force data near fracture
        collisions = self.entity_manager.get('collisions')
        forces = self.entity_manager.get('forces')

        for coll in collisions.values():
            if isinstance(coll, CollisionData):
                if coll.obj1_idx == obj_idx or coll.obj2_idx == obj_idx:
                    if coll.frame <= fracture_moment <= coll.frame + coll.frame_range:
                        fracture_collisions.append(coll)

        for force in forces.values():
            if isinstance(force, ForceDataSequence):
                if force.obj_idx == obj_idx or force.other_obj_idx == obj_idx:
                    fracture_forces.append(force)
#                    # Check if force data covers the fracture moment
#                    if force.frames[0] <= fracture_moment <= force.frames[-1]:
#                        fracture_forces.append(force)

        # Get pre-fracture state
        pre_velocity = original_trajectory.get_velocity(fracture_moment - 0.001)
        pre_angular_velocity = original_trajectory.get_angular_velocity(fracture_moment - 0.001)
        pre_force = self._get_force_at_time(fracture_forces, obj_idx, fracture_moment - 0.001)

        # Get material properties
        young_modulus = original_obj.acoustic_shader.young_modulus if original_obj.acoustic_shader else 1e9
        density = original_obj.acoustic_shader.density if original_obj.acoustic_shader else 1000.0
        damping = original_obj.acoustic_shader.damping if original_obj.acoustic_shader else 0.02

        # Get fragment states after fracture
        fragment_velocities = []
        fragment_angular_velocities = []
        fragment_data_list = []

        for frag in fragments:
            traj = fragment_trajectories.get(frag.idx)
            if traj is not None:
                # Get velocity just after fracture
                vel = traj.get_velocity(fracture_moment + 0.001)
                ang_vel = traj.get_angular_velocity(fracture_moment + 0.001)
                fragment_velocities.append(vel)
                fragment_angular_velocities.append(ang_vel)

                # Get fragment geometry at fracture moment
                vertices = traj.get_vertices(fracture_moment)
                normals = traj.get_normals(fracture_moment)
                faces = traj.get_faces()

                # Compute fragment properties
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                mesh.density = density

                fragment_data = FragmentData(
                    obj_idx=frag.idx,
                    obj_name=frag.name,
                    vertices=vertices,
                    normals=normals,
                    faces=faces,
                    mass=mesh.mass,
                    volume=mesh.volume,
                    center_of_mass=mesh.center_mass,
                    inertia_tensor=mesh.moment_inertia,
                    parent_obj_idx=obj_idx,
                    is_shard=True,
                    fracture_frame=fracture_moment
                )
                fragment_data_list.append(fragment_data)

        # Classify fracture type
        fracture_type = self._classify_fracture_type(
            original_obj,
            fragments,
            pre_velocity,
            fragment_velocities,
            fracture_moment,
            fracture_collisions
        )

        # Compute fracture energy
        fracture_energy = self._compute_fracture_energy(
            original_obj,
            fragments,
            pre_velocity,
            fragment_velocities,
            fracture_collisions,
            fracture_forces,
            fracture_moment
        )

        # Estimate crack length
        crack_length = self._estimate_crack_length(original_obj, fragments, fracture_moment)

        # Create fracture event
        event = FractureEvent(
            fracture_type=fracture_type,
            frame=fracture_moment,
            original_obj_idx=obj_idx,
            original_obj_name=original_obj.name,
            fragment_indices=fragment_indices,
            pre_fracture_velocity=pre_velocity,
            pre_fracture_angular_velocity=pre_angular_velocity,
            pre_fracture_force=pre_force,
            pre_fracture_stress=np.zeros(6),  # Will be computed later
            fragment_velocities=fragment_velocities,
            fragment_angular_velocities=fragment_angular_velocities,
            fracture_energy=fracture_energy,
            crack_velocity=500.0,  # Default crack velocity
            crack_length=crack_length,
            young_modulus=young_modulus,
            density=density,
            damping=damping,
            failure_stress=original_obj.acoustic_shader.failure_stress if hasattr(original_obj.acoustic_shader, 'failure_stress') else 1e6,
            fragment_data=fragment_data_list,
            collision_data=fracture_collisions,
            force_data=fracture_forces
        )

        # Save the event
        event.save(f"{self.fracture_dir}/event_{obj_idx}_{fracture_moment:.3f}.pkl")

        debug_print(f"Detected {fracture_type.value} fracture for {original_obj.name} "
                   f"at frame {fracture_moment:.3f}, energy: {fracture_energy:.6f}J")

        return [event]

    def _find_fracture_moment(self, original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], approx_frame: float, fragment_indices: List[int]) -> Optional[float]:
        """
        Find the exact fracture moment by analyzing trajectory data.

        Uses the method described in the fracture sound paper:
        - Look for discontinuity in velocity
        - Find the moment where the original object trajectory diverges from fragments
        """
        # Get time range around approximate fracture
        time_range = self.time_window * self.sample_rate # default to 20ms window
        start_time = max(0, approx_frame - time_range)
        end_time = approx_frame + time_range

        # Round to int time values
        time_range = int(time_range)
        start_time = int(np.floor(start_time))
        end_time = int(np.ceil(end_time))
        samples_range = end_time - start_time

        # Sample trajectories in this range
        times = np.linspace(start_time, end_time, samples_range)

        # Get positions of original object
        original_positions = np.array([original_trajectory.get_position(t) for t in times])

        # Get positions of fragments (average)
        fragment_positions = []
        for frag_idx in fragment_indices:
            if frag_idx in fragment_trajectories:
                traj = fragment_trajectories[frag_idx]
                pos = np.array([traj.get_position(t) for t in times])
                fragment_positions.append(pos)

        if len(fragment_positions) == 0:
            return approx_frame

        avg_fragment_positions = np.mean(fragment_positions, axis=0)

        # Compute distance between original and fragment trajectories
        distances = np.linalg.norm(original_positions - avg_fragment_positions, axis=1)

        # Find the point where distance starts increasing rapidly
        # This indicates the fracture moment

        # Compute derivative of distance
        distance_derivative = np.gradient(distances)

        # Find the maximum derivative (rapid separation)
        max_deriv_idx = np.argmax(distance_derivative)

        if max_deriv_idx > 0 and max_deriv_idx < len(times) - 1:
            fracture_moment = times[max_deriv_idx]
        else:
            # Fallback: use approximate frame
            fracture_moment = approx_frame

###################################################################################################################
# Use trajectory geometric approch to find the fracture moment
###################################################################################################################
        # Refine using velocity discontinuity
        # Get velocities before and after
        pre_vel = original_trajectory.get_velocity(fracture_moment - 0.005)
        post_vel = original_trajectory.get_velocity(fracture_moment + 0.005)

        # If velocity change is significant, we found the fracture moment
        if np.linalg.norm(post_vel - pre_vel) > 0.1:
            return fracture_moment

        # Otherwise, use the approximate frame
        return approx_frame

    def _classify_fracture_type(self, original_obj: Any, fragments: List[Any], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], fracture_moment: float, collisions: List[CollisionData]) -> FractureType:
        """
        Classify the fracture type based on trajectory and force data.

        Classification criteria:
        - SHATTER: Multiple fragments with high velocity, many collisions
        - CRACK: Single crack, moderate velocity, few collisions
        - SNAP: Two fragments, high energy release, sudden separation
        """
        n_fragments = len(fragments)

        # Check if there are multiple fragments (shatter)
        if n_fragments >= 3:
            # Check if fragments are moving apart rapidly
            avg_velocity = np.mean([np.linalg.norm(v) for v in fragment_velocities])
            pre_speed = np.linalg.norm(pre_velocity)

            if avg_velocity > pre_speed * 1.5 and avg_velocity > 1.0:
                return FractureType.SHATTER

        # Check for snap (two fragments)
        if n_fragments == 2:
            # Compute relative velocity between fragments
            if len(fragment_velocities) >= 2:
                rel_velocity = np.linalg.norm(fragment_velocities[0] - fragment_velocities[1])
                pre_speed = np.linalg.norm(pre_velocity)

                # If relative velocity is high and there's a sudden separation
                if rel_velocity > pre_speed * 0.5 or rel_velocity > 2.0:
                    return FractureType.SNAP

        # Check for crack (one or two fragments, lower energy)
        if n_fragments <= 2:
            # Check if there were collisions at fracture
            if len(collisions) > 0:
                # Check collision energy
                total_energy = 0
                for coll in collisions:
                    if hasattr(coll, 'impulse_range') and coll.impulse_range:
                        total_energy += coll.impulse_range * 0.001

                if total_energy < 0.1:
                    return FractureType.CRACK

        # Default: crack
        return FractureType.CRACK

    def _compute_fracture_energy(self, original_obj: Any, fragments: List[Any], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], collisions: List[CollisionData], forces: List[ForceDataSequence], fracture_moment: float) -> float:
        """
        Compute the energy released during fracture.

        Energy components:
        1. Kinetic energy change of fragments
        2. Energy from collisions at fracture
        3. Stored elastic energy release
        """
        total_energy = 0.0

        # 1. Kinetic energy change
        if len(fragment_velocities) > 0:
            # Get masses
            masses = []
            for frag in fragments:
                if frag.acoustic_shader and hasattr(frag.acoustic_shader, 'density'):
                    # Approximate mass from density and volume
                    try:
                        # Get fragment volume from trajectory
                        traj = self.entity_manager.get('trajectories')
                        for t in traj.values():
                            if hasattr(t, 'obj_idx') and t.obj_idx == frag.idx:
                                vertices = t.get_vertices(fracture_moment)
                                faces = t.get_faces()
                                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                                mass = mesh.volume * frag.acoustic_shader.density
                                masses.append(mass)
                                break
                    except:
                        masses.append(0.001)  # Default small mass
                else:
                    masses.append(0.001)

            # Compute kinetic energy of fragments
            kinetic_energy = 0.0
            for i, vel in enumerate(fragment_velocities):
                if i < len(masses):
                    kinetic_energy += 0.5 * masses[i] * np.linalg.norm(vel)**2

            # Pre-fracture kinetic energy
            pre_mass = sum(masses) if masses else 0.001
            pre_kinetic = 0.5 * pre_mass * np.linalg.norm(pre_velocity)**2

            total_energy += max(0, kinetic_energy - pre_kinetic)

        # 2. Collision energy
        collision_energy = 0.0
        for coll in collisions:
            if hasattr(coll, 'distances') and coll.distances is not None:
                # Estimate energy from penetration
                if isinstance(coll.distances, np.ndarray):
                    penetration_energy = np.sum(np.abs(coll.distances)) * 10.0
                    collision_energy += penetration_energy

        total_energy += collision_energy

        # 3. Stored elastic energy
        # E = σ²V/(2Y) where σ is failure stress, V is stressed volume
        if hasattr(original_obj.acoustic_shader, 'failure_stress'):
            failure_stress = original_obj.acoustic_shader.failure_stress
        else:
            failure_stress = 1e6  # Default

        young_modulus = original_obj.acoustic_shader.young_modulus if original_obj.acoustic_shader else 1e9

        # Estimate stressed volume (approximate)
        stressed_volume = 0.0001  # Default small volume

        elastic_energy = (failure_stress**2 * stressed_volume) / (2 * young_modulus)
        total_energy += elastic_energy

        return max(total_energy, 0.01)  # Ensure minimum energy

    def _estimate_crack_length(self, original_obj: Any, fragments: List[Any],
                                fracture_moment: float) -> float:
        """Estimate the crack length from fragment geometry."""
        try:
            # Get original mesh at fracture moment
            from pbrAudioCommon import _load_mesh
            vertices, _, faces = _load_mesh(original_obj, int(fracture_moment * self.sfps / self.sample_rate))

            # Get fragment meshes
            fragment_vertices = []
            for frag in fragments:
                traj = self.entity_manager.get('trajectories')
                for t in traj.values():
                    if hasattr(t, 'obj_idx') and t.obj_idx == frag.idx:
                        verts = t.get_vertices(fracture_moment)
                        fragment_vertices.append(verts)
                        break

            if len(fragment_vertices) < 2:
                return 0.1  # Default

            # Estimate crack length as the distance between fragment centers
            original_center = np.mean(vertices, axis=0)
            fragment_centers = [np.mean(verts, axis=0) for verts in fragment_vertices]

            # Compute average distance from original center to fragment centers
            distances = [np.linalg.norm(c - original_center) for c in fragment_centers]

            return max(distances) * 0.5

        except Exception as e:
            debug_print(f"Error estimating crack length: {e}")
            return 0.1

    def _get_force_at_time(self, force_sequences: List[ForceDataSequence], obj_idx: int, time: float) -> np.ndarray:
        """Get force vector at a specific time from force data sequences."""
        for force_seq in force_sequences:
            if force_seq.obj_idx == obj_idx or force_seq.other_obj_idx == obj_idx:
                try:
                    return force_seq.get_normal_force(time)
                except:
                    pass
        return np.zeros(3)
