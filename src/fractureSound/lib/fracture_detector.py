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

# fractureSound/lib/fracture_detector.py

import os
import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import TrajectoryData, ForceDataSequence, CollisionData

from .fracture_data import FractureEvent, FractureType, FragmentData


@dataclass
class FractureDetector:
    """
    Detects and classifies fracture events from trajectory data using geometric methods.
    
    Uses shard TrajectoryData, poses, and bounding box analysis to trace trajectories
    from the ObjectConfig.is_shard frame to the exact moments of fracture (begin-end)
    when shard bounding boxes align with the original object's bounding box.
    """
    
    entity_manager: EntityManager
    
    # Detection parameters
    position_tolerance: float = 0.001  # Position matching tolerance (meters)
    velocity_threshold: float = 0.01   # Minimum velocity change for fracture detection (m/s)
    energy_threshold: float = 0.01     # Minimum energy release for fracture (J)
    time_window: float = 0.02          # Time window for fracture detection (seconds)
    sampling_interval: float = 0.001   # Sampling interval for search (seconds)
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.sample_rate = config.system.sample_rate
        self.fps = config.system.fps
        self.fps_base = config.system.fps_base
        self.subframes = config.system.subframes
        self.sfps = (self.fps / self.fps_base) * self.subframes

        self.sampling_interval = 1 / self.sample_rate

        self.fracture_dir = f"{config.system.cache_path}/fracture"
        os.makedirs(self.fracture_dir, exist_ok=True)
    
    def detect_fracture_events(self, obj_idx: int, fragment_indices: List[int]) -> List[FractureEvent]:
        """
        Detect fracture events by geometric analysis of trajectories.
        
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

        # Get fracture frame from config (approximate starting point)
        fracture_frame_approx = original_obj.fractured
        if fracture_frame_approx is False:
            debug_print(f"No fracture frame specified for {original_obj.name}")
            return []

#        # Get shard start frame from config
#        shard_start_frame = original_obj.is_shard
#        if shard_start_frame is False:
#            shard_start_frame = fracture_frame_approx

        # Convert frames to samples
        fracture_sample_approx = fracture_frame_approx * self.sample_rate / self.sfps
#        shard_start_sample = shard_start_frame * self.sample_rate / self.sfps

        # Find the exact fracture moments using geometric analysis
        fracture_moments = self._find_fracture_moments_geometric(original_trajectory=original_trajectory, fragment_trajectories=fragment_trajectories, fracture_sample_approx=fracture_sample_approx, fragments=fragments, original_obj=original_obj)

        if fracture_moments is None or len(fracture_moments) == 0:
            debug_print(f"Could not determine fracture moments for {original_obj.name}")
            return []

        # Get the primary fracture moment (begin)
        fracture_begin = fracture_moments.get('begin')
        fracture_end = fracture_moments.get('end')
        
        if fracture_begin is None:
            debug_print(f"Could not determine fracture begin for {original_obj.name}")
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
                    if coll.frame <= fracture_end and coll.frame + coll.frame_range >= fracture_begin:
                        fracture_collisions.append(coll)

        for force in forces.values():
            if isinstance(force, ForceDataSequence):
                if force.obj_idx == obj_idx or force.other_obj_idx == obj_idx:
                    fracture_forces.append(force)

        # Get pre-fracture state (at fracture_begin)
        pre_velocity = original_trajectory.get_velocity(fracture_begin - 10)
        pre_angular_velocity = original_trajectory.get_angular_velocity(fracture_begin - 10)
        pre_force = self._get_force_at_time(fracture_forces, obj_idx, fracture_begin - 10)

        # Get material properties
        young_modulus = original_obj.acoustic_shader.young_modulus if original_obj.acoustic_shader else 1e9
        density = original_obj.acoustic_shader.density if original_obj.acoustic_shader else 1000.0
        damping = original_obj.acoustic_shader.damping if original_obj.acoustic_shader else 0.02

        # Get fragment states after fracture (at fracture_end)
        fragment_velocities = []
        fragment_angular_velocities = []
        fragment_data_list = []

        for frag in fragments:
            traj = fragment_trajectories.get(frag.idx)
            if traj is not None:
                # Get velocity just after fracture (at fracture_end)
                vel = traj.get_velocity(fracture_end + 10)
                ang_vel = traj.get_angular_velocity(fracture_end + 10)
                fragment_velocities.append(vel)
                fragment_angular_velocities.append(ang_vel)

                # Get fragment geometry at fracture_end
                vertices = traj.get_vertices(fracture_end)
                normals = traj.get_normals(fracture_end)
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
                    mass=mesh.mass if not np.isnan(mesh.mass) else 0.001,
                    volume=mesh.volume if not np.isnan(mesh.volume) else 0.0001,
                    center_of_mass=mesh.center_mass if mesh.center_mass is not None else np.mean(vertices, axis=0),
                    inertia_tensor=mesh.moment_inertia if mesh.moment_inertia is not None else np.eye(3) * 0.001,
                    parent_obj_idx=obj_idx,
                    is_shard=True,
                    fracture_frame=fracture_begin
                )
                fragment_data_list.append(fragment_data)

        # Compute fracture energy
        fracture_energy = self._compute_fracture_energy(original_obj, fragments, pre_velocity, fragment_velocities, fracture_collisions, fracture_forces, fracture_begin)

        # Classify fracture type
        fracture_type = self._classify_fracture_type(original_obj, fragments, pre_velocity, fragment_velocities, fracture_begin, fracture_end, fracture_energy, fracture_collisions)

        # Estimate crack length
        crack_length = self._estimate_crack_length(original_obj, fragments, fracture_begin)

        # Create fracture event
        event = FractureEvent(
            fracture_type=fracture_type,
            frame=fracture_begin,
            original_obj_idx=obj_idx,
            original_obj_name=original_obj.name,
            fragment_indices=fragment_indices,
            pre_fracture_velocity=pre_velocity,
            pre_fracture_angular_velocity=pre_angular_velocity,
            pre_fracture_force=pre_force,
            pre_fracture_stress=np.zeros(6),
            fragment_velocities=fragment_velocities,
            fragment_angular_velocities=fragment_angular_velocities,
            fracture_energy=fracture_energy,
            crack_velocity=self._estimate_crack_velocity(fracture_begin, fracture_end, original_obj),
            crack_duration=fracture_end - fracture_begin if fracture_end else 0.01,
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
        event.save(f"{self.fracture_dir}/event_{obj_idx}_{fracture_begin:.6f}.pkl")

        debug_print(f"Detected {fracture_type.value} fracture for {original_obj.name} "
                   f"from frame {fracture_begin:.6f} to {fracture_end if fracture_end else fracture_begin:.6f}, "
                   f"energy: {fracture_energy:.6f}J")

        return [event]

    def _find_fracture_moments_geometric(self, original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], fracture_sample_approx: float, fragments: List[Any], original_obj: Any) -> Optional[Dict[str, float]]:
        """
        Find exact fracture begin and end moments using geometric analysis.
        
        Uses bounding box alignment and trajectory interpolation to find:
        - Fracture begin: When the original object's bounding box starts to diverge
        - Fracture end: When all shard bounding boxes fit within the original bounding box
        
        Returns:
            Dict with 'begin' and 'end' sample indices
        """
#        # Get original object's mesh at fracture_approx for bounding box
#        original_vertices = original_trajectory.get_vertices(fracture_sample_approx)
#        original_bbox = self._compute_bounding_box(original_vertices)
        
        # Get shard vertices at shard_start frame
        shard_bboxes = {}
        for frag in fragments:
            traj = fragment_trajectories.get(frag.idx)
            if traj is not None:
                vertices = traj.get_vertices(fracture_frame_approx)
                shard_bboxes[frag.idx] = self._compute_bounding_box(vertices)
        
        # Search range for fracture moments
        search_start = max(0, fracture_sample_approx - int(self.sample_rate / self.sfps))
        search_end = min(original_trajectory.get_x()[-1], fracture_sample_approx)
        
        # Sample the time range for analysis
        sample_times = np.arange(search_start, search_end, max(1, int(self.sampling_interval * self.sample_rate)))
        
        # For each sample, compute the divergence between original and shard bounding boxes
        divergence_scores = []
        
        for sample_time in sample_times:
            # Get original vertices at this time
            orig_verts = original_trajectory.get_vertices(sample_time)
            orig_bbox = self._compute_bounding_box(orig_verts)
            
            # Compute shard positions at this time
            shard_positions = {}
            for frag_idx, traj in fragment_trajectories.items():
                if traj is not None:
                    verts = traj.get_vertices(sample_time)
                    shard_bboxes[frag_idx] = self._compute_bounding_box(verts)
            
            # Compute divergence score
            divergence = self._compute_bbox_divergence(orig_bbox, shard_bboxes)
            divergence_scores.append((sample_time, divergence))
        
        # Find fracture begin: point where divergence starts increasing rapidly
        divergence_scores = np.array(divergence_scores, dtype=object)
        times = divergence_scores[:, 0].astype(float)
        scores = divergence_scores[:, 1].astype(float)

        # Find the moment where divergence starts increasing
        # Use derivative analysis
        if len(scores) > 3:
            # Smooth the scores
            from scipy.ndimage import gaussian_filter1d
            smoothed_scores = gaussian_filter1d(scores, sigma=2)
            
            # Compute derivative
            derivative = np.gradient(smoothed_scores, times)
            
            # Find where derivative exceeds threshold (begin of divergence)
            threshold = 0.01 * np.max(smoothed_scores) if np.max(smoothed_scores) > 0 else 0.001
            begin_candidates = np.where(derivative > threshold)[0]
            
            if len(begin_candidates) > 0:
                fracture_begin = times[begin_candidates[0]]
            else:
                # Fallback: use the approximate fracture sample
                fracture_begin = fracture_sample_approx
        else:
            fracture_begin = fracture_sample_approx - int(0.1 * self.sample_rate)
        
        # Find fracture end: when shards are fully separated and original object is gone
        # Look for the moment when the original object's bounding box no longer
        # contains the shard bounding boxes
        
        fracture_end = None
        for sample_time, divergence in reversed(list(zip(times, scores))):
            if divergence > 0.5:  # Shards are significantly separated
                fracture_end = sample_time
                break
        
        if fracture_end is None:
            fracture_end = fracture_sample_approx
        
        # Refine using sub-sample interpolation
        fracture_begin = self._refine_fracture_moment(original_trajectory=original_trajectory, fragment_trajectories=fragment_trajectories, approx_begin=fracture_begin, approx_end=fracture_end, fragments=fragments)
        return {'begin': fracture_begin, 'end': fracture_end}

    def _compute_bounding_box(self, vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute bounding box of vertices.
        
        Returns:
            Dict with 'min', 'max', 'center', 'extents'
        """
        if len(vertices) == 0:
            return {'min': np.zeros(3), 'max': np.zeros(3), 'center': np.zeros(3), 'extents': np.zeros(3)}

        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        extents = max_coords - min_coords
        
        return {'min': min_coords, 'max': max_coords, 'center': center, 'extents': extents}

    def _compute_bbox_divergence(self, original_bbox: Dict[str, np.ndarray], shard_bboxes: Dict[int, Dict[str, np.ndarray]] ) -> float:
        """
        Compute divergence score between original and shard bounding boxes.

        Score measures how much the shards have separated from the original.
        Returns 0 when shards are contained in the original, 1 when fully separated.
        """
        if len(shard_bboxes) == 0:
            return 0.0

        # Compute the union of shard bounding boxes
        shard_min = np.array([float('inf'), float('inf'), float('inf')])
        shard_max = np.array([float('-inf'), float('-inf'), float('-inf')])

        for bbox in shard_bboxes.values():
            shard_min = np.minimum(shard_min, bbox['min'])
            shard_max = np.maximum(shard_max, bbox['max'])

        shard_center = (shard_min + shard_max) / 2
        shard_extents = shard_max - shard_min

        # Original extents
        orig_center = original_bbox['center']
        orig_extents = original_bbox['extents']

        # Compute center distance relative to extents
        center_distance = np.linalg.norm(shard_center - orig_center)
        extents_scale = np.linalg.norm(orig_extents) + 1e-10

        # Compute containment: how much of shard extents are outside original
        overlap = np.minimum(shard_max, original_bbox['max']) - np.maximum(shard_min, original_bbox['min'])
        overlap = np.maximum(overlap, 0) # abs(overlap) ??
        overlap_volume = np.prod(overlap)

        shard_volume = np.prod(shard_extents) if np.all(shard_extents > 0) else 1e-10
        original_volume = np.prod(orig_extents) if np.all(orig_extents > 0) else 1e-10

        # Divergence score components
        # 1. Center displacement
        center_divergence = center_distance / (extents_scale + 1e-10)
        center_divergence = np.clip(center_divergence, 0, 1)

        # 2. Containment loss
        if shard_volume > 0:
            containment_loss = 1 - (overlap_volume / shard_volume)
            containment_loss = np.clip(containment_loss, 0, 1)
        else:
            containment_loss = 0

        # 3. Extents change
        extents_change = np.linalg.norm(shard_extents - orig_extents) / (extents_scale + 1e-10)
        extents_change = np.clip(extents_change, 0, 1)

        # Combined score
        divergence = 0.5 * center_divergence + 0.3 * containment_loss + 0.2 * extents_change

        return float(np.clip(divergence, 0, 1))

    def _refine_fracture_moment(self, original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], approx_begin: float, approx_end: float, fragments: List[Any]) -> float:
        """
        Refine the fracture begin moment using sub-sample interpolation.
        """
        # Get sample range around approximate begin
        search_samples = 10
        start_sample = max(0, int(approx_begin - search_samples))
        end_sample = int(approx_begin + search_samples)
        
        # Sample at high resolution
        sample_times = np.linspace(start_sample, end_sample, 100)
        
        divergence_scores = []
        for sample_time in sample_times:
            orig_verts = original_trajectory.get_vertices(sample_time)
            orig_bbox = self._compute_bounding_box(orig_verts)
            
            shard_bboxes = {}
            for frag in fragments:
                traj = fragment_trajectories.get(frag.idx)
                if traj is not None:
                    verts = traj.get_vertices(sample_time)
                    shard_bboxes[frag.idx] = self._compute_bounding_box(verts)
            
            divergence = self._compute_bbox_divergence(orig_bbox, shard_bboxes)
            divergence_scores.append((sample_time, divergence))
        
        # Find the point where divergence starts increasing
        times = np.array([s[0] for s in divergence_scores])
        scores = np.array([s[1] for s in divergence_scores])
        
        # Find the inflection point
        from scipy.signal import find_peaks
        gradient = np.gradient(scores, times)
        
        # Find where gradient starts becoming positive
        for i in range(1, len(gradient)):
            if gradient[i] > 0.001 and gradient[i-1] < 0.001:
                return times[i]
        
        return approx_begin

    def _estimate_crack_velocity(self, fracture_begin: float, fracture_end: float, original_obj: Any) -> float:
        """
        Estimate crack velocity from fracture duration and object size.
        """
        if fracture_end is None or fracture_end <= fracture_begin:
            return 500.0  # Default crack velocity
        
        # Get object size
        try:
            config = self.entity_manager.get('config')
            vertices, _, _ = _load_mesh(original_obj, int(fracture_begin * self.sfps / self.sample_rate))
            size = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        except:
            size = 0.1
        
        # Crack velocity = size / duration
        duration = fracture_end - fracture_begin
        if duration > 0:
            velocity = size / duration
            return min(max(velocity, 100), 2000)  # Clamp to reasonable range
        
        return 500.0

    def _get_force_at_time(self, force_sequences: List[ForceDataSequence], obj_idx: int, time: float) -> np.ndarray:
        """Get force vector at a specific time from force data sequences."""
        for force_seq in force_sequences:
            if force_seq.obj_idx == obj_idx or force_seq.other_obj_idx == obj_idx:
                try:
                    return force_seq.get_normal_force(time)
                except:
                    pass
        return np.zeros(3)

    def _classify_fracture_type(self, original_obj: Any, fragments: List[Any], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], fracture_begin: float, fracture_end: float, fracture_energy: float, collisions: List[CollisionData]) -> FractureType:
        """Classify the fracture type based on trajectory and force data."""
        n_fragments = len(fragments)

        if n_fragments >= 3:
            avg_velocity = np.mean([np.linalg.norm(v) for v in fragment_velocities])
            pre_speed = np.linalg.norm(pre_velocity)

            if avg_velocity > pre_speed * 1.5 and avg_velocity > 1.0:
                return FractureType.SHATTER

            if fracture_end - fracture_begin < (0.100 * self.sample_rate): # Failure Speed < 100ms
                return FractureType.SHATTER

        if n_fragments == 2 and len(fragment_velocities) >= 2:
            rel_velocity = np.linalg.norm(fragment_velocities[0] - fragment_velocities[1])
            pre_speed = np.linalg.norm(pre_velocity)

            if rel_velocity > pre_speed * 0.5 or rel_velocity > 2.0:
                return FractureType.SNAP

            if (0.005 * self.sample_rate) > fracture_end - fracture_begin < (0.030 * self.sample_rate): # Failure Speed > 5ms and < 30ms
                return FractureType.SNAP

        if n_fragments <= 2:
#            if len(collisions) > 0:
#                total_energy = 0
#                for coll in collisions:
#                    if hasattr(coll, 'impulse_range') and coll.impulse_range:
#                        total_energy += coll.impulse_range * 0.001 # ToDo: compute energy in place of 0.001
            if fracture_energy >= 0.01:
                return FractureType.CRACK
            if fracture_end - fracture_begin < (0.005 * self.sample_rate): # Failure Speed < 5ms
                return FractureType.CRACK

        return FractureType.CRACK

    def _compute_fracture_energy(self, original_obj: Any, fragments: List[Any], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], collisions: List[CollisionData], forces: List[ForceDataSequence], fracture_moment: float) -> float:
        """Compute the energy released during fracture."""
        total_energy = 0.0

        # 1. Kinetic energy change
        if len(fragment_velocities) > 0:
            masses = []
            for frag in fragments:
                if frag.acoustic_shader and hasattr(frag.acoustic_shader, 'density'):
                    try:
                        traj = self.entity_manager.get('trajectories')
                        for t in traj.values():
                            if hasattr(t, 'obj_idx') and t.obj_idx == frag.idx:
                                vertices = t.get_vertices(fracture_moment)
                                faces = t.get_faces()
                                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                                mass = mesh.volume * frag.acoustic_shader.density
                                masses.append(mass if not np.isnan(mass) else 0.001)
                                break
                    except:
                        masses.append(0.001)
                else:
                    masses.append(0.001)

            kinetic_energy = 0.0
            for i, vel in enumerate(fragment_velocities):
                if i < len(masses):
                    kinetic_energy += 0.5 * masses[i] * np.linalg.norm(vel)**2

            pre_mass = sum(masses) if masses else 0.001
            pre_kinetic = 0.5 * pre_mass * np.linalg.norm(pre_velocity)**2
            total_energy += max(0, kinetic_energy - pre_kinetic)

        # 2. Collision energy
        collision_energy = 0.0
        for coll in collisions:
            if hasattr(coll, 'distances') and coll.distances is not None:
                if isinstance(coll.distances, np.ndarray):
                    penetration_energy = np.sum(np.abs(coll.distances)) * 10.0
                    collision_energy += penetration_energy
        total_energy += collision_energy

        # 3. Stored elastic energy
        if hasattr(original_obj.acoustic_shader, 'failure_stress'):
            failure_stress = original_obj.acoustic_shader.failure_stress
        else:
            failure_stress = 1e6

        young_modulus = original_obj.acoustic_shader.young_modulus if original_obj.acoustic_shader else 1e9
        stressed_volume = 0.0001  # ToDo: how to compute the stressed volume?
        elastic_energy = (failure_stress**2 * stressed_volume) / (2 * young_modulus)
        total_energy += elastic_energy

        return max(total_energy, 0.01)

    def _estimate_crack_length(self, original_obj: Any, fragments: List[Any], fracture_moment: float) -> float:
        """Estimate the crack length from fragment geometry."""
        try:
            from pbrAudioCommon import _load_mesh
            vertices, _, faces = _load_mesh(original_obj, int(fracture_moment * self.sfps / self.sample_rate))

            fragment_vertices = []
            for frag in fragments:
                traj = self.entity_manager.get('trajectories')
                for t in traj.values():
                    if hasattr(t, 'obj_idx') and t.obj_idx == frag.idx:
                        verts = t.get_vertices(fracture_moment)
                        fragment_vertices.append(verts)
                        break

            if len(fragment_vertices) < 2:
                return 0.1

            original_center = np.mean(vertices, axis=0)
            fragment_centers = [np.mean(verts, axis=0) for verts in fragment_vertices]
            distances = [np.linalg.norm(c - original_center) for c in fragment_centers]

            return max(distances) * 0.5

        except Exception as e:
            debug_print(f"Error estimating crack length: {e}")
            return 0.1
