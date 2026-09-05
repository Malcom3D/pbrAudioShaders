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
import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar

from pbrAudioCommon import EntityManager, _load_mesh
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from physicsSolver import TrajectoryData, ForceDataSequence, CollisionData

from .fracture_data import FractureEvent, FractureType, FragmentData


@dataclass
class FractureDetector:
    """
    Detects and classifies fracture events from trajectory data using geometric methods.
    
    Extended to handle detailed fracture event sequences with:
    - Forward trajectory analysis for original object
    - Reverse trajectory analysis for shard objects
    - Accurate fracture begin detection via bounding box alignment
    - Temporal fracture sequence classification
    - Collision energy computation from ForceDataSequence
    """
    
    entity_manager: EntityManager
    
    # Detection parameters
    position_tolerance: float = 0.001  # Position matching tolerance (meters)
    velocity_threshold: float = 0.01   # Minimum velocity change for fracture detection (m/s)
    energy_threshold: float = 0.01     # Minimum energy release for fracture (J)
    time_window: float = 0.02          # Time window for fracture detection (seconds)
    sampling_interval: float = 0.001   # Sampling interval for search (seconds)
    bbox_alignment_threshold: float = 0.05  # Threshold for bounding box alignment (fraction)
    
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
        
        # Cache for computed trajectories
        self._trajectory_cache = {}
    
    def detect_fracture_events(self, obj_idx: int, fragment_indices: List[int]) -> List[FractureEvent]:
        """
        Detect fracture events with detailed sequence analysis.
        
        Parameters:
        -----------
        obj_idx : int
            Index of the original object before fracture
        fragment_indices : List[int]
            Indices of the fragments after fracture

        Returns:
        --------
        List[FractureEvent]
            Detected fracture events in temporal sequence
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

        # Convert frames to samples
        fracture_sample_approx = fracture_frame_approx * self.sample_rate / self.sfps

        # Find the exact fracture moments using enhanced geometric analysis
        fracture_moments = self._find_fracture_moments_geometric_enhanced(original_trajectory=original_trajectory, fragment_trajectories=fragment_trajectories, fracture_sample_approx=fracture_sample_approx, fragments=fragments, original_obj=original_obj)

        if fracture_moments is None or len(fracture_moments) == 0:
            debug_print(f"Could not determine fracture moments for {original_obj.name}")
            return []

        # Get the fracture begin and end moments
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

        # Compute fracture energy from collision data
        fracture_energy = self._compute_fracture_energy_enhanced(original_obj=original_obj, fragments=fragments, pre_velocity=pre_velocity, fragment_velocities=fragment_velocities, fracture_collisions=fracture_collisions, fracture_forces=fracture_forces, fracture_begin=fracture_begin, fracture_end=fracture_end)

        # Classify fracture type with temporal sequence analysis
        fracture_events = self._classify_fracture_sequence(original_obj=original_obj, fragments=fragments, original_trajectory=original_trajectory, fragment_trajectories=fragment_trajectories, pre_velocity=pre_velocity, fragment_velocities=fragment_velocities, fracture_begin=fracture_begin, fracture_end=fracture_end, fracture_energy=fracture_energy, fracture_collisions=fracture_collisions)

        # Estimate crack length
        crack_length = self._estimate_crack_length(original_obj, fragments, fracture_begin)

        # Create fracture events
        events = []
        for i, (fracture_type, event_data) in enumerate(fracture_events):
            event = FractureEvent(
                fracture_type=fracture_type,
                frame=event_data.get('frame', fracture_begin),
                original_obj_idx=obj_idx,
                original_obj_name=original_obj.name,
                fragment_indices=event_data.get('fragment_indices', fragment_indices),
                pre_fracture_velocity=pre_velocity,
                pre_fracture_angular_velocity=pre_angular_velocity,
                pre_fracture_force=pre_force,
                pre_fracture_stress=np.zeros(6),
                fragment_velocities=fragment_velocities,
                fragment_angular_velocities=fragment_angular_velocities,
                fracture_energy=event_data.get('energy', fracture_energy),
                crack_velocity=self._estimate_crack_velocity(event_data.get('begin', fracture_begin), event_data.get('end', fracture_end), original_obj),
                crack_duration=event_data.get('duration', fracture_end - fracture_begin),
                crack_length=crack_length,
                young_modulus=young_modulus,
                density=density,
                damping=damping,
                failure_stress=original_obj.acoustic_shader.failure_stress if hasattr(original_obj.acoustic_shader, 'failure_stress') else 1e6,
                fragment_data=fragment_data_list,
                collision_data=event_data.get('collisions', fracture_collisions),
                force_data=event_data.get('forces', fracture_forces)
            )

            # Save the event
            event.save(f"{self.fracture_dir}/event_{obj_idx}_{event.frame:.6f}_{i:02d}.pkl")
            events.append(event)

            debug_print(f"Detected {fracture_type.value} fracture for {original_obj.name} at frame {event.frame}, duration: {event.crack_duration}s, energy: {event.fracture_energy}J")

        return events

    def _find_fracture_moments_geometric_enhanced(self, original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], fracture_sample_approx: float, fragments: List[Any], original_obj: Any) -> Optional[Dict[str, float]]:
        """
        Find exact fracture begin and end moments using enhanced trajectory analysis.
        
        - Original object trajectory: forward from fracture_frame-1 to fracture_frame
        - Shard trajectories: reverse from fracture_frame to fracture_frame-1
        - Uses bounding box alignment to find exact fracture begin
        """
        # Get the fracture frame from config
        fracture_frame = original_obj.fractured
        if fracture_frame is False:
            return None
        
        # Convert to samples
        fracture_sample = fracture_frame * self.sample_rate / self.sfps
        
        # Compute trajectories for original object (forward direction)
        # From fracture_frame-1 to fracture_frame
        debug_print(f"Computing forward trajectory for original object from frame {fracture_frame-1} to {fracture_frame}")
        
        original_traj_forward = self._compute_forward_trajectory(trajectory=original_trajectory, start_sample=fracture_sample - int(self.sample_rate / self.sfps), end_sample=fracture_sample, num_samples=50)
        
        # Compute trajectories for shard objects (reverse direction)
        # From fracture_frame to fracture_frame-1
        debug_print(f"Computing reverse trajectories for {len(fragments)} shard objects")
        
        fragment_traj_reverse = {}
        for frag in fragments:
            traj = fragment_trajectories.get(frag.idx)
            if traj is not None:
                fragment_traj_reverse[frag.idx] = self._compute_reverse_trajectory(trajectory=traj, start_sample=fracture_sample, end_sample=fracture_sample - int(self.sample_rate / self.sfps), num_samples=50)
        
        # Find fracture begin using bounding box alignment
        fracture_begin = self._find_fracture_begin_alignment(original_traj_forward=original_traj_forward, fragment_traj_reverse=fragment_traj_reverse, fragments=fragments, fracture_sample=fracture_sample)
        
        if fracture_begin is None:
            debug_print("Could not find fracture begin via alignment")
            fracture_begin = fracture_sample - int(0.5 * self.sample_rate / self.sfps)
        
        # Find fracture end (when all shards are fully separated)
        fracture_end = self._find_fracture_end(original_trajectory=original_trajectory, fragment_trajectories=fragment_trajectories, fragments=fragments, fracture_begin=fracture_begin, fracture_sample=fracture_sample)
        
        if fracture_end is None:
            fracture_end = fracture_sample + int(0.5 * self.sample_rate / self.sfps)
        
        return {'begin': fracture_begin, 'end': fracture_end}
    
    def _compute_forward_trajectory(self, trajectory: TrajectoryData, start_sample: float, end_sample: float, num_samples: int = 50) -> Dict[str, Any]:
        """
        Compute trajectory points in forward direction between two samples.
        
        Returns:
            Dict with 'times', 'positions', 'bboxes', 'centers'
        """
        times = np.linspace(start_sample, end_sample, num_samples)
        positions = np.array([trajectory.get_position(t) for t in times])
        bboxes = []
        centers = []
        
        for t in times:
            vertices = trajectory.get_vertices(t)
            bbox = self._compute_bounding_box(vertices)
            bboxes.append(bbox)
            centers.append(bbox['center'])
        
        return {
            'times': times,
            'positions': positions,
            'bboxes': bboxes,
            'centers': np.array(centers)
        }
    
    def _compute_reverse_trajectory(self, trajectory: TrajectoryData, start_sample: float, end_sample: float, num_samples: int = 50) -> Dict[str, Any]:
        """
        Compute trajectory points in reverse direction between two samples.
        
        Returns:
            Dict with 'times', 'positions', 'bboxes', 'centers'
        """
        # Reverse the direction: go from end to start
        times = np.linspace(start_sample, end_sample, num_samples)
        positions = np.array([trajectory.get_position(t) for t in times])
        bboxes = []
        centers = []
        
        for t in times:
            vertices = trajectory.get_vertices(t)
            bbox = self._compute_bounding_box(vertices)
            bboxes.append(bbox)
            centers.append(bbox['center'])
        
        # Return in reverse order for easy alignment
        return {
            'times': times,
            'positions': positions,
            'bboxes': bboxes[::-1],  # Reverse to go from fracture to pre-fracture
            'centers': np.array(centers[::-1])
        }
    
    def _find_fracture_begin_alignment(self, original_traj_forward: Dict[str, Any], fragment_traj_reverse: Dict[int, Dict[str, Any]], fragments: List[Any], fracture_sample: float) -> Optional[float]:
        """
        Find fracture begin by aligning shard bounding boxes with original bounding box.
        
        Returns:
            Sample index of fracture begin
        """
        if not fragment_traj_reverse:
            return None
        
        # Get original bbox at fracture frame (end of forward trajectory)
        original_bbox = original_traj_forward['bboxes'][-1]
        
        # For each shard, find where its bbox aligns with the original
        alignment_scores = []
        
        for frag in fragments:
            if frag.idx not in fragment_traj_reverse:
                continue
            
            reverse_traj = fragment_traj_reverse[frag.idx]
            
            for i, bbox in enumerate(reverse_traj['bboxes']):
                # Compute alignment score between shard bbox and original bbox
                score = self._compute_bbox_alignment(original_bbox, bbox)
                alignment_scores.append({
                    'idx': i,
                    'time': reverse_traj['times'][i],
                    'score': score,
                    'fragment': frag.idx
                })
        
        if not alignment_scores:
            return None
        
        # Find the best alignment (minimum score = best)
        best_match = min(alignment_scores, key=lambda x: x['score'])
        
        # If alignment is good enough, return the time
        if best_match['score'] < self.bbox_alignment_threshold:
            return best_match['time']
        
        # If no good match, try weighted average of top matches
        sorted_matches = sorted(alignment_scores, key=lambda x: x['score'])
        top_matches = sorted_matches[:len(fragments)]
        
        if top_matches:
            # Weighted average by inverse score
            weights = [1.0 / (m['score'] + 0.001) for m in top_matches]
            total_weight = sum(weights)
            weighted_time = sum(m['time'] * w for m, w in zip(top_matches, weights)) / total_weight
            return weighted_time
        
        return None
    
    def _compute_bbox_alignment(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Compute alignment score between two bounding boxes.
        
        Lower score = better alignment.
        
        The score combines:
        - Center distance (normalized)
        - Size ratio difference
        - Overlap ratio
        """
        # Center distance
        center_dist = np.linalg.norm(bbox1['center'] - bbox2['center'])
        extents1 = bbox1['extents']
        extents2 = bbox2['extents']
        
        # Normalize center distance by extents
        avg_extent = np.linalg.norm(extents1) + np.linalg.norm(extents2)
        if avg_extent > 0:
            norm_center_dist = center_dist / avg_extent
        else:
            norm_center_dist = 1.0
        
        # Size ratio
        size1 = np.prod(extents1)
        size2 = np.prod(extents2)
        if size1 > 0 and size2 > 0:
            size_ratio = min(size1, size2) / max(size1, size2)
            size_score = 1.0 - size_ratio
        else:
            size_score = 1.0
        
        # Extents ratio difference
        extents_ratio1 = extents1 / (np.linalg.norm(extents1) + 0.001)
        extents_ratio2 = extents2 / (np.linalg.norm(extents2) + 0.001)
        extents_diff = np.linalg.norm(extents_ratio1 - extents_ratio2)
        
        # Combined score
        score = 0.4 * norm_center_dist + 0.3 * size_score + 0.3 * extents_diff
        
        return float(score)
    
    def _find_fracture_end(self, original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], fragments: List[Any], fracture_begin: float, fracture_sample: float) -> Optional[float]:
        """
        Find fracture end when shards are fully separated from original.
        """
        # Search forward from fracture_begin
        search_end = fracture_begin + int(1.0 * self.sample_rate)  # 1 second max
        
        # Sample times
        times = np.linspace(fracture_begin, search_end, 100)
        
        # Compute divergence scores
        divergence_scores = []
        
        for t in times:
            # Original bbox at this time
            try:
                orig_verts = original_trajectory.get_vertices(t)
                orig_bbox = self._compute_bounding_box(orig_verts)
            except:
                continue
            
            # Shard bboxes at this time
            shard_bboxes = {}
            for frag in fragments:
                traj = fragment_trajectories.get(frag.idx)
                if traj is not None:
                    try:
                        verts = traj.get_vertices(t)
                        shard_bboxes[frag.idx] = self._compute_bounding_box(verts)
                    except:
                        pass
            
            if not shard_bboxes:
                continue
            
            # Compute divergence
            divergence = self._compute_bbox_divergence(orig_bbox, shard_bboxes)
            divergence_scores.append((t, divergence))
        
        if not divergence_scores:
            return fracture_sample + int(0.5 * self.sample_rate / self.sfps)
        
        # Find where divergence stabilizes (high divergence)
        stable_threshold = 0.6
        
        # Sort by time
        divergence_scores = sorted(divergence_scores, key=lambda x: x[0])
        times = np.array([s[0] for s in divergence_scores])
        scores = np.array([s[1] for s in divergence_scores])
        
        # Find first point where divergence exceeds threshold
        # and stays above threshold for a while
        for i in range(len(scores)):
            if scores[i] > stable_threshold:
                # Check next few points
                if i + 3 < len(scores) and all(scores[i:i+3] > stable_threshold):
                    return times[i]
        
        # Fallback: use the point of maximum divergence
        max_idx = np.argmax(scores)
        if scores[max_idx] > 0.3:
            return times[max_idx]
        
        return None
    
    def _compute_bbox_divergence(self, original_bbox: Dict[str, np.ndarray], shard_bboxes: Dict[int, Dict[str, np.ndarray]]) -> float:
        """
        Compute divergence score between original and shard bounding boxes.
        
        Enhanced to handle trajectory analysis for both forward and reverse directions.
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
        overlap = np.maximum(overlap, 0)
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

        # 4. Directional divergence (for reverse trajectory analysis)
        # Check if shards are moving away from original
        directional = 0.0
        if len(shard_bboxes) > 1:
            # Compute average direction from original to shard centers
            directions = []
            for bbox in shard_bboxes.values():
                dir_vec = bbox['center'] - orig_center
                if np.linalg.norm(dir_vec) > 0:
                    directions.append(dir_vec / np.linalg.norm(dir_vec))
            
            if directions:
                avg_dir = np.mean(directions, axis=0)
                directional = np.linalg.norm(avg_dir) * 0.5

        # Combined score with directional component
        divergence = (0.35 * center_divergence +
                      0.25 * containment_loss +
                      0.20 * extents_change +
                      0.20 * directional)

        return float(np.clip(divergence, 0, 1))

    def _classify_fracture_sequence(self, original_obj: Any, fragments: List[Any], original_trajectory: TrajectoryData, fragment_trajectories: Dict[int, TrajectoryData], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], fracture_begin: float, fracture_end: float, fracture_energy: float, fracture_collisions: List[CollisionData]) -> List[Tuple[FractureType, Dict[str, Any]]]:
        """
        Classify fracture sequence over time.
        
        Analyzes the temporal progression of fracture events, where shards
        may fracture and diverge at different times.
        
        Returns:
            List of (FractureType, event_data) tuples in temporal sequence
        """
        n_fragments = len(fragments)
        events = []
        
        # Get shard trajectories
        shard_trajectories = []
        for frag in fragments:
            traj = fragment_trajectories.get(frag.idx)
            if traj is not None:
                shard_trajectories.append(traj)
        
        # Analyze divergence times for each shard
        divergence_times = self._compute_shard_divergence_times(
            original_trajectory=original_trajectory,
            shard_trajectories=shard_trajectories,
            fragments=fragments,
            fracture_begin=fracture_begin,
            fracture_end=fracture_end
        )
        
        # Sort shards by divergence time
        sorted_shards = sorted(divergence_times, key=lambda x: x['time'])
        
        # Group shards by divergence time clusters
        clusters = self._cluster_divergence_times(sorted_shards)
        
        # Classify each cluster
        for cluster in clusters:
            cluster_shards = [s['idx'] for s in cluster]
            cluster_time = np.mean([s['time'] for s in cluster])
            cluster_velocities = []
            
            for shard_idx in cluster_shards:
                # Find corresponding fragment
                for i, frag in enumerate(fragments):
                    if frag.idx == shard_idx:
                        cluster_velocities.append(fragment_velocities[i])
                        break
            
            # Calculate average velocity for this cluster
            avg_velocity = np.mean([np.linalg.norm(v) for v in cluster_velocities]) if cluster_velocities else 0
            pre_speed = np.linalg.norm(pre_velocity) if pre_velocity is not None else 0
            
            # Classify based on cluster characteristics
            n_cluster = len(cluster_shards)
            
            if n_cluster >= 3 and avg_velocity > pre_speed * 1.5:
                fracture_type = FractureType.SHATTER
            elif n_cluster == 2 and avg_velocity > pre_speed * 0.5:
                fracture_type = FractureType.SNAP
            else:
                fracture_type = FractureType.CRACK
            
            # Get collisions for this cluster
            cluster_collisions = []
            for coll in fracture_collisions:
                if coll.obj1_idx in cluster_shards or coll.obj2_idx in cluster_shards:
                    if abs(coll.frame - cluster_time) < 0.01:
                        cluster_collisions.append(coll)
            
            # Estimate energy for this cluster
            cluster_energy = fracture_energy * (n_cluster / n_fragments)
            
            events.append((
                fracture_type,
                {
                    'frame': cluster_time,
                    'begin': cluster_time - 0.005,
                    'end': cluster_time + 0.01,
                    'duration': 0.015,
                    'fragment_indices': cluster_shards,
                    'energy': cluster_energy,
                    'collisions': cluster_collisions
                }
            ))
        
        # If no events were created, create a single default event
        if not events:
            # Determine fracture type
            avg_velocity = np.mean([np.linalg.norm(v) for v in fragment_velocities]) if fragment_velocities else 0
            pre_speed = np.linalg.norm(pre_velocity) if pre_velocity is not None else 0
            
            if n_fragments >= 3 and avg_velocity > pre_speed * 1.5:
                fracture_type = FractureType.SHATTER
            elif n_fragments == 2 and avg_velocity > pre_speed * 0.5:
                fracture_type = FractureType.SNAP
            else:
                fracture_type = FractureType.CRACK
            
            events.append((
                fracture_type,
                {
                    'frame': fracture_begin,
                    'begin': fracture_begin,
                    'end': fracture_end,
                    'duration': fracture_end - fracture_begin,
                    'fragment_indices': [f.idx for f in fragments],
                    'energy': fracture_energy,
                    'collisions': fracture_collisions
                }
            ))
        
        return events
    
    def _compute_shard_divergence_times(self, original_trajectory: TrajectoryData, shard_trajectories: List[TrajectoryData], fragments: List[Any], fracture_begin: float, fracture_end: float) -> List[Dict[str, Any]]:
        """
        Compute divergence time for each shard.
        
        Returns:
            List of dicts with 'idx', 'time', 'score'
        """
        divergence_times = []
        
        for i, traj in enumerate(shard_trajectories):
            if traj is None:
                continue
            
            # Sample times between fracture_begin and fracture_end
            times = np.linspace(fracture_begin, fracture_end, 50)
            
            # Track divergence score over time
            scores = []
            
            for t in times:
                # Original bbox at this time
                try:
                    orig_verts = original_trajectory.get_vertices(t)
                    orig_bbox = self._compute_bounding_box(orig_verts)
                except:
                    continue
                
                # Shard bbox at this time
                try:
                    shard_verts = traj.get_vertices(t)
                    shard_bbox = self._compute_bounding_box(shard_verts)
                except:
                    continue
                
                # Compute divergence
                score = self._compute_single_shard_divergence(orig_bbox, shard_bbox)
                scores.append((t, score))
            
            if not scores:
                continue
            
            # Find the time when divergence crosses a threshold
            threshold = 0.3
            divergence_time = None
            
            for t, score in scores:
                if score > threshold:
                    divergence_time = t
                    break
            
            if divergence_time is None:
                divergence_time = fracture_begin + (fracture_end - fracture_begin) / 2
            
            divergence_times.append({
                'idx': fragments[i].idx,
                'time': divergence_time,
                'score': max([s[1] for s in scores]) if scores else 0.5
            })
        
        return divergence_times
    
    def _compute_single_shard_divergence(self, original_bbox: Dict[str, np.ndarray], shard_bbox: Dict[str, np.ndarray]) -> float:
        """
        Compute divergence between original bbox and a single shard bbox.
        """
        # Center distance
        center_dist = np.linalg.norm(shard_bbox['center'] - original_bbox['center'])
        extents_scale = np.linalg.norm(original_bbox['extents']) + 1e-10
        center_divergence = center_dist / extents_scale
        
        # Containment
        shard_extents = shard_bbox['extents']
        orig_extents = original_bbox['extents']
        
        overlap_min = np.maximum(shard_bbox['min'], original_bbox['min'])
        overlap_max = np.minimum(shard_bbox['max'], original_bbox['max'])
        overlap = overlap_max - overlap_min
        overlap = np.maximum(overlap, 0)
        
        shard_volume = np.prod(shard_extents) if np.all(shard_extents > 0) else 1e-10
        overlap_volume = np.prod(overlap)
        
        if shard_volume > 0:
            containment_loss = 1 - (overlap_volume / shard_volume)
        else:
            containment_loss = 1.0
        
        # Combined score
        divergence = 0.6 * center_divergence + 0.4 * containment_loss
        return float(np.clip(divergence, 0, 1))
    
    def _cluster_divergence_times(self, sorted_shards: List[Dict[str, Any]], time_threshold: float = 0.005) -> List[List[Dict[str, Any]]]:
        """
        Cluster shards by divergence time proximity.
        """
        if not sorted_shards:
            return []
        
        clusters = []
        current_cluster = [sorted_shards[0]]
        
        for i in range(1, len(sorted_shards)):
            time_diff = abs(sorted_shards[i]['time'] - sorted_shards[i-1]['time'])
            
            if time_diff < time_threshold:
                current_cluster.append(sorted_shards[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_shards[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters

    def _compute_fracture_energy_enhanced(self, original_obj: Any, fragments: List[Any], pre_velocity: np.ndarray, fragment_velocities: List[np.ndarray], fracture_collisions: List[CollisionData], fracture_forces: List[ForceDataSequence], fracture_begin: float, fracture_end: float) -> float:
        """
        Compute fracture energy using collision data from ForceDataSequence.
        
        Analyzes CollisionData within the impulse range to compute energy.
        """
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
                                vertices = t.get_vertices(fracture_begin)
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
        
        # 2. Collision energy from ForceDataSequence
        collision_energy = self._compute_collision_energy(fracture_collisions=fracture_collisions, fracture_forces=fracture_forces, fracture_begin=fracture_begin, fracture_end=fracture_end)
        total_energy += collision_energy
        
        # 3. Stored elastic energy
        if hasattr(original_obj.acoustic_shader, 'failure_stress'):
            failure_stress = original_obj.acoustic_shader.failure_stress
        else:
            mat = original_obj.acoustic_shader
            failure_stress = self._compute_failure_stress(mat.young_modulus, mat.poisson_ratio, mat.roughness, mat.roughness, mat.friction, mat.density, mat.damping, mat.sound_speed)

        young_modulus = original_obj.acoustic_shader.young_modulus if original_obj.acoustic_shader else 1e9
        stressed_volume = 0.0001  # To be refined
        elastic_energy = (failure_stress**2 * stressed_volume) / (2 * young_modulus)
        total_energy += elastic_energy

        return max(total_energy, 0.01)
    
    def _compute_collision_energy(self, fracture_collisions: List[CollisionData], fracture_forces: List[ForceDataSequence], fracture_begin: float, fracture_end: float) -> float:
        """
        Compute collision energy from ForceDataSequence data.
        
        Uses CollisionData.frame, impulse_range, and frame_range to
        extract force data within the collision window.
        """
        total_energy = 0.0
        
        for coll in fracture_collisions:
            # Compute the time window for this collision
            start_time = coll.frame - coll.impulse_range / 2
            end_time = coll.frame + coll.impulse_range
            
            # Clamp to fracture window
            start_time = max(start_time, fracture_begin)
            end_time = min(end_time, fracture_end)
            
            if start_time >= end_time:
                continue
            
            # Find corresponding force data
            for force_seq in fracture_forces:
                if (force_seq.obj_idx == coll.obj1_idx and force_seq.other_obj_idx == coll.obj2_idx) or \
                   (force_seq.obj_idx == coll.obj2_idx and force_seq.other_obj_idx == coll.obj1_idx):
                    
                    # Sample force data within the window
                    sample_times = np.linspace(start_time, end_time, min(100, int(end_time - start_time) + 1))
                    
                    for t in sample_times:
                        try:
                            # Get force magnitude
                            normal_force = force_seq.get_normal_force_magnitude(t)
                            tangential_force = force_seq.get_tangential_force_magnitude(t)
                            
                            # Get velocity for power calculation
                            rel_vel = force_seq.get_relative_velocity(t)
                            vel_mag = np.linalg.norm(rel_vel)
                            
                            # Power = F * v
                            power = (normal_force + tangential_force) * vel_mag
                            
                            # Energy = power * dt
                            dt = sample_times[1] - sample_times[0] if len(sample_times) > 1 else 1.0 / self.sample_rate
                            total_energy += power * dt
                        except:
                            continue
        
        return total_energy
    
    def _get_force_at_time(self, force_sequences: List[ForceDataSequence], obj_idx: int, time: float) -> np.ndarray:
        """Get force vector at a specific time from force data sequences."""
        for force_seq in force_sequences:
            if force_seq.obj_idx == obj_idx or force_seq.other_obj_idx == obj_idx:
                try:
                    return force_seq.get_normal_force(time)
                except:
                    pass
        return np.zeros(3)
    
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
    
    def _estimate_crack_velocity(self, fracture_begin: float, fracture_end: float, original_obj: Any) -> float:
        """
        Estimate crack velocity from fracture duration and object size.
        """
        if fracture_end is None or fracture_end <= fracture_begin:
            return 500.0  # Default crack velocity
        
        # Get object size
        try:
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
    
    def _compute_failure_stress(self, young_modulus: float, poisson_ratio: float, roughness: float = None, friction: float = None, density: float = None, damping: float = None, sound_speed: float = None) -> float:
        """
        Approximate the failure stress (Pa) from material physical parameters.
        Uses a combined von Mises with friction criterion for shear failure adjustment.

        The failure stress is the maximum stress a material can withstand before fracture.

        Parameters:
        -----------
        young_modulus : float
            Young's modulus (Pa)
        poisson_ratio : float
            Poisson's ratio (dimensionless)
        roughness : float, optional
            Surface roughness (Ra in meters)
        friction : float, optional
            Coefficient of friction (dimensionless)
        density : float, optional
            Material density (kg/m³)
        damping : float, optional
            Damping ratio (dimensionless)
        sound_speed : float, optional
            Speed of sound in material (m/s)
        """
        # Poisson's ratio indicates material ductility/brittleness
        if poisson_ratio < 0.2:
            # Very brittle materials (e.g., ceramics, glass)
            failure_strain = 0.001
            ductility_factor = 0.3
        elif poisson_ratio < 0.25:
            # Brittle materials (e.g., cast iron)
            failure_strain = 0.002
            ductility_factor = 0.4
        elif poisson_ratio < 0.3:
            # Semi-brittle (e.g., some metals, concrete)
            failure_strain = 0.005
            ductility_factor = 0.5
        elif poisson_ratio < 0.35:
            # Moderate ductility (e.g., steel, aluminum)
            failure_strain = 0.01
            ductility_factor = 0.6
        elif poisson_ratio < 0.4:
            # Ductile materials (e.g., copper, gold)
            failure_strain = 0.03
            ductility_factor = 0.7
        elif poisson_ratio < 0.45:
            # Highly ductile (e.g., lead, rubber-like)
            failure_strain = 0.05
            ductility_factor = 0.8
        else:
            # Very ductile (e.g., polymers, elastomers)
            failure_strain = 0.08
            ductility_factor = 0.9

        # Roughness creates stress concentrations that reduce failure stress
        roughness_factor = 1.0
        if roughness is not None and roughness > 0:
            # Typical roughness ranges:
            #   - 1e-6 to 1e-5: polished surfaces
            #   - 1e-5 to 1e-4: machined surfaces
            #   - 1e-4 to 1e-3: rough surfaces
            #   - > 1e-3: very rough surfaces

            # Clamp roughness to reasonable range
            roughness_clamped = max(1e-6, min(1e-3, roughness))

            # Stress concentration factor due to roughness
            # Higher roughness = lower failure stress
            roughness_factor = 1.0 / (1.0 + 50.0 * roughness_clamped)

            # Apply additional reduction for very rough surfaces
            if roughness_clamped > 1e-4:
                roughness_factor *= 0.85

        friction_factor = 1.0
        if friction is not None and friction > 0:
            # Friction modifies the effective shear strength
            # Using a Coulomb-Mohr type criterion approximation:
            #   τ_failure = c + μ * σ_n
            # where c = cohesion, μ = friction coefficient, σ_n = normal stress

            # Estimate cohesion from Young's modulus
            # Cohesion is typically 5-15% of Young's modulus for many materials
            cohesion_ratio = 0.08 * (1.0 + 0.5 * (1.0 - ductility_factor))
            cohesion = young_modulus * cohesion_ratio

            # Estimate normal stress from Young's modulus and Poisson's ratio
            # Using Hooke's law approximation: σ_n ≈ E * ε_failure / (1 - ν²)
            normal_stress = young_modulus * failure_strain / (1 - poisson_ratio**2)

            # Shear strength from Coulomb-Mohr criterion
            shear_strength = cohesion + friction * normal_stress

            # Friction factor reduces the effective tensile failure stress
            # Higher friction = more shear failure, lower tensile failure
            if shear_strength > 0:
                # The von Mises equivalent stress with friction adjustment
                # For tensile failure with shear influence:
                #   σ_vonMises = sqrt(σ_tensile² + 3τ²)
                #   σ_effective = σ_tensile / sqrt(1 + 3*(τ/σ_tensile)²)
                tau_ratio = shear_strength / (young_modulus * failure_strain + 1e-10)
                friction_factor = 1.0 / np.sqrt(1.0 + 3.0 * tau_ratio**2)
            else:
                friction_factor = 0.95

        damping_factor = 1.0
        if damping is not None and damping > 0:
            # Higher damping = more energy dissipation = higher effective failure stress
            # But also indicates more viscoelastic behavior = lower brittle strength
            damping_factor = 1.0 + 2.0 * damping

            # Cap damping factor to avoid unrealistic values
            damping_factor = min(damping_factor, 2.5)

            # Reduce factor for very high damping (viscoelastic materials)
            if damping > 0.5:
                damping_factor *= 0.8

        density_factor = 1.0
        if density is not None and density > 0:
            # Higher density materials tend to have higher failure stress
            # Reference density: 2000 kg/m³ (typical for rocks)
            density_ratio = density / 2000.0
            density_factor = np.power(density_ratio, 0.2)  # Weak dependence

            # Limit density factor range
            density_factor = max(0.8, min(1.5, density_factor))

        sound_speed_factor = 1.0
        if sound_speed is not None and sound_speed > 0 and density is not None and density > 0:
            # Bulk modulus from sound speed: K = ρ * c²
            bulk_modulus = density * sound_speed**2

            # Estimate failure stress from bulk modulus
            # Typically 5-15% of bulk modulus
            bulk_modulus_factor = 0.08 * (1.0 + ductility_factor)

            # Sound speed based failure stress
            sound_speed_stress = bulk_modulus * bulk_modulus_factor

            # Normalize relative to Young's modulus based failure stress
            e_based_stress = young_modulus * failure_strain
            if e_based_stress > 0:
                sound_speed_factor = sound_speed_stress / e_based_stress
                sound_speed_factor = max(0.6, min(1.8, sound_speed_factor))

        # Base failure stress from Young's modulus and failure strain
        base_failure_stress = young_modulus * failure_strain

        # Apply all factors
        failure_stress = (base_failure_stress * roughness_factor * friction_factor * damping_factor * density_factor * sound_speed_factor)

        # Clamp to reasonable ranges for common materials
        # Theoretical limits: 1 MPa to 10 GPa
        failure_stress = max(1e6, min(10e9, failure_stress))

        # Additional material-specific adjustments based on Poisson's ratio
        if poisson_ratio > 0.45:
            # Elastomers have lower failure stress
            failure_stress = min(failure_stress, 50e6)  # 50 MPa max
        elif poisson_ratio < 0.15:
            # Very brittle materials like ceramics
            failure_stress = min(failure_stress, 500e6)  # 500 MPa max

        return failure_stress
    
    def _estimate_crack_length(self, original_obj: Any, fragments: List[Any], fracture_moment: float) -> float:
        """Estimate the crack length from fragment geometry."""
        try:
            vertices, _, _ = _load_mesh(original_obj, int(fracture_moment * self.sfps / self.sample_rate))

            fragment_vertices = []
            trajectories = self.entity_manager.get('trajectories')
            for frag in fragments:
                for t in trajectories.values():
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
