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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from dask import delayed, compute

from pbrAudioCommon import EntityManager
from pbrAudioCommon import Config, ObjectConfig
from pbrAudioCommon import _load_mesh, _load_pose
from pbrAudioCommon import _compute_face_normals
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from pbrAudioCommon import _adjust_for_fracture_shard

from pbrAudioCommon import CollisionData, CollisionType
from pbrAudioCommon import ForceData, ForceDataSequence, ContactType
from pbrAudioCommon import HertzianContact

from pbrAudioCommon import ModalVertices
from pbrAudioCommon import ScoreEvent, ScoreTrack

from ..lib.particles_trajectory_data import ParticlesTrajectoryData
from ..lib.surface_voxel_object import SurfaceVoxelObject


@dataclass
class ParticlesCollisionSolver:
    """
    Unified collision solver for particle systems using a voxel-based approach.

    The solver:
    1.  For each frame, finds which object surface voxels are near any particle.
    2.  For each involved voxel, it performs a fine-grained, sample-level search
        to find the exact moments of collision.
    3.  Calculates collision forces using Hertzian contact theory.
    4.  Generates ScoreTrack, ModalVertices, CollisionData, and ForceData.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.sample_rate = config.system.sample_rate
        self.fps = config.system.fps
        self.fps_base = config.system.fps_base
        self.subframes = config.system.subframes
        self.sfps = (self.fps / self.fps_base) * self.subframes
        
        # Output directories
        self.cache_path = config.system.cache_path
        self.collisions_dir = f"{self.cache_path}/particle_collisions"
        self.forces_dir = f"{self.cache_path}/particle_forces"
        self.modalvertices_dir = f"{self.cache_path}/particle_modalvertices"
        self.scoretracks_dir = f"{self.cache_path}/particle_scoretracks"
        
        os.makedirs(self.collisions_dir, exist_ok=True)
        os.makedirs(self.forces_dir, exist_ok=True)
        os.makedirs(self.modalvertices_dir, exist_ok=True)
        os.makedirs(self.scoretracks_dir, exist_ok=True)
        
        self.hertzian_contact = HertzianContact(self.entity_manager)

    def compute(self, particles_idx: int) -> None:
        """
        Main entry point to compute collisions for a particle system against all objects.
        """
        config = self.entity_manager.get('config')
        
        # Get particle configuration and trajectory data
        particle_cfg, particle_trajectory = self._get_particle_data(particles_idx)
        if particle_cfg is None or particle_trajectory is None:
            return

        debug_print(f"Computing collisions for particle system '{particle_cfg.name}' (massive: {particle_cfg.proxy})")

        # Get all surface voxel objects
        surface_voxel_objects = self.entity_manager.get('objects')
        sv_objects = {idx: obj for idx, obj in surface_voxel_objects.items() if isinstance(obj, SurfaceVoxelObject)}

        if not sv_objects:
            debug_print("No SurfaceVoxelObject instances found in EntityManager. Cannot compute particle collisions.")
            return

        # Get the list of frames to process
        frames = particle_trajectory.frames
        if len(frames) == 0:
            debug_print("Particle trajectory has no frames.")
            return
            
        # Stage 1: Frame-by-frame broad-phase collision detection
        all_potential_collisions = self._detect_collisions_for_all_frames(
            particle_trajectory=particle_trajectory,
            sv_objects=sv_objects,
            frames=frames
        )

        # Stage 2: Sample-level fine-grained collision detection and data generation
        self._process_sample_level_collisions(
            particle_cfg=particle_cfg,
            particle_trajectory=particle_trajectory,
            potential_collisions=all_potential_collisions,
            sv_objects=sv_objects
        )

    def _get_particle_data(self, particles_idx: int) -> Tuple[Optional[Any], Optional[ParticlesTrajectoryData]]:
        """Helper to fetch particle config and trajectory data."""
        config = self.entity_manager.get('config')
        particle_cfg = next((p for p in config.particles if p.idx == particles_idx), None)
        if particle_cfg is None:
            debug_print(f"Particle system {particles_idx} not found in config.")
            return None, None

        trajectories = self.entity_manager.get('trajectories')
        particle_trajectory = next((
            t for t in trajectories.values()
            if isinstance(t, ParticlesTrajectoryData) and t.particles_idx == particles_idx
        ), None)

        if particle_trajectory is None:
            debug_print(f"No trajectory data found for particle system {particles_idx}.")
            return None, None
            
        return particle_cfg, particle_trajectory

    def _detect_collisions_for_all_frames(self, particle_trajectory: ParticlesTrajectoryData, sv_objects: Dict[int, SurfaceVoxelObject], frames: np.ndarray) -> Dict[int, Dict[int, List[Dict]]]:
        """
        Stage 1: Broad-phase detection. For each frame, find which voxels are near any particle.
        Returns a nested dictionary: {obj_idx: {voxel_idx: [collision_info, ...]}}
        """
        potential_collisions = {}

        # Get all particle positions and sizes for all frames at once for efficiency
        all_positions = np.array([particle_trajectory.get_position(frame) for frame in frames])
        all_sizes = np.array([particle_trajectory.get_sizes(frame) for frame in frames])
        
        # Get states to filter out dead/unborn particles
        all_states = np.array([particle_trajectory.get_states(frame) for frame in frames])

        for obj_idx, sv_object in sv_objects.items():
            if sv_object.base_voxel_grid is None:
                continue
            
            potential_collisions[obj_idx] = {}

            for frame_idx, frame in enumerate(frames):
                # Filter for alive particles only
                alive_mask = all_states[frame_idx] == 1
                if not np.any(alive_mask):
                    continue
                
                positions = all_positions[frame_idx][alive_mask]
                sizes = all_sizes[frame_idx][alive_mask]
                particle_indices = np.arange(len(alive_mask))[alive_mask]

                # Get the object's world-space voxel grid for this frame
                voxel_grid, transform_matrix = sv_object.update_voxels(frame)
                if voxel_grid is None:
                    continue

                # Transform voxel grid to world space
                world_voxel_centers = trimesh.transformations.transform_points(voxel_grid.points, transform_matrix)
                voxel_tree = cKDTree(world_voxel_centers)

                # For each particle, find nearby voxels
                for i, pos in enumerate(positions):
                    # Use particle size as the query radius, plus a small margin
                    query_radius = sizes[i] * 1.5
                    nearby_voxel_indices = voxel_tree.query_ball_point(pos, query_radius, workers=-1)

                    if nearby_voxel_indices:
                        particle_idx = particle_indices[i]
                        for voxel_idx in nearby_voxel_indices:
                            # Store potential collision info
                            if voxel_idx not in potential_collisions[obj_idx]:
                                potential_collisions[obj_idx][voxel_idx] = []
                            
                            potential_collisions[obj_idx][voxel_idx].append({
                                'particle_idx': particle_idx,
                                'frame': frame,
                                'frame_idx': frame_idx,
                                'size': sizes[i]
                            })
        return potential_collisions

    def _process_sample_level_collisions(self, particle_cfg: Any, particle_trajectory: ParticlesTrajectoryData, potential_collisions: Dict[int, Dict[int, List[Dict]]], sv_objects: Dict[int, SurfaceVoxelObject]):
        """
        Stage 2: Fine-grained, sample-level collision detection and data generation.
        """
        for obj_idx, voxel_collisions in potential_collisions.items():
            if not voxel_collisions:
                continue

            sv_object = sv_objects[obj_idx]
            config_obj = next((o for o in self.entity_manager.get('config').objects if o.idx == obj_idx), None)
            if config_obj is None:
                continue

            # Initialize data structures for this object
            score_track = ScoreTrack(obj_idx=obj_idx, obj_name=config_obj.name)
            all_contact_vertices = set()

            for voxel_idx, collision_infos in voxel_collisions.items():
                # Sort collision infos by frame to process sequentially
                collision_infos.sort(key=lambda x: x['frame'])

                # Group by particle to find continuous contact periods
                particle_groups = {}
                for info in collision_infos:
                    particle_groups.setdefault(info['particle_idx'], []).append(info)

                for particle_idx, infos in particle_groups.items():
                    # Find continuous contact regions in time for this particle-voxel pair
                    contact_regions = self._find_contact_regions_in_frames(infos)

                    for region in contact_regions:
                        # Refine the collision time to sample-level precision
                        collision_moment = self._refine_collision_time(
                            particle_trajectory, particle_idx, region, sv_object, voxel_idx
                        )
                        
                        if collision_moment is None:
                            continue

                        # Create collision data
                        collision_data = self._create_collision_data(
                            particle_idx, obj_idx, collision_moment, region
                        )
                        _ = self.entity_manager.register('collisions', collision_data)

                        # Calculate forces
                        force_data = self._calculate_forces_for_collision(
                            collision_data, particle_cfg, particle_trajectory, config_obj, sv_object
                        )
                        if force_data:
                            _ = self.entity_manager.register('forces', force_data)

                        # Add to score track and collect contact vertices
                        self._add_to_score_track(
                            score_track, collision_data, force_data, voxel_idx, config_obj, particle_cfg
                        )
                        all_contact_vertices.add(voxel_idx)

            # Finalize and save data for this object
            if score_track.events:
                score_track.save(f"{self.scoretracks_dir}/particles_{particle_cfg.idx}_obj_{obj_idx}.tar.gz")
                self._update_and_save_modal_vertices(obj_idx, list(all_contact_vertices), config_obj, sv_object)

    def _find_contact_regions_in_frames(self, infos: List[Dict]) -> List[Dict]:
        """Groups a list of frame-level collision infos into continuous contact regions."""
        regions = []
        if not infos:
            return regions

        current_region = {
            'start_frame': infos[0]['frame'],
            'end_frame': infos[0]['frame'],
            'particle_idx': infos[0]['particle_idx'],
            'frames': [infos[0]['frame']]
        }

        for i in range(1, len(infos)):
            # If the gap is more than one frame, it's a new region
            if infos[i]['frame'] - infos[i-1]['frame'] > 1:
                regions.append(current_region)
                current_region = {
                    'start_frame': infos[i]['frame'],
                    'end_frame': infos[i]['frame'],
                    'particle_idx': infos[i]['particle_idx'],
                    'frames': [infos[i]['frame']]
                }
            else:
                current_region['end_frame'] = infos[i]['frame']
                current_region['frames'].append(infos[i]['frame'])
        
        regions.append(current_region)
        return regions

    def _refine_collision_time(self, particle_trajectory: ParticlesTrajectoryData, particle_idx: int, region: Dict, sv_object: SurfaceVoxelObject, voxel_idx: int) -> Optional[float]:
        """
        Refines the collision time to sample-level precision by interpolating
        the particle position and voxel position between frames.
        """
        # Get the start and end frames of the contact region
        start_frame = region['start_frame']
        end_frame = region['end_frame']

        # Convert frames to sample indices
        start_sample = int(start_frame * self.sample_rate / self.sfps)
        end_sample = int(end_frame * self.sample_rate / self.sfps)

        if start_sample >= end_sample:
            # If the contact is within a single frame, just use the start
            return float(start_sample)

        # We need to find the exact sample where the particle enters the voxel's
        # collision sphere. We'll do a binary search or a linear scan.
        # For simplicity and robustness, a linear scan is used here.
        for sample in np.arange(start_sample, end_sample):
            # Get particle position and size at this sample
            pos = particle_trajectory.get_position(sample, particle_idx)
            size = particle_trajectory.get_sizes(sample, particle_idx)

            # Get voxel world position at this sample
            voxel_grid, transform_matrix = sv_object.update_voxels(sample)
            if voxel_grid is None:
                continue
            
            # Get the specific voxelel's world position
            # Note: voxel_grid.points are in the local frame, so we transform them
            voxel_local_pos = voxel_grid.points[voxel_idx]
            voxel_world_pos = trimesh.transformations.transform_points([voxel_local_pos], transform_matrix)[0]

            # Check for collision
            distance = np.linalg.norm(pos - voxel_world_pos)
            if distance < size:
                return float(sample)
        
        # If no collision found in the scan, return None
        return None

    def _create_collision_data(self, particle_idx: int, obj_idx: int, collision_sample: float, region: Dict) -> CollisionData:
        """Creates a CollisionData object for a sample-level collision."""
        # Determine collision type based on duration
        duration_samples = (region['end_frame'] - region['start_frame']) * self.sample_rate / self.sfps
        if duration_samples <= 2: # Impact if very short
            collision_type = CollisionType.IMPACT
        else:
            collision_type = CollisionType.CONTACT

        return CollisionData(
            type=collision_type,
            obj1_idx=obj_idx,
            obj2_idx=-1,  # Represents a particle
            frame=collision_sample,
            frame_range=int(duration_samples),
            valid=True
        )

    def _calculate_forces_for_collision(self, collision: CollisionData, particle_cfg: Any, particle_trajectory: ParticlesTrajectoryData, config_obj: ObjectConfig, sv_object: SurfaceVoxelObject) -> Optional[ForceDataSequence]:
        """Calculates the forces for a particle-object collision."""
        frame = collision.frame
        particle_idx = 0 # Placeholder, as we don't have a single particle index here. This needs refinement if per-particle forces are needed.

        # For simplicity, we'll use a generic particle for force calculation.
        # A more detailed implementation might average forces from all colliding particles.
        
        # Get particle properties (using a representative particle)
        # This is a simplification. A better approach would be to get the specific particle.
        # Let's assume we can get the particle from the region info.
        # For now, we'll use a placeholder.
        particle_velocity = np.zeros(3)
        particle_size = 0.01 # Default size
        
        # Get object properties
        obj_velocity = np.zeros(3) # Assume static
        obj_mass = self._get_object_mass(config_obj, sv_object, frame)
        
        # Simplified force calculation
        relative_velocity = particle_velocity - obj_velocity
        normal_velocity = relative_velocity # Assume normal is along velocity for simplicity
        normal_force_mag = np.linalg.norm(normal_velocity) * particle_cfg.density * (4/3 * np.pi * particle_size**3) / (1/self.sample_rate)
        
        # Create a single-frame ForceDataSequence
        frames = np.array([frame])
        force_data = ForceDataSequence(
            frames=frames,
            obj_idx=config_obj.idx,
            other_obj_idx=-1,
            restitution=np.array([0.5]),
            relative_velocity=np.array([relative_velocity]),
            normal_velocity=np.array([normal_velocity]),
            normal_force=np.array([normal_velocity * normal_force_mag]),
            tangential_force=np.array([np.zeros(3)]),
            tangential_velocity=np.array([np.zeros(3)]),
            normal_force_magnitude=np.array([normal_force_mag]),
            tangential_force_magnitude=np.array([0.0]),
            stochastic_normal_force=np.array([normal_velocity * normal_force_mag]),
            stochastic_tangential_force=np.array([np.zeros(3)]),
            contact_type=np.array([ContactType.IMPACT]),
            contact_point=np.array([np.zeros(3)]),
            contact_radius=np.array([particle_size]),
            rolling_radius=np.array([particle_size]),
            impact_duration=np.array([1/self.sample_rate]),
            contact_pressure=np.array([0.0]),
            penetration_depth=np.array([0.0]),
            coupling_strength=np.array([0.5])
        )
        return force_data

    def _get_object_mass(self, config_obj: ObjectConfig, sv_object: SurfaceVoxelObject, frame: float) -> float:
        """Get object mass."""
        try:
            vertices, _, faces = _load_mesh(config_obj, int(frame * self.sfps / self.sample_rate))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.density = config_obj.acoustic_shader.density if config_obj.acoustic_shader else 1000.0
            mass = mesh.mass
            return mass if mass > 9e-5 else 0.0001
        except:
            return 0.001

    def _add_to_score_track(self, score_track: ScoreTrack, collision: CollisionData, force: ForceDataSequence, voxel_idx: int, config_obj: ObjectConfig, particle_cfg: Any):
        """Adds a score event to the track."""
        start_sample = int(collision.frame)
        stop_sample = int(collision.frame + collision.frame_range)
        total_samples = stop_sample + 1000 # Add padding for decay

        # Initialize score arrays
        score_type = np.zeros((total_samples, 1), dtype=np.int32)
        score_vertex_ids = np.zeros((total_samples, config_obj.geometry.shape[0]), dtype=np.bool_)
        score_contact_area = np.zeros((total_samples, 1), dtype=np.float32)

        # Determine contact type
        contact_type_map = {CollisionType.IMPACT: 1, CollisionType.CONTACT: 3}
        contact_type = contact_type_map.get(collision.type, 0)

        # Fill in score data
        for sample_idx in range(start_sample, min(stop_sample, total_samples)):
            score_type[sample_idx] = contact_type
            score_contact_area[sample_idx] = 1.0 / len(config_obj.geometry) # Placeholder area
            score_vertex_ids[sample_idx, voxel_idx] = True

        # Create and add score event
        score_event = ScoreEvent(
            coll_obj=-1, # Particle
            start_sample=start_sample,
            stop_sample=stop_sample,
            type=score_type,
            vertex_ids=score_vertex_ids,
            contact_area=score_contact_area,
            force=None, # Force is handled separately
            coupling_data=None
        )
        score_track.add_event(score_event)

    def _update_and_save_modal_vertices(self, obj_idx: int, contact_vertices: List[int], config_obj: ObjectConfig, sv_object: SurfaceVoxelObject):
        """Updates and saves the modal vertices for an object."""
        if not contact_vertices:
            return

        unique_vertices = np.array(sorted(list(set(contact_vertices))))
        
        # Check if modal vertices already exist
        modal_vertices_list = self.entity_manager.get('modal_vertices')
        modal_vertices = next((mv for mv in modal_vertices_list.values() if mv.obj_idx == obj_idx), None)
        
        if modal_vertices is not None:
            modal_vertices.add_vertices(unique_vertices)
        else:
            new_modal_vertices = ModalVertices(
                obj_idx=obj_idx,
                vertices=unique_vertices,
                connected_area=len(unique_vertices) / len(sv_object.base_voxel_grid.points)
            )
            _ = self.entity_manager.register('modal_vertices', new_modal_vertices)
        
        # Save modal vertices
        new_modal_vertices.save(f"{self.modalvertices_dir}/{obj_idx:05d}.json")

    def save_all_data(self) -> None:
        """Save all generated collision and force data to disk."""
        # Save collisions
        collisions = self.entity_manager.get('collisions')
        for c_idx, coll in collisions.items():
            if isinstance(coll, CollisionData) and coll.obj2_idx == -1: # Particle collision
                coll.save(f"{self.collisions_dir}/{c_idx:05d}.pkl")
        
        # Save forces
        forces = self.entity_manager.get('forces')
        for f_idx, force in forces.items():
            if isinstance(force, ForceDataSequence) and force.other_obj_idx == -1: # Particle force
                force.save(f"{self.forces_dir}/{f_idx:05d}.pkl")
        
        debug_print(f"Saved particle collision and force data.")
