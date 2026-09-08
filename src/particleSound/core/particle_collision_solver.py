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
from scipy.interpolate import CubicSpline
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

from ..lib.particle_trajectory_data import ParticleTrajectoryData


@dataclass
class ParticleCollisionSolver:
    """
    Unified collision solver for particle systems.
    
    Combines the functionality of DistanceSolver, ForceSolver, and CollisionSolver
    for detecting and resolving collisions between particles and objects.
    
    The solver:
    1. Detects collisions between particles and objects using distance analysis
    2. Calculates collision forces using Hertzian contact theory
    3. Resolves collisions by finding contact vertices for modal synthesis
    """
    
    entity_manager: EntityManager
    
    # Detection parameters
    collision_margin: float = 0.05  # System collision margin
    samples_per_object: int = 1000  # Samples for distance calculation
    velocity_threshold: float = 0.01  # Minimum velocity for collision detection
    
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
        self.distances_dir = f"{self.cache_path}/particle_distances"
        self.collisions_dir = f"{self.cache_path}/particle_collisions"
        self.forces_dir = f"{self.cache_path}/particle_forces"
        self.modalvertices_dir = f"{self.cache_path}/particle_modalvertices"
        self.scoretracks_dir = f"{self.cache_path}/particle_scoretracks"
        
        os.makedirs(self.distances_dir, exist_ok=True)
        os.makedirs(self.collisions_dir, exist_ok=True)
        os.makedirs(self.forces_dir, exist_ok=True)
        os.makedirs(self.modalvertices_dir, exist_ok=True)
        os.makedirs(self.scoretracks_dir, exist_ok=True)
        
        # Initialize Hertzian contact calculator
        self.hertzian_contact = HertzianContact(self.entity_manager)
        
        # Cache for particle data
        self._particle_cache = {}
        
        # Cache for object data
        self._object_cache = {}
    
    def compute(self, particles_idx: int, obj_idx: int = None) -> None:
        """
        Compute collisions for a particle system.
        
        Parameters:
        -----------
        particles_idx : int
            Index of the particle system
        obj_idx : int, optional
            Index of specific object to check collisions with.
            If None, checks against all objects.
        """
        config = self.entity_manager.get('config')
        
        # Get particle configuration
        particle_cfg = None
        for p in config.particles:
            if p.idx == particles_idx:
                particle_cfg = p
                break
        
        if particle_cfg is None:
            raise ValueError(f"Particle system {particles_idx} not found")
        
        # Get particle trajectory data
        trajectories = self.entity_manager.get('trajectories')
        particle_trajectory = None
        for t_idx in trajectories.keys():
            if isinstance(trajectories[t_idx], ParticleTrajectoryData) and trajectories[t_idx].particles_idx == particles_idx:
                particle_trajectory = trajectories[t_idx]
                break
        
        if particle_trajectory is None:
            debug_print(f"No trajectory data found for particle system {particles_idx}")
            return
        
        # Determine which objects to check
        objects_to_check = []
        if obj_idx is not None:
            for obj in config.objects:
                if obj.idx == obj_idx:
                    objects_to_check.append(obj)
                    break
        else:
            objects_to_check = config.objects
        
        # For each object, compute particle-object collisions
        for config_obj in objects_to_check:
            if config_obj.static:
                # Static objects: particles collide with them
                self._compute_static_collisions(
                    particles_idx=particles_idx,
                    particle_cfg=particle_cfg,
                    particle_trajectory=particle_trajectory,
                    config_obj=config_obj
                )
            else:
                # Dynamic objects: check if particles can interact
                # (Particles typically only collide with static objects or other particles)
                debug_print(f"Warning: Particle collisions with dynamic objects not fully supported yet")
    
    def _compute_static_collisions(
        self,
        particles_idx: int,
        particle_cfg: Any,
        particle_trajectory: ParticleTrajectoryData,
        config_obj: ObjectConfig
    ) -> None:
        """
        Compute collisions between particles and a static object.
        """
        debug_print(f"Computing particle-object collisions: particles={particle_cfg.name}, object={config_obj.name}")
        
        # Get object trajectory (static)
        trajectories = self.entity_manager.get('trajectories')
        obj_trajectory = None
        for t_idx in trajectories.keys():
            if hasattr(trajectories[t_idx], 'obj_idx') and trajectories[t_idx].obj_idx == config_obj.idx:
                obj_trajectory = trajectories[t_idx]
                break
        
        if obj_trajectory is None:
            debug_print(f"Warning: No trajectory found for object {config_obj.name}")
            # Try to create static trajectory
            obj_trajectory = self._create_static_trajectory(config_obj)
            if obj_trajectory is None:
                return
        
        # Get object mesh data
        obj_vertices = obj_trajectory.get_vertices(0)
        obj_faces = obj_trajectory.get_faces()
        obj_normals = obj_trajectory.get_normals(0)
        
        # Build KD-tree for object vertices
        obj_tree = cKDTree(obj_vertices)
        
        # Get particle data
        n_particles = particle_trajectory.particles_count
        frames = self._get_particle_frames(particle_trajectory)
        
        # Initialize collision tracking
        collision_events = []
        force_sequences = []
        
        # Process each particle
        for particle_idx in range(n_particles):
            # Get particle positions over time
            positions = self._get_particle_positions(particle_trajectory, particle_idx, frames)
            
            # Get particle sizes
            sizes = self._get_particle_sizes(particle_trajectory, particle_idx, frames)
            
            # Detect collisions
            particle_collisions = self._detect_particle_collisions(
                particle_idx=particle_idx,
                positions=positions,
                sizes=sizes,
                frames=frames,
                obj_vertices=obj_vertices,
                obj_faces=obj_faces,
                obj_tree=obj_tree,
                config_obj=config_obj
            )
            
            collision_events.extend(particle_collisions)
        
        # Process collisions to generate forces and score data
        if collision_events:
            self._process_collision_events(
                particles_idx=particles_idx,
                particle_cfg=particle_cfg,
                particle_trajectory=particle_trajectory,
                config_obj=config_obj,
                obj_trajectory=obj_trajectory,
                collision_events=collision_events
            )
    
    def _create_static_trajectory(self, config_obj: ObjectConfig) -> Any:
        """
        Create a static trajectory for an object if none exists.
        """
        try:
            from physicsSolver import TrajectoryData
            from pbrAudioCommon import _load_mesh, _load_pose
            
            # Load mesh data
            vertices, normals, faces = _load_mesh(config_obj, 0)
            
            # Load pose data (static: single position/rotation)
            positions, rotations = _load_pose(config_obj)
            
            # Create static trajectory
            return TrajectoryData(
                obj_idx=config_obj.idx,
                static=True,
                sfps=self.sfps,
                sample_rate=self.sample_rate,
                positions=positions,
                rotations=rotations,
                vertices=vertices,
                normals=normals,
                faces=faces
            )
        except Exception as e:
            debug_print(f"Error creating static trajectory for {config_obj.name}: {e}")
            return None
    
    def _get_particle_frames(self, particle_trajectory: ParticleTrajectoryData) -> np.ndarray:
        """Get the frame times for a particle trajectory."""
        if particle_trajectory.is_static:
            return np.array([0])
        
        # Get from the first particle's position spline
        # The frames are stored in the spline's x values
        if particle_trajectory.positions is not None and particle_trajectory.positions.shape[0] > 0:
            if hasattr(particle_trajectory.positions[0, 0], 'x'):
                return particle_trajectory.positions[0, 0].x
        
        # Fallback: generate frames
        n_frames = 100  # Default
        return np.arange(n_frames) * self.sample_rate / self.sfps
    
    def _get_particle_positions(self, particle_trajectory: ParticleTrajectoryData, 
                                particle_idx: int, frames: np.ndarray) -> np.ndarray:
        """Get particle positions at all frames."""
        positions = np.zeros((len(frames), 3))
        
        for i, frame in enumerate(frames):
            pos = particle_trajectory.get_position(frame, particle_idx)
            positions[i] = pos
        
        return positions
    
    def _get_particle_sizes(self, particle_trajectory: ParticleTrajectoryData,
                           particle_idx: int, frames: np.ndarray) -> np.ndarray:
        """Get particle sizes at all frames."""
        sizes = np.zeros(len(frames))
        
        for i, frame in enumerate(frames):
            size = particle_trajectory.get_sizes(frame, particle_idx)
            if isinstance(size, np.ndarray):
                sizes[i] = size[0] if size.shape[0] > 0 else 0.01
            else:
                sizes[i] = size if size is not None else 0.01
        
        return sizes
    
    def _detect_particle_collisions(
        self,
        particle_idx: int,
        positions: np.ndarray,
        sizes: np.ndarray,
        frames: np.ndarray,
        obj_vertices: np.ndarray,
        obj_faces: np.ndarray,
        obj_tree: cKDTree,
        config_obj: ObjectConfig
    ) -> List[Dict[str, Any]]:
        """
        Detect collisions between a single particle and an object.
        
        Returns:
        --------
        List of collision event dictionaries
        """
        collisions = []
        
        # Object center for distance calculations
        obj_center = np.mean(obj_vertices, axis=0)
        
        # Object bounding sphere radius
        obj_radius = np.max(np.linalg.norm(obj_vertices - obj_center, axis=1))
        
        # Track contact state
        in_contact = False
        contact_start = 0
        contact_distances = []
        contact_positions = []
        
        for i, frame in enumerate(frames):
            # Get particle position and size
            pos = positions[i]
            size = sizes[i]
            
            # Calculate distance from particle to object
            # Use KD-tree for nearest vertex distance
            dist_to_vertex, nearest_idx = obj_tree.query(pos)
            
            # Also check distance to object center for bounding sphere test
            dist_to_center = np.linalg.norm(pos - obj_center)
            
            # Effective collision distance includes particle radius
            collision_distance = dist_to_vertex - size
            
            # Check if particle is near object
            if collision_distance < self.collision_margin:
                # Particle is in contact or near contact
                if not in_contact:
                    # Start of contact
                    in_contact = True
                    contact_start = i
                    contact_distances = []
                    contact_positions = []
                
                contact_distances.append(collision_distance)
                contact_positions.append(pos)
            else:
                # Particle is not in contact
                if in_contact:
                    # End of contact region
                    in_contact = False
                    
                    # Determine if this was an impact or continuous contact
                    contact_duration = i - contact_start
                    
                    if contact_duration <= 2:
                        # Impact event (1-2 frames)
                        collision = self._create_impact_collision(
                            particle_idx=particle_idx,
                            frame=frames[i],
                            position=pos,
                            distance=collision_distance,
                            obj_idx=config_obj.idx,
                            obj_vertices=obj_vertices,
                            obj_faces=obj_faces
                        )
                    else:
                        # Continuous contact
                        collision = self._create_contact_collision(
                            particle_idx=particle_idx,
                            start_frame=frames[contact_start],
                            end_frame=frames[i-1],
                            start_position=contact_positions[0] if contact_positions else pos,
                            distances=contact_distances,
                            obj_idx=config_obj.idx,
                            obj_vertices=obj_vertices,
                            obj_faces=obj_faces
                        )
                    
                    if collision is not None:
                        collisions.append(collision)
        
        # Handle contact at end of sequence
        if in_contact:
            contact_duration = len(frames) - contact_start
            if contact_duration <= 2:
                collision = self._create_impact_collision(
                    particle_idx=particle_idx,
                    frame=frames[-1],
                    position=positions[-1],
                    distance=contact_distances[-1] if contact_distances else 0,
                    obj_idx=config_obj.idx,
                    obj_vertices=obj_vertices,
                    obj_faces=obj_faces
                )
            else:
                collision = self._create_contact_collision(
                    particle_idx=particle_idx,
                    start_frame=frames[contact_start],
                    end_frame=frames[-1],
                    start_position=contact_positions[0] if contact_positions else positions[-1],
                    distances=contact_distances,
                    obj_idx=config_obj.idx,
                    obj_vertices=obj_vertices,
                    obj_faces=obj_faces
                )
            
            if collision is not None:
                collisions.append(collision)
        
        return collisions
    
    def _create_impact_collision(
        self,
        particle_idx: int,
        frame: float,
        position: np.ndarray,
        distance: float,
        obj_idx: int,
        obj_vertices: np.ndarray,
        obj_faces: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Create an impact collision event.
        """
        # Find closest point on object
        tree = cKDTree(obj_vertices)
        dist, nearest_idx = tree.query(position)
        
        # Get contact point on object
        contact_point = obj_vertices[nearest_idx]
        
        # Get contact normal (approximate using vertex normal)
        # For simplicity, use direction from object center to contact point
        obj_center = np.mean(obj_vertices, axis=0)
        normal = position - contact_point
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0, 1, 0])
        
        return {
            'type': CollisionType.IMPACT,
            'particle_idx': particle_idx,
            'frame': frame,
            'position': position,
            'contact_point': contact_point,
            'contact_normal': normal,
            'distance': distance,
            'obj_idx': obj_idx,
            'duration': 1
        }
    
    def _create_contact_collision(
        self,
        particle_idx: int,
        start_frame: float,
        end_frame: float,
        start_position: np.ndarray,
        distances: List[float],
        obj_idx: int,
        obj_vertices: np.ndarray,
        obj_faces: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Create a continuous contact collision event.
        """
        # Find closest point on object from start position
        tree = cKDTree(obj_vertices)
        dist, nearest_idx = tree.query(start_position)
        
        contact_point = obj_vertices[nearest_idx]
        
        # Calculate duration in samples
        duration = int((end_frame - start_frame) * self.sfps / self.sample_rate)
        duration = max(duration, 1)
        
        return {
            'type': CollisionType.CONTACT,
            'particle_idx': particle_idx,
            'frame': start_frame,
            'end_frame': end_frame,
            'position': start_position,
            'contact_point': contact_point,
            'distance': np.mean(distances) if distances else 0,
            'obj_idx': obj_idx,
            'duration': duration,
            'frame_range': duration
        }
    
    def _process_collision_events(
        self,
        particles_idx: int,
        particle_cfg: Any,
        particle_trajectory: ParticleTrajectoryData,
        config_obj: ObjectConfig,
        obj_trajectory: Any,
        collision_events: List[Dict[str, Any]]
    ) -> None:
        """
        Process collision events to generate forces and score data.
        """
        debug_print(f"Processing {len(collision_events)} collision events for particle system {particles_idx_idx}")
        
        # Initialize score tracks
        score_track = ScoreTrack(
            obj_idx=config_obj.idx,
            obj_name=config_obj.name
        )
        _ = self.entity_manager.register('score_tracks', score_track)
        
        # Get object mesh data for modal vertices
        obj_vertices = obj_trajectory.get_vertices(0)
        obj_faces = obj_trajectory.get_faces()
        
        # Initialize modal vertices tracking
        all_contact_vertices = []
        
        # Process each collision event
        for event in collision_events:
            # Calculate forces for this collision
            force_data = self._calculate_collision_forces(
                event=event,
                particle_cfg=particle_cfg,
                particle_trajectory=particle_trajectory,
                config_obj=config_obj,
                obj_trajectory=obj_trajectory
            )
            
            if force_data is not None:
                # Register force data
                _ = self.entity_manager.register('forces', force_data)
                
                # Find contact vertices for modal synthesis
                contact_vertices = self._find_contact_vertices(
                    event=event,
                    obj_vertices=obj_vertices,
                    obj_faces=obj_faces,
                    contact_point=event['contact_point'],
                    collision_margin=self.collision_margin
                )
                
                if len len(contact_vertices) > 0:
                    all_contact_vertices.extend(contact_vertices)
                    
                    # Create score event
                    self._create_score_event(
                        score_track=score_track,
                        event=event,
                        contact_vertices=contact_vertices,
                        force_data=force_data,
                        obj_vertices=obj_vertices,
                        obj_faces=obj_faces
                    )
        
        # Update modal vertices
        if all_contact_vertices:
            unique_vertices = np.unique(np.array(all_contact_vertices))
            
            # Check if modal vertices already exist for this object
            modal_vertices_list = self.entity_manager.get('modal_vertices')
            modal_vertices = None
            for mv_idx in modal_vertices_list.keys():
                if modal_vertices_list[mv_idx].obj_idx == config_obj.idx:
                    modal_vertices = modal_vertices_list[mv_idx]
                    break
            
            if modal_vertices is not None:
                modal_vertices.add_vertices(unique_vertices)
            else:
                new_modal_vertices = ModalVertices(
                    obj_idx=config_obj.idx,
                    vertices=unique_vertices,
                    connected_area=len(unique_vertices) / len(obj_faces)
                )
                _ = self.entity_manager.register('modal_vertices', new_modal_vertices)
            
            # Save modal vertices
            new_modal_vertices.save(f"{self.modalvertices_dir}/{config_obj.idx:05d}.json")
        
        # Save score track
        score_track.save(f"{self.scoretracks_dir}/{config_obj.idx:05d}.tar.gz")
    
    def _calculate_collision_forces(
        self,
        event: Dict[str, Any],
        particle_cfg: Any,
        particle_trajectory: ParticleTrajectoryData,
        config_obj: ObjectConfig,
        obj_trajectory: Any
    ) -> Optional[ForceDataSequence]:
        """
        Calculate collision forces for a particle-object collision.
        """
        frame = event['frame']
        particle_idx = event['particle_idx']
        
        # Get particle velocity at collision
        particle_velocity = particle_trajectory.get_velocity(frame, particle_idx)
        
        # Get particle mass (approximate from size and density)
        particle_size = particle_trajectory.get_sizes(frame, particle_idx)
        if isinstance(particle_size, np.ndarray):
            particle_size = particle_size[0] if particle_size.shape[0] > 0 else 0.01
        
        # Approximate particle as sphere with given radius
        particle_volume = (4/3) * np.pi * particle_size**3
        particle_mass = particle_cfg.density * particle_volume if hasattr(particle_cfg, 'density') else 0.001
        
        # Get object properties
        obj_velocity = np.zeros(3)  # Static object
        obj_mass = self._get_object_mass(config_obj, obj_trajectory, frame)
        
        # Calculate relative velocity
        relative_velocity = particle_velocity - obj_velocity
        relative_speed = np.linalg.norm(relative_velocity)
        
        # Get contact normal
        contact_normal = event.get('contact_normal', np.array([0, 1, 0]))
        
        # Decompose velocity
        normal_velocity = np.dot(relative_velocity, contact_normal) * contact_normal
        tangential_velocity = relative_velocity - normal_velocity
        
        # Get material properties
        young_modulus = config_obj.acoustic_shader.young_modulus if config_obj.acoustic_shader else 1e9
        poisson_ratio = config_obj.acoustic_shader.poisson_ratio if config_obj.acoustic_shader else 0.3
        density = config_obj.acoustic_shader.density if config_obj.acoustic_shader else 1000.0
        damping = config_obj.acoustic_shader.damping if config_obj.acoustic_shader else 0.02
        
        # Calculate collision force (simplified Hertzian)
        # For particle-object collision, use effective radius
        R_particle = particle_size
        R_object = self._get_object_effective_radius(config_obj, obj_trajectory, frame)
        
        if R_particle > 0 and R_object > 0:
            R_eff = (R_particle * R_object) / (R_particle + R_object)
        else:
            R_eff = R_particle if R_particle > 0 else 0.01
        
        # Effective modulus
        E_star = young_modulus / (2 * (1 - poisson_ratio**2))
        
        # Calculate normal force using impulse-momentum
        restitution = 0.5  # Default coefficient of restitution
        if hasattr(config_obj.acoustic_shader, 'restitution'):
            restitution = config_obj.acoustic_shader.restitution
        
        # Normal impulse
        normal_speed = np.abs(np.dot(relative_velocity, contact_normal))
        normal_impulse = particle_mass * (1 + restitution) * normal_speed
        
        # Impact duration (approximate)
        if normal_speed > 0:
            impact_duration = 2.94 * (particle_mass / E_star)**0.4 * (1/R_eff)**0.2 / normal_speed**0.2
        else:
            impact_duration = 0.001
        
        # Normal force
        normal_force_mag = normal_impulse / max(impact_duration, 1e-6)
        normal_force = normal_force_mag * contact_normal
        
        # Tangential force (friction)
        friction = 0.3
        if hasattr(config_obj.acoustic_shader, 'friction'):
            friction = config_obj.acoustic_shader.friction
        
        tangential_speed = np.linalg.norm(tangential_velocity)
        if tangential_speed > 0:
            tangential_force_mag = min(friction * normal_force_mag, particle_mass * tangential_speed / max(impact_duration, 1e-6))
            tangential_direction = tangential_velocity / tangential_speed
            tangential_force = tangential_force_mag * tangential_direction
        else:
            tangential_force = np.zeros(3)
            tangential_force_mag = 0
        
        # Determine contact type
        if event['type'] == CollisionType.IMPACT:
            contact_type = ContactType.IMPACT
        else:
            # For continuous contact, determine if sliding or rolling
            if tangential_speed > 0.1 * normal_speed:
                contact_type = ContactType.SLIDING
            else:
                contact_type = ContactType.STATIC
        
        # Create force data sequence
        frames = np.array([frame])
        
        # Create single-frame force data
        force_data = ForceData(
            frame=frame,
            obj1_idx=config_obj.idx,
            obj2_idx=-1,  # Particle index placeholder
            restitution=restitution,
            relative_velocity=relative_velocity,
            normal_velocity=normal_velocity,
            normal_force=normal_force,
            tangential_force=tangential_force,
            tangential_velocity=tangential_velocity,
            normal_force_magnitude=normal_force_mag,
            tangential_force_magnitude=tangential_force_mag,
            stochastic_normal_force=normal_force,
            stochastic_tangential_force=tangential_force,
            contact_type=contact_type,
            contact_point=event.get('contact_point'),
            contact_radius=R_particle,
            rolling_radius=R_particle,
            impact_duration=impact_duration if event['type'] == CollisionType.IMPACT else None,
            contact_pressure=normal_force_mag / (np.pi * R_particle**2) if R_particle > 0 else 0,
            penetration_depth=0.0,
            coupling_strength=0.5  # Default coupling strength
        )
        
        # Create sequence with single frame
        force_sequence = ForceDataSequence(
            frames=frames,
            obj_idx=config_obj.idx,
            other_obj_idx=-1,
            restitution=np.array([restitution]),
            relative_velocity=np.array([relative_velocity]),
            normal_velocity=np.array([normal_velocity]),
            normal_force=np.array([normal_force]),
            tangential_force=np.array([tangential_force]),
            tangential_velocity=np.array([tangential_velocity]),
            normal_force_magnitude=np.array([normal_force_mag]),
            tangential_force_magnitude=np.array([tangential_force_mag]),
            stochastic_normal_force=np.array([normal_force]),
            stochastic_tangential_force=np.array([tangential_force]),
            contact_type=np.array([contact_type]),
            contact_point=np.array([event.get('contact_point', np.zeros(3))]),
            contact_radius=np.array([R_particle]),
            rolling_radius=np.array([R_particle]),
            impact_duration=np.array([impact_duration if event['type'] == CollisionType.IMPACT else 0.0]),
            contact_pressure=np.array([normal_force_mag / (np.pi * R_particle**2) if R_particle > 0 else 0.0]),
            penetration_depth=np.array([0.0]),
            coupling_strength=np.array([0.5])
        )
        
        return force_sequence
    
    def _get_object_mass(self, config_obj: ObjectConfig, obj_trajectory: Any, frame: float) -> float:
        """Get object mass."""
        try:
            vertices = obj_trajectory.get_vertices(frame)
            faces = obj_trajectory.get_faces()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.density = config_obj.acoustic_shader.density if config_obj.acoustic_shader else 1000.0
            mass = mesh.mass
            return mass if mass > 9e-5 else 0.0001
        except:
            return 0.001
    
    def _get_object_effective_radius(self, config_obj: ObjectConfig, obj_trajectory: Any, frame: float) -> float:
        """Get object effective radius."""
        try:
            vertices = obj_trajectory.get_vertices(frame)
            center = np.mean(vertices, axis=0)
            radii = np.linalg.norm(vertices - center, axis=1)
            return float(np.mean(radii))
        except:
            return 0.1
    
    def _find_contact_vertices(
        self,
        event: Dict[str, Any],
        obj_vertices: np.ndarray,
        obj_faces: np.ndarray,
        contact_point: np.ndarray,
        collision_margin: float
    ) -> List[int]:
        """
        Find vertices on the object that are in contact with the particle.
        """
        # Build KD-tree for object vertices
        tree = cKDTree(obj_vertices)
        
        # Find vertices within collision margin of contact point
        radius = collision_margin * 2.0
        vertices_idx = tree.query_ball_point(contact_point, radius, workers=-1)
        
        if len(vertices_idx) > 0:
            # Find faces that contain these vertices
            vertices_idx = np.array(vertices_idx)
            faces_idx = np.where(np.any(np.isin(obj_faces, vertices_idx), axis=1))[0]
            
            # Get unique vertices from these faces
            if len(faces_idx) > 0:
                contact_vertices = np.unique(obj_faces[faces_idx].flatten())
                return contact_vertices.tolist()
        
        return vertices_idx.tolist() if len(vertices_idx) > 0 else []
    
    def _create_score_event(
        self,
        score_track: ScoreTrack,
        event: Dict[str, Any],
        contact_vertices: List[int],
        force_data: ForceDataSequence,
        obj_vertices: np.ndarray,
        obj_faces: np.ndarray
    ) -> None:
        """
        Create a score event for the collision.
        """
        # Determine frame range
        start_frame = int(event['frame'])
        duration = event.get('duration', 1)
        end_frame = start_frame + duration
        
        # Get total samples from trajectory
        # For now, use a reasonable estimate
        total_samples = int(end_frame + 1000)  # Add some padding for decay
        
        # Initialize score arrays
        score_type = np.zeros((total_samples, 1),), dtype=np.int32)
        score_vertex_ids = np.zeros((total_samples, len(obj_vertices)), dtype=np.bool_)
        score_contact_area = np.zeros((total_samples, 1), dtype=np.float32)
        
        # Set contact type (1=impact, 2=scraping, 3=sliding, 4=rolling)
        if event['type'] == CollisionType.IMPACT:
            contact_type = 1
        else:
            contact_type = 3  # Default to sliding for continuous contact
        
        # Fill in score data
        for sample_idx in range(start_frame, min(end_frame, total_samples)):
            score_type[sample_idx] = contact_type
            score_contact_area[sample_idx] = len(contact_vertices) / len(obj_faces)
            
            # Set vertex IDs
            for v_idx in contact_vertices:
                if v_idx < len(obj_vertices):
                    score_vertex_ids[sample_idx, v_idx] = True
        
        # Create score event
        score_event = ScoreEvent(
            coll_obj=-1,  # Particle index placeholder
            start_sample=start_frame,
            stop_sample=end_frame,
            type=score_type,
            vertex_ids=score_vertex_ids,
            contact_area=score_contact_area,
            force=None,
            coupling_data=None
        )
        
        # Add to score track
        score_track.add_event(score_event)
    
    def compute_batch(self, particles_indices: List[int], obj_indices: List[int] = None) -> None:
        """
        Compute collisions for multiple particle systems in parallel.
        
        Parameters:
        -----------
        particles_indices : List[int]
            List of particle system indices
        obj_indices : List[int], optional
            List of object indices to check collisions with
        """
        tasks = []
        for particles_idx in particles_indices:
            if obj_indices:
                for obj_idx in obj_indices:
                    tasks.append(self._delayed_compute(particles_idx, obj_idx))
            else:
                tasks.append(self._delayed_compute(particles_idx, None))
        
        if tasks:
            compute(*tasks)
    
    @delayed
    def _delayed_compute(self, particles_idx: int, obj_idx: int = None) -> None:
        """Delayed computation for parallel processing."""
        self.compute(particles_idx, obj_idx)
    
    def save_collision_data(self) -> None:
        """Save all collision data to disk."""
        # Save collisions
        collisions = self.entity_manager.get('collisions')
        for c_idx in collisions.keys():
            if hasattr(collisions[c_idx], 'particle_idx'):
                collisions[c_idx].save(f"{self.collisions_dir}/{c_idx:05d}.pkl")
        
        # Save forces
        forces = self.entity_manager.get('forces')
        for f_idx in forces.keys():
            if hasattr(forces[f_idx], 'particle_idx'):
                forces[f_idx].save(f"{self.forces_dir}/{f_idx:05d}.pkl")
        
        debug_print(f"Saved particle collision data to {self.collisions_dir}")
