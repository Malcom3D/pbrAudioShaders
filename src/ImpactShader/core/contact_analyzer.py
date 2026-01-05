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
import math
import trimesh
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
import glob
import re
from scipy import signal
from scipy.spatial import ConvexHull

from ..core.impact_manager import ImpactManager
from ..lib.impacts_data import ImpactEvent, ObjectContact, ContactForce, ContactType

@dataclass
class ContactAnalyzer:
    """Analyzes contact interactions using physically-based stochastic models"""
    impact_manager: ImpactManager
    
    # Physical constants
    AIR_DENSITY: float = 1.225  # kg/m³
    SPEED_OF_SOUND: float = 343.0  # m/s
    
    # Contact analysis parameters
    SCRAPING_NOISE_BANDWIDTH: Tuple[float, float] = (100.0, 10000.0)  # Hz
    ROLLING_RESONANCE_RANGE: Tuple[float, float] = (50.0, 2000.0)  # Hz
    MIN_CONTACT_DURATION: float = 0.001  # s
    MAX_CONTACT_DURATION: float = 5.0  # s
    
    def compute(self) -> None:
        """MainMain computation function for contact analysis"""
        print('ContactAnalyzer.compute')
        
        config = self.impact_manager.get('config')
        
        # Step 1: Load optimized meshes for all objects
        meshes = self._load_object_meshes(config.objects)
        
        # Step 2: Read and interpolate object trajectories
        trajectories = self._read_object_trajectories(config.objects)
        interpolated_trajectories = self._interpolate_trajectories(
            trajectories, 
            config.system.sample_rate
        )
        
        # Step 3: Detect and analyze contact events
        contact_events = self._analyze_contacts(
            meshes, 
            interpolated_trajectories,
            config.objects,
            config.system.collision_margin
        )
        
        # Step 4: Register contact events with ImpactManager
        for event_idx, event in enumerate(contact_events):
            self.impact_manager.register(event)
            
        print(f"Total contact events detected: {len(contact_events)}")
    
    def _load_object_meshes(self, objects: List) -> Dict[int, Dict]:
        """Load and prepare meshes for all objects"""
        meshes = {}
        
        for obj in objects:
            optimized_obj_path = os.path.join(obj.obj_path, f"optimized_{obj.name}.obj")
            
            if not os.path.exists(optimized_obj_path):
                raise FileNotFoundError(f"Optimized mesh not found: {optimized_obj_path}")
            
            mesh = trimesh.load(optimized_obj_path, force='mesh')
            
            # Calculate convex hull for smooth surface approximation
            convex_hull = None
            if len(mesh.vertices) > 3:
                try:
                    hull = ConvexHull(mesh.vertices)
                    convex_hull = mesh.vertices[hull.vertices]
                except:
                    convex_hull = mesh.vertices
            
            meshes[obj.idx] = {
                'mesh': mesh,
                'convex_hull': convex_hull,
                'volume': mesh.volume,
                'surface_area': mesh.area,
                'bounds': mesh.bounds
            }
            
            print(f"Loaded mesh for object {obj.idx}: {len(mesh.vertices)} vertices, "
                  f"volume: {mesh.volume:.6f} m³")
        
        return meshes
    
    def _read_object_trajectories(self, objects: List) -> Dict[int, Dict]:
        """Read object trajectories from obj files"""
        config = self.impact_manager.get('config')
        trajectories = {}
        
        for obj in objects:
            obj_idx = obj.idx
            obj_path = obj.obj_path
            
            obj_files = glob.glob(os.path.join(obj_path, "*.obj"))
            obj_files = [f for f in obj_files if not f.endswith(f"optimized_{obj.name}.obj")]
            
            if not obj_files:
                raise FileNotFoundError(f"No obj files found in {obj_path}")
            
            obj_files.sort(key=self._extract_frame_number)
            
            positions = []
            rotations = []
            times = []
            
            for i, obj_file in enumerate(obj_files):
                mesh = trimesh.load(obj_file, force='mesh')
                position = mesh.center_mass
                positions.append(position)
                
                if hasattr(mesh, 'principal_inertia_transform'):
                    transform = mesh.principal_inertia_transform
                    rotation = transform[:3, :3]
                else:
                    rotation = np.eye(3)
                
                rotations.append(rotation)
                times.append(i / config.system.fps)
            
            trajectories[obj_idx] = {
                'positions': np.array(positions),
                'rotations': np.array(rotations),
                'times': np.array(times),
                'frames': len(obj_files)
            }
        
        return trajectories
    
    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename"""
        basename = os.path.basename(filename)
        numbers = re.findall(r'\d+', basename)
        return int(numbers[-1]) if numbers else 0
    
    def _interpolate_trajectories(self, trajectories: Dict, sample_rate: float) -> Dict[int, Dict]:
        """Interpolate trajectories to audio sample rate"""
        interpolated = {}
        
        for obj_idx, traj in trajectories.items():
            times = traj['times']
            positions = traj['positions']
            rotations = traj['rotations']
            
            if len(times) > 1:
                # Position interpolation
                pos_interp = interp1d(
                    times, positions, axis=0, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                
                # Rotation interpolation using quaternions
                from scipy.spatial.transform import Rotation
                rot_matrices = rotations
                rots = Rotation.from_matrix(rot_matrices)
                quats = rots.as_quat()
                
                quat_interp = interp1d(
                    times, quats, axis=0, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                
                total_time = times[-1]
                new_times = np.arange(0, total_time, 1/sample_rate)
                if new_times[-1] < total_time:
                    new_times = np.append(new_times, total_time)
                
                new_positions = pos_interp(new_times)
                new_quats = quat_interp(new_times)
                new_rots = Rotation.from_quat(new_quats).as_matrix()
                
                velocities = np.gradient(new_positions, new_times, axis=0)
                accelerations = np.gradient(velocities, new_times, axis=0)
                angular_velocities = self._calculate_angular_velocities(new_rots, new_times)
                
                interpolated[obj_idx] = {
                    'times': new_times,
                    'positions': new_positions,
                    'rotations': new_rots,
                    'velocities': velocities,
                    'accelerations': accelerations,
                    'angular_velocities': angular_velocities,
                    'sample_rate': sample_rate,
                    'num_samples': len(new_times)
                }
            else:
                interpolated[obj_idx] = {
                    'times': times,
                    'positions': positions,
                    'rotations': rotations,
                    'velocities': np.zeros_like(positions),
                    'accelerations': np.zeros_like(positions),
                    'angular_velocities': np.zeros((1, 3)),
                    'sample_rate': sample_rate,
                    'num_samples': len(times)
                }
        
        return interpolated
    
    def _calculate_angular_velocities(self, rotations: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate angular velocities from rotation matrices"""
        n_samples = len(rotations)
        angular_velocities = np.zeros((n_samples, 3))
        
        if n_samples < 2:
            return angular_velocities
        
        for i in range(1, n_samples):
            dt = times[i] - times[i-1]
            if dt > 0:
                R_diff = rotations[i] @ rotations[i-1].T
                skew_sym = 0.5 * (R_diff - R_diff.T)
                wx = skew_sym[2, 1]
                wy = skew_sym[0, 2]
                wz = skew_sym[1, 0]
                angular_velocities[i] = np.array([wx, wy, wz]) / dt
        
        angular_velocities[0] = angular_velocities[1]
        return angular_velocities
    
    def _analyze_contacts(self, meshes: Dict, trajectories: Dict, 
                         objects: List, collision_margin: float) -> List[ImpactEvent]:
        """Analyze contacts between objects over time"""
        contact_events = []
        object_indices = list(meshes.keys())
        
        # Get all object pairs
        object_pairs = []
        for i in range(len(object_indices)):
            for j in range(i + 1, len(object_indices)):
                object_pairs.append((object_indices[i], object_indices[j]))
        
        # Track ongoing contacts
        ongoing_contacts = {}
        
        sample_rate = trajectories[object_indices[0]]['sample_rate']
        num_samples = trajectories[object_indices[0]]['num_samples']
        
        print(f"Analyzing contacts at {sample_rate}Hz for {num_samples} samples...")
        
        for time_idx in range(num_samples):
            current_time = trajectories[object_indices[0]]['times'][time_idx]
            
            for obj1_idx, obj2_idx in object_pairs:
                # Check for contact at this time
                contact_info = self._check_contact_at_time(
                    time_idx, current_time,
                    obj1_idx, obj2_idx,
                    meshes, trajectories,
                    collision_margin
                )
                
                if contact_info:
                    # Check if this is a continuation of an ongoing contact
                    contact_key = (obj1_idx, obj2_idx)
                    
                    if contact_key in ongoing_contacts:
                        # Update ongoing contact
                        ongoing_contacts[contact_key]['end_time'] = current_time
                        ongoing_contacts[contact_key]['contacts'].append(contact_info)
                    else:
                        # Start new new contact
                        ongoing_contacts[contact_key] = {
                            'start_time': current_time,
                            'end_time': current_time,
                            'contacts': [contact_info],
                            'obj1_idx': obj1_idx,
                            'obj2_idx': obj2_idx
                        }
                else:
                    # Contact ended, create event if duration is significant
                    contact_key = (obj1_idx, obj2_idx)
                    if contact_key in ongoing_contacts:
                        contact_data = ongoing_contacts.pop(contact_key)
                        duration = contact_data['end_time'] - contact_data['start_time']
                        
                        if duration >= self.MIN_CONTACT_DURATION:
                            event = self._create_contact_event(
                                contact_data, meshes, trajectories, objects
                            )
                            if event:
                                contact_events.append(event)
        
        # Process any remaining ongoing contacts at the end
        for contact_key, contact_data in ongoing_contacts.items():
            duration = contact_data['end_time'] - contact_data['start_time']
            if duration >= self.MIN_CONTACT_DURATION:
                event = self._create_contact_event(
                    contact_data, meshes, trajectories, objects
                )
                if event:
                    contact_events.append(event)
        
        return contact_events
    
    def _check_contact_at_time(self, time_idx: int, current_time: float,
                              obj1_idx: int, obj2_idx: int,
                              meshes: Dict, trajectories: Dict,
                              collision_margin: float) -> Optional[Dict]:
        """Check for contact between two objects at a specific time"""
        # Get current states
        pos1 = trajectories[obj1_idx]['positions'][time_idx]
        rot1 = trajectories[obj1_idx]['rotations'][time_idx]
        vel1 = trajectories[obj1_idx]['velocities'][time_idx]
        ang_vel1 = trajectories[obj1_idx]['angular_velocities'][time_idx]
        
        pos2 = trajectories[obj2_idx]['positions'][time_idx]
        rot2 = trajectories[obj2_idx]['rotations'][time_idx]
        vel2 = trajectories[obj2_idx]['velocities'][time_idx]
        ang_vel2 = trajectories[obj2_idx]['angular_velocities'][time_idx]
        
        # Transform meshes
        mesh1 = self._transform_mesh(meshes[obj1_idx]['mesh'], pos1, rot1)
        mesh2 = self._transform_mesh(meshes[obj2_idx]['mesh'], pos2, rot2)
        
        # Check collision
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object(f'obj_{obj1_idx}', mesh1)
        collision_manager.add_object(f'obj_{obj2_idx}', mesh2)
        
        in_collision, _ = collision_manager.in_collision_single(
            mesh=mesh1, return_data=True
        )
        
        if in_collision:
            # Get detailed contact information
            distance1, data1 = collision_manager.min_distance_single(mesh=mesh1, return_data=True)
            distance2, data2 = collision_manager.min_distance_single(mesh=mesh2, return_data=True)
            
            if 0 < abs(distance2) <= collision_margin:
                # Get contact points
                point1 = data1.point(f'obj_{obj1_idx}')
                point2 = data2.point(f'obj_{obj2_idx}')
                
                # Calculate contact normal
                normal = point2 - point1
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 0:
                    normal = normal / normal_norm
                else:
                    normal = pos2 - pos1
                    normal_n_norm = np.linalg.norm(normal)
                    if normal_norm > 0:
                        normal = normal / normal_norm
                    else:
                        normal = np.array([1, 0, 0])
                
                # Calculate relative velocity at contact point
                r1 = point1 - (pos1 + meshes[obj1_idx]['mesh'].center_mass)
                r2 = point2 - (pos2 + meshes[obj2_idx]['mesh'].center_mass)
                
                v1 = vel1 + np.cross(ang_vel1, r1)
                v2 = vel2 + np.cross(ang_vel2, r2)
                v_rel = v2 - v1
                
                # Find nearest vertices
                dist1, vertex_idx1 = self._nearest_vertex(mesh1, point1)
                dist2, vertex_idx2 = self._nearest_vertex(mesh2, point2)
                
                # Determine contact type
                contact_type = self._determine_contact_type(v_rel, normal, ang_vel1, ang_vel2)
                
                return {
                    'time': current_time,
                    'point1': point1,
                    'point2': point2,
                    'normal': normal,
                    'distance': distance2,
                    'vertex_idx1': vertex_idx1,
                    'vertex_idx2': vertex_idx2,
                    'v_rel': v_rel,
                    'contact_type': contact_type,
                    'obj1_idx': obj1_idx,
                    'obj2_idx': obj2_idx
                }
        
        return None
    
    def _determine_contact_type(self, v_rel: np.ndarray, normal: np.ndarray,
                               ang_vel1: np.ndarray, ang_vel2: np.ndarray) -> ContactType:
        """Determine the type of contact based on velocities"""
        # Normal component of relative velocity
        v_normal = np.dot(v_rel, normal)
        
        # Tangential component
        v_tangent = v_rel - v_normal * normal
        v_tangent_mag = np.linalg.norm(v_tangent)
        
        # Angular velocity magnitudes
        ang_vel1_mag = np.linalg.norm(ang_vel1)
        ang_vel2_mag = np.linalg.norm(ang_vel2)
        
        # Determine contact type based on paper criteria
        if abs(v_normal) > 0.1:  # Significant normal velocity -> impact
            return ContactType.IMPACT
        elif v_tangent_mag > 0.05 and (ang_vel1_mag < 1.0 or ang_vel2_mag < 1.0):
            # Significant tangential velocity without much rotation -> scraping
            return ContactType.SCRAPING
        elif ang_vel1_mag > 0.5 or ang_vel2_mag > 0.5:
            # Significant rotation -> rolling
            return ContactType.ROLLING
        else:
            # Mixed or uncertain
            return ContactType.MIXED
    
    def _create_contact_event(self, contact_data: Dict, meshes: Dict, 
                             trajectories: Dict, objects: List) -> Optional[ImpactEvent]:
        """Create a contact event from accumulated contact data"""
        if not contact_data['contacts']:
            return None
        
        # Calculate average contact location
        avg_point = np.mean([c['point1'] for c in contact_data['contacts']], axis=0)
        
        # Create event
        event_idx = len(self.impact_manager.get('impacts'))
        event = ImpactEvent(
            idx=event_idx,
            start_time=contact_data['start_time'],
            end_time=contact_data['end_time'],
            duration=contact_data['end_time'] - contact_data['start_time'],
            coord=tuple(avg_point)
        )
        
        # Process contacts for each object
        obj_contacts = {}
        
        for contact in contact_data['contacts']:
            # Object 1
            if contact['obj1_idx'] not in obj_contacts:
                obj_contacts[contact['obj1_idx']] = ObjectContact(
                    obj_idx=contact['obj1_idx']
                )
            
            # Calculate force for object 1
            force1 = self._calculate_contact_force(
                contact, contact['obj1_idx'], contact['obj2_idx'],
                meshes, trajectories, is_first_object=True
            )
            
            contact_force1 = ContactForce(
                id_vertex=contact['vertex_idx1'],
                force_vector=force1,
                contact_type=contact['contact_type'],
                contact_normal=contact['normal'],
                relative_velocity=contact['v_rel'],
                duration=event.duration
            )
            
            obj_contacts[contact['obj1_idx']].add_contact(contact_force1)
            
            # Object 2
            if contact['obj2_idx'] not in obj_contacts:
                obj_contacts[contact['obj2_idx']] = ObjectContact(
                    obj_idx=contact['obj2_idx']
                )
            
            # Calculate force for object 2 (opposite direction)
            force2 = self._calculate_contact_force(
                contact, contact['obj2_idx'], contact['obj1_idx'],
                meshes, trajectories, is_first_object=False
            )
            
            contact_force2 = ContactForce(
                id_vertex=contact['vertex_idx2'],
                force_vector=force2,
                contact_type=contact['contact_type'],
                contact_normal=-contact['normal'],  # Opposite normal
                relative_velocity=-contact['v_rel'],  # Opposite velocity
                duration=event.duration
            )
            
            obj_contacts[contact['obj2_idx']].add_contact(contact_force2)
        
        # Add object contacts to event
        for obj_contact in obj_contacts.values():
            event.add_object_contact(obj_contact)
        
        print(event.generate_contact_description())
        return event
    
    def _calculate_contact_force(self, contact: Dict, obj_idx: int, other_idx: int,
                                meshes: Dict, trajectories: Dict,
                                is_first_object: bool) -> np.ndarray:
        """Calculate contact force using physically-based model"""
        # Get object properties
        mesh_data = meshes[obj_idx]
        volume = mesh_data['volume']
        
        # Find object configuration for material properties
        config = self.impact_manager.get('config')
        obj_config = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                obj_config = obj
                break
        
        if obj_config is None:
            # Default properties
            density = 1000.0  # kg/m³
            young_modulus = 1e9  # Pa
        else:
            density = obj_config.density if obj_config.density else 1000.0
            young_modulus = obj_config.young_modulus if obj_config.young_modulus else 1e9
        
        # Calculate mass
        mass = density * volume
        
        # Get velocities
        v_rel = contact['v_rel']
        normal = contact['normal']
        
        # Normal component
        v_normal = np.dot(v_rel, normal)
        
        # Calculate normal force using Hertzian contact theory
        # Simplified version for sphere-sphere contact
        if v_normal < 0:  # Approaching
            # Effective radius (simplified)
            R_eff = 0.01  # m, approximate
            
            # Effective Young modulus
            E_eff = young_modulus / (2 * (1 - 0.3**2))  # Poisson ratio = 0.3
            
            # Maximum deformation (simplified)
            delta_max = abs(v_normal) * 0.001  # Small time step
            
            # Hertzian contact force
            F_normal = (4/3) * E_eff * np.sqrt(R_eff) * (delta_max ** 1.5)
            F_normal = min(F_normal, mass * 9.81 * 10)  # Limit to 10g
            
            # Direction is along normal
            force_normal = F_normal * (-normal if v_normal < 0 else normal)
        else:
            force_normal = np.zeros(3)
        
        # Tangential force (friction)
        v_tangent = v_rel - v_normal * normal
        v_tangent_mag = np.linalg.norm(v_tangent)
        
        if v_tangent_mag > 0:
            # Coulomb friction
            friction_coeff = 0.3  # Dynamic friction coefficient
            force_friction = -friction_coeff * F_normal * (v_tangent / v_tangent_mag)
        else:
            force_friction = np.zeros(3)
        
        # Total force
        total_force = force_normal + force_friction
        
        # Scale based on contact type
        if contact['contact_type'] == ContactType.SCRAPING:
            total_force *= 0.5  # Reduced force for scraping
        elif contact['contact_type'] == ContactType.ROLLING:
            total_force *= 0.3  # Further reduced for rolling
        
        return total_force
    
    def _transform_mesh(self, mesh: trimesh.Trimesh, 
                       position: np.ndarray, 
                       rotation: np.ndarray) -> trimesh.Trimesh:
        """Transform mesh to given position and rotation"""
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        
        transformed = mesh.copy()
        transformed.apply_transform(transform)
        
        return transformed
    
    def _nearest_vertex(self, mesh: trimesh.Trimesh, point: np.ndarray) -> Tuple[float, int]:
        """Find nearest vertex to a point"""
        vertices = mesh.vertices
        distances = np.linalg.norm(vertices - point, axis=1)
        min_idx = np.argmin(distances)
        return distances[min_idx], int(min_idx)

    def get_trajectories(self) -> Dict:
        """
        Get the interpolated trajectories for acceleration noise generation.
        
        Returns:
            Dictionary of trajectory data for all objects
        """
        return self.interpolated_trajectories if hasattr(self, 'interpolated_trajectories') else {}

