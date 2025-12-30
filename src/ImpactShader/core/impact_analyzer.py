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
from dask import delayed, compute
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
import glob
import re

from ..core.impact_manager import ImpactManager
from ..lib.impacts_data import ImpactData, ObjCollision, Collision

@dataclass
class ImpactAnalyzer:
    impact_manager:: ImpactManager
    
    def compute(self) -> None:
        print('ImpactAnalyzer.compute')
        """
        Main computation function that:
        1. Loads optimized meshes for all objects
        2. Reads and interpolates object trajectories from obj files
        3. Detects collisions between objects over time
        4. Computes impact forces and collision details using proper physics
        5. Registers ImpactData instances with ImpactManager
        """
        config = self.impact_manager.get('config')
        
        # Step 1: Load optimized meshes for all objects
        meshes = {}
        masses = {}
        inertia_tensors = {}
        centers_of_mass = {}
        
        for config_obj in config.objects:
            # Load optimized mesh
            optimized_obj_path = os.path.join(config_obj.obj_path, f"optimized_{config_obj.name}.obj")
            
            if not os.path.exists(optimized_obj_path):
                raise FileNotFoundError(f"Optimized mesh not found: {optimized_obj_path}")
            
            mesh = trimesh.load(optimized_obj_path, force='mesh')
            meshes[config_obj.idx] = mesh
            
            # Calculate mass, center of mass, and inertia tensor using trimesh
            volume = mesh.volume
            mass = config_obj.density * volume if config_obj.density else 1.0  # Default mass if density not specified
            masses[config_obj.idx] = mass
            
            # Get center of mass (trimesh calculates this properly)
            center_of_mass = mesh.center_mass
            centers_of_mass[config_obj.idx] = center_of_mass
            
            # Calculate inertia tensor using trimesh's moment of inertia calculation
            # This gives the inertia tensor about the center of mass
            inertia_tensor = mesh.moment_inertia
            inertia_tensors[config_obj.idx] = inertia_tensor
            
            print(f"Object {config_obj.idx} ({config_obj.name}):")
            print(f"  Mass: {mass:.4f} kg")
            print(f"  Volume: {volume:.6f} m³")
            print(f"  Center of mass: {center_of_mass}")
            print(f"  Inertia tensor:\n{inertia_tensor}")
        
        # Step 2: Read object trajectories from obj files
        trajectories = self._read_object_trajectories(config.objects)
        
        # Step 3: Interpolate trajectories to audio sample rate temporal resolution
        interpolated_trajectories = self._interpolate_trajectories(
            trajectories, 
            config.system.sample_rate
        )
        
        # Step 4: Detect collisions over time
        impacts = self._detect_collisions(
            meshes, 
            interpolated_trajectories, 
            masses, 
            inertia_tensors,
            centers_of_mass
        )
        
        # Step 5: Register impacts with ImpactManager
        for impact_idx, impact_data in enumerate(impacts):
            self.impact_manager.register(impact_data)
            
        print(f"Total impacts detected: {len(impacts)}")
    
    def _read_object_trajectories(self, objects: List) -> Dict[int, Dict]:
        """
        Read object trajectories from obj files.
        
        Returns:
            Dictionary mapping object idx to trajectory data
        """
        config = self.impact_manager.get('config')
        trajectories = {}
        
        for obj in objects:
            obj_idx = obj.idx
            obj_path = obj.obj_path
            
            # Get all obj files in the directory
            obj_files = glob.glob(os.path.join(obj_path, "*.obj"))
            
            # Filter out optimized mesh if present
            obj_files = [f for f in obj_files if not f.endswith(f"optimized_{obj.name}.obj")]
            
            if not obj_files:
                raise FileNotFoundError(f"No obj files found in {obj_path}")
            
            # Sort files by frame number
            obj_files.sort(key=self._extract_frame_number)
            
            # Read positions and rotations from each frame
            positions = []
            rotations = []
            times = []
            
            for i, obj_file in enumerate(obj_files):
                mesh = trimesh.load(obj_file, force='mesh')
                
                # Get center of mass position
                position = mesh.center_mass
                positions.append(position)
                
                # Get rotation matrix from mesh transformation
                # trimesh stores the transformation in mesh.principal_inertia_transform
                # or we can extract it from the mesh's vertices
                if hasattr(mesh, 'principal_inertia_transform'):
                    transform = mesh.principal_inertia_transform
                    rotation = transform[:3, :3]
                else:
                    # Fallback: use identity matrix
                    rotation = np.eye(3)
                
                rotations.append(rotation)
                
                # Time based on frame number
                times.append(i(i / config.system.fps)
            
            trajectories[obj_idx] = {
                'positions': np.array(positions),
                'rotations': np.array(rotations),
                'times': np.array(times),
                'frames': len(obj_files)
            }
            
            print(f"Object {obj_idx} trajectory: {len(obj_files)} frames, "
                  f"duration: {times[-1]:.3f}s")
        
        return trajectories
    
    def _extract_frame_number(self, filename: str) -> int:
        """
        Extract frame number from filename.
        Supports formats like: icosphere0001.obj, 0001.obj, frame_001.obj
        """
        basename = os.path.basename(filename)
        
        # Try to find numbers in the filename
        numbers = re.findall(r'\d+', basename)
        
        if numbers:
            return int(numbers[-1])  # Use the last number found
        
        # If no numbers found, use alphabetical order
        return 0
    
    def _interpolate_trajectories(self, trajectories: Dict, sample_rate: float) -> Dict[int, Dict]:
        """
        Interpolate trajectories to audio sample rate temporal resolution for collision detection.
        
        Args:
            trajectories: Original trajectory data
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Interpolated trajectories with audio sample rate temporal resolution
        """
        interpolated = {}
        
        for obj_idx, traj in trajectories.items():
            times = traj['times']
            positions = traj['positions']
            rotations = traj['rotations']
            
            # Create interpolation functions
            if len(times) > 1:
                # Linear interpolation for positions
                pos_interp = interp1d(
                    times, 
                    positions, 
                    axis=0, 
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                # For rotations, we need to interpolate quaternions for proper rotation interpolation
                # Convert rotation matrices to quaternions first
                from scipy.spatial.transform import Rotation
                
                # Convert rotation matrices to quaternions
                rot_matrices = rotations
                rots = Rotation.from_matrix(rot_matrices)
                quats = rots.as_quat()  # [x, y, z, w] format
                
                # Create quaternion interpolation function
                quat_interp = interp1d(
                    times,
                    quats,
                    axis=0,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                # Create new time array with audio sample rate resolution
                total_time = times[-1]
                new_times = np.arange(0, total_time, 1/sample_rate)
                
                
                # Ensure we include the last time point
                if new_times[-1] < total_time:
                    new_times = np.append(new_times, total_time)
                
                # Interpolate positions
                new_positions = pos_interp(new_times)
                
                # Interpolate quaternions and convert back to rotation matrices
                new_quats = quat_interp(new_times)
                new_rots = Rotation.from_quat(new_quats)
                new_rotations = new_rots.as_matrix()
                
                # Calculate velocities and accelerations (for force calculation)
                velocities = np.gradient(new_positions, new_times, axis=0)
                accelerations = np.gradient(velocities, new_times, axis=0)
                
                # Calculate angular velocities from rotation matrices
                angular_velocities = self._calculate_angular_velocities(
                    new_rotations, new_times
                )
                
                interpolated[obj_idx] = {
                    'times': new_times,
                    'positions': new_positions,
                    'rotations': new_rotations,
                    'velocities': velocities,
                    'accelerations': accelerations,
                    'angular_velocities': angular_velocities,
                    'sample_rate': sample_rate,
                    'num_samples': len(new_times)
                }
                
                print(f"Object {obj_idx} interpolated: {len(new_times)} samples "
                      f"at {sample_rate}Hz, duration: {new_times[-1]:.3f}s")
            else:
                # Single frame case
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
        """
        Calculate angular velocities from rotation matrices.
        
        Args:
            rotations: Array of rotation matrices [n_samples, 3, 3]
            times: Array of time points
            
        Returns:
            Array of angular velocity vectors [n_samples, 3]
        """
        n_samples = len(rotations)
        angular_velocities = np.zeros((n_samples, 3))
        
        if n_samples < 2:
            return angular_velocities
        
        # Calculate angular velocity using finite differences of rotation matrices
        for i in range(1, n_samples):
            dt = times[i] - times[i-1]
            if dt > 0:
                # R_diff = R_i * R_{i-1}^T
                R_diff = rotations[i] @ rotations[i-1].T
                
                # Extract skew-symmetric part: (R_diff - R_diff^T)/2
                skew_sym = 0.5 * (R_diff - R_diff.T)
                
                # Extract angular velocity vector from skew-symmetric matrix
                # [0, -wz, wy; wz, 0, -wx; -wy, wx, 0]
                wx = skew_sym[2, 1]
                wy = skew_sym[0, 2]
                wz = skew_sym[1, 0]
                
                angular_velocities[i] = np.array([wx, wy, wz]) / dt
        
        # Forward difference for first element
        angular_velocities[0] = angular_velocities[1]
        
        return angular_velocities
    
    def _detect_collisions(self, meshes: Dict, trajectories: Dict, 
                          masses: Dict, inertia_tensors: Dict,
                          centers_of_mass: Dict) -> List[ImpactData]:
        """
        Detect collisions between objects over time.
        
        Returns:
            List of ImpactData objects for each detected collision
        """
        impacts = []
        
        # Get all object pairs for collision checking
        object_indices = list(meshes.keys())
        object_pairs = []
        
        for i in range(len(object_indices)):
            for j in range(i + 1, len(object_indices)):
                object_pairs.append((object_indices[i], object_indices[j]))
        
        # Get collision margin from config
        config = self.impact_manager.get('config')
        collision_margin = config.system.collision_margin
        
        # Check collisions at each time step (using audio sample rate resolution)
        sample_rate = trajectories[object_indices[0]]['sample_rate']
        num_samples = trajectories[object_indices[0]]['num_samples']
        
        print(f"Checking collisions at {sample_rate}Hz for {num_samples} samples...")
        
        # Use Dask for parallel collision detection
        tasks = []
        for time_idx in range(num_samples):
            current_time = trajectories[object_indices[0]]['times'][time_idx]
            
            for obj1_idx, obj2_idx in object_pairs:
                task = delayed(self._check_collision_at_time)(
                    time_idx, current_time,
                    obj1_idx, obj2_idx,
                    meshes, trajectories,
                    masses, inertia_tensors, centers_of_mass,
                    collision_margin
                )
                tasks.append(task)
        
        # Execute all collision checks in parallel
        results = compute(*tasks)
        
        # Flatten results and filter out None (no collision)
        for result in results:
            if result is not None:
                impacts.append(result)
        
        return impacts
    
    def _check_collision_at_time(self, time_idx: int, current_time: float,
                                obj1_idx: int, obj2_idx: int,
                                meshes: Dict, trajectories: Dict,
                                masses: Dict, inertia_tensors: Dict,
                                centers_of_mass: Dict,
                                collision_margin: float) -> Optional[ImpactData]:
        """
        Check for collision between two objects at a specific time.
        
        Returns:
            ImpactData if collision detected, None otherwise
        """
        # Get current positions and rotations
        pos1 = trajectories[obj1_idx]['positions'][time_idx]
        rot1 = trajectories[obj1_idx]['rotations'][time_idx]
        vel1 = trajectories[obj1_idx]['velocities'][time_idx]
        ang_vel1 = trajectories[obj1_idx]['angular_velocities'][time_idx]
        
        pos2 = trajectories[obj2_idx]['positions'][time_idx]
        rot2 = trajectories[obj2_idx]['rotations'][time_idx]
        vel2 = trajectories[obj2_idx]['velocities'][time_idx]
        ang_vel2 = trajectories[obj2_idx]['angular_velocities'][time_idx]
        
        # Transform meshes to current positions and rotations
        mesh1_transformed = self._transform_mesh(
            meshes[obj1_idx], pos1, rot1
        )
        mesh2_transformed = self._transform_mesh(
            meshes[obj2_idx], pos2, rot2
        )
        
        # Check for collision using trimesh collision detection
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object(f'obj_{obj1_idx}', mesh1_transformed)
        collision_manager.add_object(f'obj_{obj2_idx}', mesh2_transformed)
        
        # Check if objects are in collision
        in_collision, collision_data = collision_manager.in_collision_single(
            mesh=mesh1_transformed,
            return_data=True
        )
        
        if in_collision:
            # Get detailed collision information
            distance, data = collision_manager.min_distance_single(
                mesh=mesh1_transformed,
                return_data=True
            )
            
            if distance <= collision_margin:
                # Get collision points
                point1 = data.point(f'obj_{obj1_idx}')
                point2 = data.point(f'obj_{obj2_idx}')
                
                # Calculate impact coordinate as midpoint between collision points
                impact_coord = (point1 + point2) / 2.0
                
                # Calculate collision normal (from obj1 to obj2 at collision point)
                normal = point2 - point1
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 0:
                    normal = normal / normal_norm
                else:
                    # If points are coincident, use direction between centers
                    normal = pos2 - pos1
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm > 0:
                        normal = normal / normal_norm
                    else:
                        normal = np.array([1, 0, 0])
                
                # Create ImpactData for this collision
                impact_data = self._create_impact_data(
                    impact_coord, obj1_idx, obj2_idx,
                    mesh1_transformed, mesh2_transformed,
                    pos1, pos2, vel1, vel2, ang_vel1, ang_vel2,
                    masses[obj1_idx], masses[obj2_idx],
                    inertia_tensors[obj1_idx], inertia_tensors[obj2_idx],
                    centers_of_mass[obj1_idx], centers_of_mass[obj2_idx],
                    normal, current_time,
                    collision_margin
                )
                
                return impact_data
        
        return None
    
    def _transform_m_mesh(self, mesh: trimesh.Trimesh, 
                       position: np.ndarray, 
                       rotation: np.ndarray) -> trimesh.Trimesh:
        """
        Transform mesh to given position and rotation.
        
        Returns:
            Transformed copy of the mesh
        """
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        
        # Apply transformation
        transformed = mesh.copy()
        transformed.apply_transform(transform)
        
        return transformed
    
    def _nearest_optimized_vertex(self, mesh: trimesh.Trimesh, 
                                 imp_coord: np.ndarray) -> Tuple[float, int]:
        """
        Find nearest vertex on optimized mesh to impact coordinate.
        
        Returns:
            distance and vertex index of nearest vertex on optimized mesh
        """
        vertices = mesh.vertices
        distances = np.linalg.norm((vertices - imp_coord, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        return min_dist, int(min_idx)
    
    def _create_impact_data(self, impact_coord: np.ndarray, 
                           obj1_idx: int, obj2_idx: int,
                           mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                           pos1: np.ndarray, pos2: np.ndarray,
                           vel1: np.ndarray, vel2: np.ndarray,
                           ang_vel1: np.ndarray, ang_vel2: np.ndarray,
                           mass1: float, mass2: float,
                           inertia1: np.ndarray, inertia2: np.ndarray,
                           com1: np.ndarray, com2: np.ndarray,
                           normal: np.ndarray, time: float,
                           collision_margin: float) -> ImpactData:
        """
        Create ImpactData object from collision information using proper physics.
        
        Returns:
            ImpactData object with all collision details
        """
        # Find nearest vertices on optimized meshes
        dist1, vertex_idx1 = self._nearest_optimized_vertex(mesh1, impact_coord)
        dist2, vertex_idx2 = self._nearest_optimized_vertex(mesh2, impact_coord)
        
        # Calculate impact forces using impulse-based collision response
        # Based on rigid body dynamics
        
        # 1. Calculate relative velocity at collision point
        # Position vectors from center of mass to collision point
        r1 = impact_coord - (pos1 + com1)
        r2 = impact_coord - (pos2 + com2)
        
        # Velocity at collision point = linear velocity + angular velocity × r
        v1 = vel1 + np.cross(ang_vel1, r1)
        v2 = vel2 + np.cross(ang_vel2, r2)
        
        # Relative velocity
        v_rel = v2 - v1
        
        # Velocity along collision normal
        v_rel_normal = np.dot(v_rel, normal)
        
        # Only process collisions with approaching objects
        if v_rel_normal >= 0:
            return None
        
        # 2. Calculate effective mass at collision point
        # This accounts for both linear and rotational inertia
        
        # Helper function to calculate effective mass
        def calculate_effective_mass(mass, inertia, r, normal):
            # Calculate inverse inertia tensor in world coordinates
            # For simplicity, we assume inertia tensor is diagonal in body coordinates
            # and use the current rotation
            
            # Calculate term: normal · (I^-1 · (r × normal)) × r
            r_cross_n = np.cross(r, normal)
            
            # For simplified calculation, use scalar approximation
            # More accurate would use full tensor math
            effective_mass_linear = 1.0 / mass
            
            # Approximate rotational contribution
            # Using average of diagonal elements of inertia tensor
            I_avg = np.trace(inertia) / 3.0
            if I_avg > 0:
                effective_mass_rotational = np.dot(r_cross_n, r_cross_n) / I_avg
            else:
                effective_mass_rotational = 0
            
            effective_mass = 1.0 / (effective_mass_linear + effective_mass_rotational)
            return effective_mass
        
        # Calculate effective masses
        m_eff1 = calculate_effective_mass(mass1, inertia1, r1, normal)
        m_eff2 = calculate_effective_mass(mass2, inertia2, r2, normal)
        
        # Combined effective mass
        m_eff = m_eff1 + m_eff2
        
        # 3. Calculate impulse magnitude
        # Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic)
        e = 0.7  # Typical value for hard objects
        
        # Impulse magnitude (negative because objects are approaching)
        J = -(1 + e) * v v_rel_normal * m_eff
        
        # 4. Calculate force vectors (impulse over small time)
        # Use audio sample period as time step
        config = self.impact_manager.get('config')
        dt = 1.0 / config.system.sample_rate
        
        # Force magnitude
        force_magnitude = abs(J) / dt if dt > 0 else abs(J) / 0.001
        
        # Force vectors in opposite directions along normal
        force1 = -force_magnitude * normal
        force2 = force_magnitude * normal
        
        # 5. Create ImpactData object
        impact_idx = len(self.impact_manager.get('impacts'))
        impact_data = ImpactData(
            idx=impact_idx,
            time=time,
            coord=tuple(impact_coord)
        )
        
        # Add collisions for both objects
        collision_obj1 = ObjCollision(
            obj_idx=obj1_idx,
            collision=Collision(
                id_vertex=int(vertex_idx1),
                force_vector=force1
            )
        )
        
        collision_obj2 = ObjCollision(
            obj_idx=obj2_idx,
            collision=Collision(
                id_vertex=int(vertex_idx2),
                force_vector=force2
            )
        )
        
        impact_data.add_collision(collision_obj1)
        impact_data.add_collision(collision_obj2)
        
        # Print collision details for debugging
        print(f"Collision detected at t={time:.3f}s:")
        print(f"  Objects: {obj1_idx} ↔ {obj2_idx}")
        print(f"  Coordinate: {impact_coord}")
        print(f"  Relative velocity: {v_rel_normal:.3f} m/s")
        print(f"  Force magnitude: {force_magnitude:.3f} N")
        print(f"  Vertex IDs: {vertex_idx1}, {vertex_idx2}")
        
        return impact_data
