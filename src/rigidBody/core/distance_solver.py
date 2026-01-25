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
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks

from ..core.entity_manager import EntityManager
from ..utils.config import Config, ObjectConfig
from ..lib.collision_data import CollisionData, CollisionType

from ..lib.functions import _load_pose, _load_mesh

@dataclass
class DistanceSolver:
    entity_manager: EntityManager

    def compute(self, objs_idx: Tuple[int, int]) -> List[Tuple[int, float]]:
        config = self.entity_manager.get('config')
        collision_margin = config.system.collision_margin
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
 
        trajectory, frames  = ([] for _ in range(2))
        config_objs = [config.objects[objs_idx[0]], config.objects[objs_idx[1]]]
        if config_objs[0].static and config_objs[1].static:
            # exit: objs_idx[0] and objs_idx[1] are static
            return
        elif not config_objs[0].static or not config_objs[1].static:
            trajectories = self.entity_manager.get('trajectories')
            for idx in trajectories.keys():
                if 'TrajectoryData' in str(type(trajectories[idx])):
                    if trajectories[idx].obj_idx == config_objs[0].idx or trajectories[idx].obj_idx == config_objs[1].idx:
                        trajectory.append(trajectories[idx])
                        frames.append(trajectories[idx].get_x())
        frames = np.unique(np.sort(np.concatenate((frames[0], frames[1]))))

        # assign trajectory
        trajectory1 = trajectory[0] if trajectory[0].obj_idx == config_objs[0].idx else trajectory[1]
        trajectory2 = trajectory[1] if trajectory[1].obj_idx == config_objs[1].idx else trajectory[0]

        distances = []
    
        for idx in range(len(frames)):
            distance, closest_points = self._distance(config_objs=config_objs, trajectory1=trajectory1, trajectory2=trajectory2, frame=frames[idx], sfps=sfps, sample_rate=sample_rate, collision_margin=collision_margin)
            distances.append(distance)
        distances = np.array(distances)

        # Analyze distances to detect collisions and contacts
        collision_events = self._analyze_distances(distances=distances, times=frames, config_objs=config_objs, collision_margin=collision_margin, sfps=sfps, sample_rate=sample_rate)

        for collision_data in collision_events:
            collision_idx = len(self.entity_manager.get('collisions')) + 1
            self.entity_manager.register('collisions', collision_data, collision_idx)

    def _analyze_distances(self, distances: np.ndarray, times: np.ndarray, config_objs: List[Any], collision_margin: float, sfps: float, sample_rate: int) -> List[CollisionData]:
        """
        Analyze distance data to detect impacts and continuous contacts.
        
        Parameters:
        -----------
        distances : np.ndarray
            Minimum distances between objects at each time sample
        times : np.ndarray
            Corresponding time samples
        config_objs : List
            Object configurations
        collision_margin : float
            System collision margin
        sfps : float
            Subframes per second
        
        Returns:
        --------
        List[CollisionData]
            Detected collision events
        """
        # Step 1: Compute adaptive threshold
        threshold = self._compute_adaptive_threshold(distances, collision_margin)
        
        print(f"Adaptive threshold for {config_objs[0].name} and {config_objs[1].name}: {threshold}")
        
        # Step 2: Identify contact regions (where distance <= threshold)
        contact_mask = distances <= threshold
        contact_regions = self._find_contact_regions(contact_mask, times)
        
        # Step 3: Classify contact regions as impacts or continuous contacts
        collision_events = []
        
        for region in contact_regions:
            region_times = times[region['start']:region['end']]
            region_distances = distances[region['start']:region['end']]
            
            # Classify based on duration and distance profile
            duration = region_times[-1] - region_times[0]
            avg_distance = np.mean(region_distances)
            min_distance = np.min(region_distances)
            
            # Check if this is an impact (short duration, sharp distance change)
            is_impact = self._is_impact_event(region_distances, region_times, duration, threshold, sfps)
            
            if is_impact:
                # For impacts, find the exact impact time (minimum distance)
                impact_idx = region['start'] + np.argmin(region_distances)
                impact_time = times[impact_idx]
                impact_distance = distances[impact_idx]
                
                print(f"Impact detected between {config_objs[0].name} and {config_objs[1].name} "
                      f"at frame {impact_time*sfps/sample_rate:.2f}, distance: {impact_distance:.6f}")
                
                # Create collision data for impact
                collision_data = CollisionData(type=CollisionType.IMPACT, obj1_idx=config_objs[0].idx, obj2_idx=config_objs[1].idx, frame=impact_time, avg_distance=avg_distance, distances=impact_distance)
                collision_events.append(collision_data)
            
            else:
                # Continuous contact - create collision data at the start
                contact_start_time = region_times[0]
                contact_range_time = region_times[-1] - region_times[0]
#                contact_range_time = len(region_times) * sample_rate / sfps
                contact_start_distance = region_distances[0]
                
                print(f"Continuous contact between {config_objs[0].name} and {config_objs[1].name} "
                      f"from frame {region_times[0]*sfps/sample_rate:.2f} to {region_times[-1]*sfps/sample_rate:.2f}, "
                      f"avg distance: {avg_distance:.6f}")
                
                # Create collision data for continuous contact (at start)
                collision_data = CollisionData(type=CollisionType.CONTACT, obj1_idx=config_objs[0].idx, obj2_idx=config_objs[1].idx, frame=contact_start_time, frame_range=contact_range_time, avg_distance=avg_distance, distances=region_distances)
                collision_events.append(collision_data)
        
        return collision_events
    
    def _compute_adaptive_threshold(self, distances: np.ndarray, collision_margin: float) -> float:
        """
        Compute adaptive threshold for contact detection.
        
        Uses statistical analysis of distance data to determine appropriate threshold.
        """
        # Remove outliers for better threshold calculation
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_distances = distances[(distances >= lower_bound) & (distances <= upper_bound)]
        
        if len(filtered_distances) == 0:
            filtered_distances = distances
        
        # Method 1: Use mean + small multiple of std for non-contact distances
        mean_dist = np.mean(filtered_distances)
        std_dist = np.std(filtered_distances)
        
        # Method 2: Use percentile-based approach
        percentile_10 = np.percentile(filtered_distances, 10)
        
        # Method 3: Find natural break in distance distribution
        # Use histogram to find valley between contact and non-contact distances
        hist, bin_edges = np.histogram(filtered_distances, bins=50)
        
        # Find local minima in histogram (potential threshold)
        minima_indices = find_peaks(-hist)[0]
        
        if len(minima_indices) > 0:
            # Use first significant minimum
            threshold_from_hist = bin_edges[minima_indices[0]]
        else:
            threshold_from_hist = percentile_10
        
        # Combine methods with weights
        threshold = (
            0.4 * (mean_dist - 0.5 * std_dist) +  # Statistical approach
            0.4 * percentile_10 +                  # Percentile approach
            0.2 * threshold_from_hist              # Histogram approach
        )
        
        # Ensure threshold is reasonable
        threshold = max(threshold, collision_margin * 0.1)  # At least 10% of collision margin
        threshold = min(threshold, collision_margin * 2.0)  # At most 200% of collision margin
        
        return threshold
    
    def _find_contact_regions(self, contact_mask: np.ndarray, times: np.ndarray) -> List[Dict]:
        """
        Find continuous regions where contact occurs.
        
        Returns list of dictionaries with start and end indices.
        """
        regions = []
        in_contact = False
        start_idx = 0
        
        for i, is_contact in enumerate(contact_mask):
            if is_contact and not in_contact:
                # Start of contact region
                in_contact = True
                start_idx = i
            elif not is_contact and in_contact:
                # End of contact region
                in_contact = False
                regions.append({
                    'start': start_idx,
                    'end': i,
                    'duration': times[i-1] - times[start_idx]
                })
        
        # Handle case where contact continues to end
        if in_contact:
            regions.append({
                'start': start_idx,
                'end': len(contact_mask),
                'duration': times[-1] - times[start_idx]
            })
        
        return regions
    
    def _is_impact_event(self, region_distances: np.ndarray, region_times: np.ndarray, duration: float, threshold: float, sfps: float) -> bool:
        """
        Determine if a contact region represents an impact event.
        
        Criteria for impact:
        1. Short duration (typically < 0.1 seconds)
        2. Sharp distance change (high gradient)
        3. Distance goes significantly below threshold and back up
        """
        # Criterion 1: Duration threshold
        max_impact_duration = 1 / sfps
        min_samples_for_contact = 3
        
        if duration > max_impact_duration:
            return False
        
        if len(region_distances) < min_samples_for_contact:
            return True  # Very brief contact is likely impact
        
        # Criterion 2: Distance gradient analysis
        gradients = np.gradient(region_distances, region_times)
        max_gradient = np.max(np.abs(gradients))
        
        # Criterion 3: Distance profile - impact should have V-shaped profile
        # Find minimum distance point
        min_idx = np.argmin(region_distances)
        
        # Check if distances decrease then increase (V-shape)
        if 0 < min_idx < len(region_distances) - 1:
            left_slope = (region_distances[min_idx] - region_distances[0]) / (region_times[min_idx] - region_times[0])
            right_slope = (region_distances[-1] - region_distances[min_idx]) / (region_times[-1] - region_times[min_idx])
            
            # For impact, left slope should be negative, right slope positive
            is_v_shaped = left_slope < -0.1 and right_slope > 0.1
            
            # Combined decision
            is_impact = (
                (duration < max_impact_duration) and
                (max_gradient > 1.0) and  # Rapid distance change
                is_v_shaped and
                (region_distances[min_idx] < threshold * 0.5)  # Goes well below threshold
            )
        else:
            is_impact = (duration < max_impact_duration) and (max_gradient > 1.0)
        
        return is_impact

    def _distance(self, config_objs: Tuple[Any, Any], trajectory1: Any, trajectory2: Any, frame: float, sfps: int, sample_rate: int, collision_margin: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate minimum distance between transformed meshes.
    
        Parameters:
        -----------
        trajectory1 : Any
            TrajectoryData object for first object
        trajectory2 : Any
            TrajectoryData object for second object
        frame : float
            Frame number (can be fractional for subframes)
    
        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            Minimum distance between transformed meshes and verification results
        """
        frame_idx = (sfps * frame / sample_rate) - 1
        if frame_idx.is_integer():
            frame_idx = int(frame_idx)
            vertices1, normals1, faces1 = _load_mesh(config_objs[0], frame_idx)
            vertices2, normals2, faces2 = _load_mesh(config_objs[1], frame_idx)
        else:
            vertices1 = trajectory1.get_vertices(frame)
            vertices2 = trajectory2.get_vertices(frame)
            normals1 = trajectory1.get_normals(frame)
            normals2 = trajectory2.get_normals(frame)
            faces1 = trajectory1.get_faces(frame)
            faces2 = trajectory2.get_faces(frame)

        mesh1 = trimesh.Trimesh(vertices=vertices1, vertex_normals=normals1, faces=faces1)
        mesh2 = trimesh.Trimesh(vertices=vertices2, vertex_normals=normals2, faces=faces2)

        # Calculate minimum distance between transformed meshes
        min_distance, closest_points = self._calculate_min_distance(mesh1=mesh1, mesh2=mesh2)
    
        return min_distance, closest_points

    def _calculate_min_distance(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, workers: int = -1) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Calculate minimum distance between two meshes using KDTree.
    
        Parameters:
        -----------
        mesh1 : trimesh.Trimesh
            First transformed mesh
        mesh2 : trimesh.Trimesh
            Second transformed mesh

        Returns:
        --------
        Tuple[float, Dict[str, np.ndarray]]
            Minimum distance and closest points information
        """
        method = 'approx'
        # Use KDTree for efficient distance calculation
        from scipy.spatial import cKDTree
    
        # Create KDTree for mesh2 vertices
        tree2 = cKDTree(mesh2.vertices)
    
        # Query distances from mesh1 vertices to mesh2
        distances, indices = tree2.query(mesh1.vertices, workers=workers)
    
        # Find minimum distance
        min_dist_idx = np.argmin(distances)
        min_distance = distances[min_dist_idx]

        # Get closest points
        closest_point1 = mesh1.vertices[min_dist_idx]
        closest_point2 = mesh2.vertices[indices[min_dist_idx]]
    
        # Create bounding boxes around approximate closest points
        search_radius = min_distance * 2.0  # Search in twice the approximate distance

        # Find vertices near the approximate closest points
        mask1 = np.linalg.norm(mesh1.vertices - closest_point1, axis=1) < search_radius
        mask2 = np.linalg.norm(mesh2.vertices - closest_point2, axis=1) < search_radius

        if np.any(mask1) and np.any(mask2):
            method = 'refine'
            # Build KD-tree for nearby vertices
            nearby_vertices2 = mesh2.vertices[mask2]
            tree2 = cKDTree(nearby_vertices2)
        
            # Query nearby vertices from mesh1
            distances, indices = tree2.query(mesh1.vertices[mask1], workers=workers)
        
            # Find minimal distance
            min_dist_idx = np.argmin(distances)
            min_distance = distances[min_dist_idx]
            closest_point1 = mesh1.vertices[mask1][min_dist_idx]
            closest_point2 = nearby_vertices2[indices[min_dist_idx]]
        
        closest_points = {
            'method': method,
            'mesh1_point': closest_point1,
            'mesh2_point': closest_point2,
            'mesh1_vertex_idx': min_dist_idx,
            'mesh2_vertex_idx': indices[min_dist_idx]
        }
    
        return min_distance, closest_points
