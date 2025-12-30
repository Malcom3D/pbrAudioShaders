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
from typing import List, Tuple
from dataclasses import dataclass, field
from itertools import groupby
from scipy.optimize import minimize, least_squares
from dask import delayed, compute

from ..core.entity_manager import EntityManager
from ..utils.config import Config, ObjectConfig
from ..lib.collision_data import tmpCollisionData

@dataclass
class CollisionSolver:
    entity_manager: EntityManager

    def compute(self, objs_idx: Tuple[int, int], detected_distances: List[Tuple[int, float]]):
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
        collision_margin = config.system.collision_margin

        # find idx of mins
        frames_idx, dists = np.hsplit(np.array(detected_distances), 2)
        t_frames = frames_idx / sfps
        idx_smallest = np.where(dists < collision_margin)[0]
        if idx_smallest.any():
            consec_idx = []
            for k, g in groupby(enumerate(idx_smallest), lambda x: x[0] - x[1]):
                group = list(g)
                consec_idx.append([idx for _, idx in group])

            # load objs pose
            array_length = 5
            for config_obj in config.objects:
                if config_obj.idx == objs_idx[0] or config_obj.idx == objs_idx[1]:
                    if not config_obj.static:
#                        positions, rotations, landmarks_vertices = self._load_pose(config_obj)
#                        landmarks_0, landmarks_1, landmarks_2 = np.hsplit(landmarks_vertices, 3)

                        # split array in before and after(flipped) collision frame_idx +/- 3 (or 4 if len(consec_idx[index]) == 2) and find collision_time, collision_point
                        print(f"{objs_idx} distances")
                        tasks = [self._solve_collision('distances', dists, consec_idx[index], t_frames, array_length) for index in range(len(consec_idx))]
                        results = compute(*tasks)
                        for name, x, intersection in results:
                            tmp_collision_data = tmpCollisionData(name=name, obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], frame=consec_idx[index], delta_time=x, value=intersection['point'], frames_idx=frames_idx, idx_smallest=idx_smallest, dists=dists)
                            collision_idx = len(self.entity_manager.get('collisions')) + 1
                            self.entity_manager.register('collisions', tmp_collision_data, collision_idx)

#                        print(f"{objs_idx} pos")
#                        pos = [self._solve_collision('positions', positions, consec_idx[index], t_frames, array_length) for index in range(len(consec_idx))]
#                        results = compute(*pos)
#                        for name, x, intersection in results:
#                            tmp_collision_data = tmpCollisionData(name='positions', obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], delta_time=x, value=intersection['point'])
#                            self.entity_manager.register('collisions', tmpCollisionData, collision_idx)

#                        print(f"{objs_idx} land_0")
#                        lands_0 = [self._solve_collision('landmarks_0', landmarks_0, consec_idx[index], t_frames, array_length) for index in range(len(consec_idx))]
#                        results = compute(*lands_0)
#                        for name, x, frame, intersection in results:
#                            tmp_collision_data = tmpCollisionData(name='landmarks_0', obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], frame=frame, delta_time=x, value=intersection['point'])
#                            self.entity_manager.register('collisions', tmpCollisionData, collision_idx)

#                        print(f"{objs_idx} land_1")
#                        lands_1 = [self._solve_collision('landmarks_1', landmarks_1, consec_idx[index], t_frames, array_length) for index in range(len(consec_idx))]
#                        results = compute(*lands_1)
#                        for name, x, frame, intersection in results:
#                            tmp_collision_data = tmpCollisionData(name='landmarks_1', obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], frame=frame, delta_time=x, value=intersection['point'])
#                            self.entity_manager.register('collisions', tmpCollisionData, collision_idx)

#                        print(f"{objs_idx} land_2")
#                        lands_2 = [self._solve_collision('landmarks_2', landmarks_2, consec_idx[index], t_frames, array_length) for index in range(len(consec_idx))]
#                        results = compute(*lands_2)
#                        for name, x, frame, intersection in results:
#                            tmp_collision_data = tmpCollisionData(name='landmarks_2', obj_idx1=objs_idx[0], obj_idx2=objs_idx[1], frame=frame, delta_time=x, value=intersection['point'])
#                            self.entity_manager.register('collisions', tmpCollisionData, collision_idx)

    def _load_pose(self, config_obj: ObjectConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose_path = config_obj.pose_path
        obj_name = config_obj.name

        npz_file = os.path.join(pose_path, f"{obj_name}.npz")

        if not npz_file:
            raise ValueError(f"No pose files found for {obj_name} in {pose_path}")

        pose = np.load(npz_file)
        positions = pose[pose.files[0]]
        rotations = pose[pose.files[1]]
        landmarks_vertices = pose[pose.files[2]]

        return positions, rotations, landmarks_vertices
        
    def _bernstein_poly(self, n, i, t):
        """Bernstein polynomial of degree n."""
        from scipy.special import comb
        return comb(n, i) * (t**i) * ((1 - t)**(n - i))
    
    def _bezier_curve(self, control_points, t):
        """Evaluate Bézier curve at parameter t."""
        n = len(control_points) - 1
        if n == 0:
            return control_points[0]
        
        result = np.zeros_like(control_points[0])
        for i in range(n + 1):
            result += self._bernstein_poly(n, i, t) * control_points[i]
        return result
    
    def _fit_bezier_curve(self, trajectory: np.ndarray, t_norm: np.ndarray, max_degree: int = 3):
        """Fit Bézier curve to trajectory data and determine optimal degree."""
        n_samples = len(trajectory)
        best_result = None
        best_results = []
        best_score = float('inf')
        
        for degree in range(1, max_degree + 1):
            # Number of control points = degree + 1
            n_control = degree + 1
            
            # Initial guess for control points
            init_control = np.zeros((n_control, trajectory.shape[1]))
            
            def objective(params):
                control = params.reshape((n_control, trajectory.shape[1]))
                error = 0
                for j, t in enumerate(t_norm):
                    predicted = self._bezier_curve(control, t)
                    error += np.sum((predicted - trajectory[j])**2)
                return error
            
#            # Flatten for optimization
#            init_params = init_control.flatten()
            for i in range(n_control):
                idx = int((i / degree) * (n_samples - 1))
                init_control[i] = trajectory[idx]
            
                init_params = init_control.flatten()
                
                # Optimize control points
                result = minimize(objective, init_params, method='L-BFGS-B')
            
                if result.fun < best_score:
                    best_score = result.fun
                    best_results.append({'degree': degree, 'control_points': result.x.reshape((n_control, trajectory.shape[1])), 'error': result.fun})

        key = 'error'
        min_value = min(d[key] for d in best_results if key in d)
        index = [i for i, d in enumerate(best_results) if d[key] == min_value]
        print(index[0], best_results[index[0]])
        return best_results[index[0]]
    
    def _find_intersection(self, curve1, curve2, degree1, degree2):
        """Find intersection point between two Bézier curves."""
        def intersection_error(params):
            t1, t2 = params
            point1 = self._bezier_curve(curve1, t1)
            point2 = self._bezier_curve(curve2, t2)
            return np.sum((point1 - point2)**2)
        
#        # Initial guesses (midpoint of each curve)
#        init_guess = [0.5, 0.5]
        
        # Constrain t from [0, 1] to [0, 3]
        bounds = [(0,  3), (0, 3)]
        
        for t1_start in np.linspace(0, 5, 10):
           for t2_start in np.linspace(0, 5, 10):
                # Initial guesses
                init_guess = [t1_start, t2_start]
                result = minimize(intersection_error, init_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.fun < 2e-6:  # Tolerance for intersection
            t1_opt, t2_opt = result.x
            intersection_point = self._bezier_curve(curve1, t1_opt)
            return {
                't1': t1_opt,
                't2': t2_opt,
                'point': intersection_point,
                'error': result.fun
            }
        return None

    @delayed
    def _solve_collision(self, name: str, array: np.ndarray, consec_idx: List[int], t_frames: np.ndarray, array_length: int = 5):
        """split array in before and after(flipped) collision frame_idx +/- 3 (or 4 if len(consec_idx[index]) == 2)"""
        array_length = -abs(array_length)
        collision_frame_idx = consec_idx[0]
        if len(consec_idx) == 1:
            before_collision = array[:collision_frame_idx - 1][array_length:]
            after_collision = np.flip(array[collision_frame_idx + 2:])[array_length:]
            # split t_norm as before_collision array
            t_frame_before = t_frames[:collision_frame_idx - 1][array_length:]
        elif len(consec_idx) >= 2:
            before_collision = array[:collision_frame_idx - 2][array_length:]
            after_collision = np.flip(array[collision_frame_idx + 2:])[array_length:]
            # split t_frames as before_collision array
            t_frame_before = t_frames[:collision_frame_idx - 2][array_length:]

        """Main solving routine."""
        t_norm = (t_frame_before - t_frame_before[0]) / (t_frame_before[-1] - t_frame_before[0])
        dist1_result = self._fit_bezier_curve(before_collision, t_norm)
        dist2_result = self._fit_bezier_curve(after_collision, t_norm)
        
        # Find intersections
        intersection = self._find_intersection(dist1_result['control_points'], dist2_result['control_points'], dist1_result['degree'], dist2_result['degree'])
        
        # Calculate intersection time x
#        x = intersection['t1'] * (self.kT - self.T0)  # Convert to actual time
        print(f"intersection: {intersection} dist1: {dist1_result} dist2: {dist2_result}")
        x = intersection['t1'] * (t_frame_before[-1] - t_frame_before[0])  # Convert to actual time
        return name, x, intersection
