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
import trimesh
import numpy as np
from numba import jit, prange
from scipy.spatial import cKDTree
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from itertools import groupby

from ..core.entity_manager import EntityManager
from ..lib.collision_data import CollisionData
from ..lib.functions import _parse_lib
from ..lib.modal_vertices import ModalVertices
from ..lib.score_data import ScoreEvent, ScoreTrack


@dataclass
class CollisionSolver:
    entity_manager: EntityManager

    def compute(self, collision: CollisionData) -> None:
        config = self.entity_manager.get('config')
        sample_counter = self.entity_manager.get('sample_counter')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        forces = self.entity_manager.get('forces')

        obj1_idx = collision.obj1_idx
        obj2_idx = collision.obj2_idx
        for conf_obj in config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
            if conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if trajectories[t_idx].obj_idx == obj1_idx:
                self.trajectory1 = trajectories[t_idx]
            if trajectories[t_idx].obj_idx == obj2_idx:
                self.trajectory2 = trajectories[t_idx]

        mesh1_faces = self.trajectory1.get_faces()
        mesh2_faces = self.trajectory2.get_faces()

        if isinstance(config_obj2.connected, np.ndarray) and config_obj1.idx in config_obj2.connected[:,0] and isinstance(config_obj1.connected, np.ndarray) and config_obj2.idx in config_obj1.connected[:,0]:
            self._connected_facing_face(obj1_idx=obj1_idx, obj2_idx=obj2_idx, mesh1_faces=mesh1_faces, mesh2_faces=mesh2_faces)
        if not config_obj1.static or not config_obj2.static:
            for f_idx in forces.keys():
                if forces[f_idx].obj_idx == obj1_idx and obj2_idx in forces[f_idx].other_obj_idx or forces[f_idx].obj_idx == obj2_idx and obj1_idx in forces[f_idx].other_obj_idx:
                    force = forces[f_idx]
        else:
            return

        total_samples = int(self.trajectory1.get_x()[-1] if not config_obj1.static else self.trajectory2.get_x()[-1])
        sample_counter.total_samples = total_samples

        start_samples = int(collision.frame - collision.impulse_range / 2)
        stop_samples = int(collision.frame + collision.impulse_range)
        impact_end = stop_samples

        if collision.type.value == 'contact':
            stop_samples = int(collision.frame + collision.frame_range + collision.impulse_range)
        if not stop_samples <= total_samples:
            stop_samples = total_samples

        # Load pre-computed distance data
        distances_dir = f"{config.system.cache_path}/distances"
        distance_file = f"{distances_dir}/{obj1_idx}_{obj2_idx}.npz"
        
        if os.path.exists(distance_file):
            # Use pre-computed distance data for faster collision detection
            distance_data = np.load(distance_file)
            distances = distance_data[distance_data.files[0]]
            closest_points1 = distance_data[distance_data.files[1]]
            closest_points2 = distance_data[distance_data.files[2]]

            # Get the frames corresponding to our sample range
            frames = self.trajectory1.get_x() if not config_obj1.static else self.trajectory2.get_x()

            distances = CubicSpline(frames, distances, extrapolate=1)
            closest_points1 = [CubicSpline(frames, closest_points1[:, i], extrapolate=1) for i in range(closest_points1.shape[1])]
            closest_points2 = [CubicSpline(frames, closest_points2[:, i], extrapolate=1) for i in range(closest_points2.shape[1])]

            score_tracks = self.entity_manager.get('score_tracks')
            score_track1, score_track2 = (None for _ in range(2))
            for st_idx in score_tracks.keys():
                if score_tracks[st_idx].obj_idx == obj1_idx:
                   score_track1 = score_tracks[st_idx]
                elif score_tracks[st_idx].obj_idx == obj2_idx:
                   score_track2 = score_tracks[st_idx]

            if score_track1 == None:
                score_track1 = ScoreTrack(obj_idx=obj1_idx, obj_name=config_obj1)
                score_track_idx = len(self.entity_manager.get('score_tracks')) + 1
                self.entity_manager.register('score_tracks', score_track1, score_track_idx)
                score_track1 = self.entity_manager.get('score_tracks', score_track_idx)
            if score_track2 == None:
                score_track2 = ScoreTrack(obj_idx=obj2_idx, obj_name=config_obj1)
                score_track_idx = len(self.entity_manager.get('score_tracks')) + 1
                self.entity_manager.register('score_tracks', score_track2, score_track_idx)
                score_track2 = self.entity_manager.get('score_tracks', score_track_idx)

            samples_idx, vertex1_id_list, vertex2_id_list = ([] for _ in range(3))
            for sample_idx in range(start_samples, stop_samples):
                samples_idx.append(sample_idx)
                collision_margin = distances(sample_idx) * (1 + collision.threshold)

                # Get pre-computed closest points for this frame
                cp1 = np.array([closest_points1[0](sample_idx), closest_points1[1](sample_idx), closest_points1[2](sample_idx)])
                cp2 = np.array([closest_points2[0](sample_idx), closest_points2[1](sample_idx), closest_points2[2](sample_idx)])
                    
                # Find vertices near the collision using KDTree
                mesh1_vertices = self.trajectory1.get_vertices(sample_idx)
                mesh2_vertices = self.trajectory2.get_vertices(sample_idx)
                    
                # Build KDTree for each mesh
                tree1 = cKDTree(mesh1_vertices)
                tree2 = cKDTree(mesh2_vertices)

                # Get contact type
                force_frames = force.frames
                ctf = force_frames[np.where(force_frames <= sample_idx)]
                if not ctf.shape[0] == 0 and not ctf[-1] == force_frames[-1]:
                    contact_type = force.get_contact_type(ctf[-1])
                elif not ctf.shape[0] == 0 and ctf[-1] == force_frames[-1]:
                    contact_type = force.get_contact_type(force_frames[-2])
                else:
                    contact_type = force.get_contact_type(force_frames[0])

                # Find vertices within collision margin of the closest points
                radius = collision_margin * 2.0  # Use slightly larger radius to capture nearby faces
                if contact_type == 4:
                    radius = collision_margin * 4.0  # Use slightly larger radius to capture nearby rolling faces

                # Query for vertices near the closest closest point on each mesh
                vertices1_idx = tree1.query_ball_point(cp1, radius)
                vertices2_idx = tree2.query_ball_point(cp2, radius)

                if vertices1_idx and vertices2_idx:
                    # Convert vertex indices to face indices
                    vertices1_idx = np.array(vertices1_idx)
                    vertices2_idx = np.array(vertices2_idx)
                        
                    # Find faces that contain these vertices
                    mesh1_faces_idx = np.where(np.any(np.isin(mesh1_faces, vertices1_idx), axis=1))[0]
                    mesh2_faces_idx = np.where(np.any(np.isin(mesh2_faces, vertices2_idx), axis=1))[0]

                    # Get vertex coordinates for each triangle
                    v10 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 0]]
                    v11 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 1]]
                    v12 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 2]]

                    v20 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 0]]
                    v21 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 1]]
                    v22 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 2]]

                    # Compute edge vectors
                    e11 = v11 - v10
                    e12 = v12 - v10
                    e21 = v21 - v20
                    e22 = v22 - v20
    
                    # Compute cross product (area vector)
                    cross1 = np.cross(e11, e12)
                    cross2 = np.cross(e21, e22)
    
                    # Compute normalized area
                    total_area1 = np.sum(trimesh.Trimesh(vertices=mesh1_vertices, faces=mesh1_faces).area_faces)
                    total_area2 = np.sum(trimesh.Trimesh(vertices=mesh2_vertices, faces=mesh2_faces).area_faces)
                    face_area1 = np.sum(0.5 * np.sqrt(np.sum(cross1**2, axis=1))) / total_area1
                    face_area2 = np.sum(0.5 * np.sqrt(np.sum(cross2**2, axis=1))) / total_area2

                    # Get unique vertex indices from these faces
                    cvidx1 = np.unique(mesh1_faces[mesh1_faces_idx].flatten())
                    cvidx2 = np.unique(mesh2_faces[mesh2_faces_idx].flatten())
                        
                    vertex1_id_list += cvidx1.tolist()
                    vertex2_id_list += cvidx2.tolist()

                    print(f"facing faces between {config_obj1.name} and {config_obj2.name} at frame {sample_idx}: {mesh1_faces_idx.shape[0]} {mesh2_faces_idx.shape[0]} at distance {collision_margin}")
                    if sample_idx <= impact_end:
                        impact_type = np.array([1])
                        score_track1.add_event(ScoreEvent(type=impact_type, sample_idx=sample_idx, contact_area=face_area1, vertex_ids=cvidx1))
                        score_track2.add_event(ScoreEvent(type=impact_type, sample_idx=sample_idx, contact_area=face_area2, vertex_ids=cvidx2))
                    score_track1.add_event(ScoreEvent(type=contact_type, sample_idx=sample_idx, contact_area=face_area1, vertex_ids=cvidx1))
                    score_track2.add_event(ScoreEvent(type=contact_type, sample_idx=sample_idx, contact_area=face_area2, vertex_ids=cvidx2))
                else:
                    print(f"facing faces between {config_obj1.name} and {config_obj2.name} at frame {sample_idx}: 0 0 at distance {collision_margin}")
                    score_track1.add_event(ScoreEvent(type=contact_type, sample_idx=sample_idx, contact_area=0, vertex_ids=np.array([])))
                    score_track2.add_event(ScoreEvent(type=contact_type, sample_idx=sample_idx, contact_area=0, vertex_ids=np.array([])))

            vertex1_id_list = np.unique(np.array(vertex1_id_list))
            vertex2_id_list = np.unique(np.array(vertex2_id_list))

            collision.samples = np.array(samples_idx)
            modal_vertices = self.entity_manager.get('modal_vertices')
            mod_v1, mod_v2 = (None for _ in range(2))
            for mv_idx in modal_vertices.keys():
                if modal_vertices[mv_idx].obj_idx == obj1_idx:
                    mod_v1 = modal_vertices[mv_idx]
                elif modal_vertices[mv_idx].obj_idx == obj2_idx:
                    mod_v2 = modal_vertices[mv_idx]
            
            if not mod_v1 == None:
                mod_v1.add_vertices(vertex1_id_list)
            else:
                modal_vertices1 = ModalVertices(obj_idx=obj1_idx, vertices=vertex1_id_list)
                modal_idx = len(self.entity_manager.get('modal_vertices')) + 1
                self.entity_manager.register('modal_vertices', modal_vertices1, modal_idx)

            if not mod_v2 == None:
                mod_v2.add_vertices(vertex2_id_list)
            else:
                modal_vertices2 = ModalVertices(obj_idx=obj2_idx, vertices=vertex2_id_list)
                modal_idx = len(self.entity_manager.get('modal_vertices')) + 1
                self.entity_manager.register('modal_vertices', modal_vertices2, modal_idx)

    def _connceted_facing_face(self, obj1_idx: int, obj2_idx: int, mesh1_faces: np.ndarray, mesh2_faces: np.ndarray) -> None:
        # Load pre-computed distance data
        config = self.entity_manager.get('config')
        distances_dir = f"{config.system.cache_path}/distances"
        distance_file = f"{distances_dir}/connected_{obj1_idx}_{obj2_idx}.npz"

        if os.path.exists(distance_file):
            # Use pre-computed distance data for faster collision detection
            distance_data = np.load(distance_file)
            collision_margin = distance_data[distance_data.files[0]]
            cp1 = distance_data[distance_data.files[1]]
            cp2 = distance_data[distance_data.files[2]]

            vertex1_id_list, vertex2_id_list = ([] for _ in range(2))
            # Find vertices near the collision using KDTree
            mesh1_vertices = self.trajectory1.get_vertices(0)
            mesh2_vertices = self.trajectory2.get_vertices(0)

            # Build KDTree for each mesh
            tree1 = cKDTree(mesh1_vertices)
            tree2 = cKDTree(mesh2_vertices)
                    
            # Find vertices within collision margin of the closest points
            radius = collision_margin * 2.0  # Use slightly larger radius to capture nearby faces

            # Query for vertices near the closest closest point on each mesh
            vertices1_idx = tree1.query_ball_point(cp1, radius)
            vertices2_idx = tree2.query_ball_point(cp2, radius)

            if vertices1_idx and vertices2_idx:
                # Convert vertex indices to face indices
                vertices1_idx = np.array(vertices1_idx)
                vertices2_idx = np.array(vertices2_idx)
                         
                # Find faces that contain these vertices
                mesh1_faces_idx = np.where(np.any(np.isin(mesh1_faces, vertices1_idx), axis=1))[0]
                mesh2_faces_idx = np.where(np.any(np.isin(mesh2_faces, vertices2_idx), axis=1))[0]

                # Get vertex coordinates for each triangle
                v10 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 0]]
                v11 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 1]]
                v12 = mesh1_vertices[mesh1_faces[mesh1_faces_idx][:, 2]]

                v20 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 0]]
                v21 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 1]]
                v22 = mesh2_vertices[mesh2_faces[mesh2_faces_idx][:, 2]]

                # Compute edge vectors
                e11 = v11 - v10
                e12 = v12 - v10
                e21 = v21 - v20
                e22 = v22 - v20

                # Compute cross product (area vector)
                cross1 = np.cross(e11, e12)
                cross2 = np.cross(e21, e22)

                # Compute normalized area
                total_area1 = np.sum(trimesh.Trimesh(vertices=mesh1_vertices, faces=mesh1_faces).area_faces)
                total_area2 = np.sum(trimesh.Trimesh(vertices=mesh2_vertices, faces=mesh2_faces).area_faces)
                face_area1 = np.sum(0.5 * np.sqrt(np.sum(cross1**2, axis=1))) / total_area1
                face_area2 = np.sum(0.5 * np.sqrt(np.sum(cross2**2, axis=1))) / total_area2

                # Get unique vertex indices from these faces
                cvidx1 = np.unique(mesh1_faces[mesh1_faces_idx].flatten())
                cvidx2 = np.unique(mesh2_faces[mesh2_faces_idx].flatten())
                        
                vertex1_id_list += cvidx1.tolist()
                vertex2_id_list += cvidx2.tolist()

                vertex1_id_list = np.unique(np.array(vertex1_id_list))
                vertex2_id_list = np.unique(np.array(vertex2_id_list))

                modal_vertices = self.entity_manager.get('modal_vertices')
                mod_v1, mod_v2 = (None for _ in range(2))
                for mv_idx in modal_vertices.keys():
                    if modal_vertices[mv_idx].obj_idx == obj1_idx:
                        mod_v1 = modal_vertices[mv_idx]
                    elif modal_vertices[mv_idx].obj_idx == obj2_idx:
                        mod_v2 = modal_vertices[mv_idx]
                        
                if not mod_v1 == None:
                    mod_v1.add_vertices(vertex1_id_list)
                else:
                    modal_vertices1 = ModalVertices(obj_idx=obj1_idx, vertices=vertex1_id_list, connected_area=face_area1/vertex1_id_list.shape[0])
                    modal_idx = len(self.entity_manager.get('modal_vertices')) + 1
                    self.entity_manager.register('modal_vertices', modal_vertices1, modal_idx)
                   
                if not mod_v2 == None:
                    mod_v2.add_vertices(vertex2_id_list)
                else:
                    modal_vertices2 = ModalVertices(obj_idx=obj2_idx, vertices=vertex2_id_list, connected_area=face_area2/vertex2_id_list.shape[0])
                    modal_idx = len(self.entity_manager.get('modal_vertices')) + 1
                    self.entity_manager.register('modal_vertices', modal_vertices2, modal_idx)
