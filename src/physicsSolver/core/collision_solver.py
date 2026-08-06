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
import blosc2
import trimesh
import numpy as np
from numba import jit, prange
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from itertools import groupby

from pbrAudioCommon import EntityManager
from pbrAudioCommon import ScoreEvent, ScoreTrack
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from ellipsoidalProxy import ProxyPhysics

from ..lib.collision_data import CollisionData
from ..lib.modal_vertices import ModalVertices
from ..lib.force_data import ContactType

@dataclass
class CollisionSolver:
    entity_manager: EntityManager
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')

    def compute(self, collision: CollisionData) -> None:
        """Optimized collision solver with proxy mesh special handling."""
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = (fps / fps_base) * subframes
        
        forces = self.entity_manager.get('forces')
        
        # Determine object indices and configurations
        obj1_idx, obj2_idx = self._get_object_indices(collision, forces)
        config_obj1, config_obj2 = self._get_object_configs(obj1_idx, obj2_idx)
        trajectory1, trajectory2 = self._get_trajectories(obj1_idx, obj2_idx)
        
        # Check if either object is a proxy mesh
        is_proxy1 = config_obj1.proxy_type is not False
        is_proxy2 = config_obj2.proxy_type is not False
        if is_proxy1 or is_proxy2:
            self.proxy_physics = ProxyPhysics(self.entity_manager)
        
        # Handle connected objects
        if self._is_connected(config_obj1, config_obj2):
            self._connected_facing_face(obj1_idx, obj2_idx, trajectory1, trajectory2)
        
        total_samples = int(trajectory1.get_x()[-1] if not config_obj1.static else trajectory2.get_x()[-1])
        
        # Calculate sample range
        start_samples, stop_samples, impact_end = self._calculate_sample_range(collision, total_samples, sample_rate, sfps, config_obj1, config_obj2)
        
        # Load distance data
        distances_data = self._load_distance_data(collision)
        if distances_data is None:
            return
        
        distances, closest_points1, closest_points2 = distances_data
        frames = np.unique(np.sort(np.concatenate((trajectory1.get_x(), trajectory2.get_x()))))
        
        # Create spline interpolators
        distances_spline = CubicSpline(frames, distances, extrapolate=1)
        cp1_splines = [CubicSpline(frames, closest_points1[:, i], extrapolate=1) for i in range(3)]
        cp2_splines = [CubicSpline(frames, closest_points2[:, i], extrapolate=1) for i in range(3)]
        
        # Initialize score tracks
        score_track1, score_track2 = self._init_score_tracks(obj1_idx, obj2_idx, config_obj1, config_obj2)
        
        # Get mesh faces
        mesh1_faces = trajectory1.get_faces()
        mesh2_faces = trajectory2.get_faces()
        
        # configure blosc2 compression BLOSCLZ LZ4 
        cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, typesize=1, clevel=1, nthreads=8)
        dparams = blosc2.DParams(nthreads=16)

        # Initialize score data arrays
        mesh1_verts = trajectory1.get_vertices(0)
        score_type1 = np.zeros((total_samples, 1), dtype=np.int32)
        score_vertex_ids1 = np.full((total_samples, mesh1_verts.shape[0]), np.bool_(False), dtype=np.bool_)
        score_vertex_ids1 = blosc2.asarray(score_vertex_ids1, cparams=cparams, dparams=dparams)
        score_contact_area1 = np.zeros((total_samples, 1), dtype=np.float32)
        
        mesh2_verts = trajectory2.get_vertices(0)
        score_type2 = np.zeros((total_samples, 1), dtype=np.int32)
        score_vertex_ids2 = np.full((total_samples, mesh2_verts.shape[0]), np.bool_(False), dtype=np.bool_)
        score_vertex_ids2 = blosc2.asarray(score_vertex_ids2, cparams=cparams, dparams=dparams)
        score_contact_area2 = np.zeros((total_samples, 1), dtype=np.float32)
        
        vertex1_id_list = []
        vertex2_id_list = []
        
        # Process samples
        for sample_idx in range(start_samples, stop_samples):
            collision_margin = distances_spline(sample_idx) * (1 + collision.threshold)
            
            cp1 = np.array([cp1_splines[0](sample_idx), cp1_splines[1](sample_idx), cp1_splines[2](sample_idx)])
            cp2 = np.array([cp2_splines[0](sample_idx), cp2_splines[1](sample_idx), cp2_splines[2](sample_idx)])
            
            # Get contact type
            contact_type = self._get_contact_type(forces, obj1_idx, obj2_idx, sample_idx)
            
            # Use optimized proxy mesh collision detection
            if is_proxy1 or is_proxy2:
                vertices1_idx, vertices2_idx, face_area1, face_area2 = self.proxy_physics.optimized_proxy_collision(obj1_idx, obj2_idx, cp1, cp2, collision_margin, contact_type, trajectory1, trajectory2, mesh1_faces, mesh2_faces, sample_idx)
            else:
                # Original method for non-proxy meshes
                vertices1_idx, vertices2_idx, face_area1, face_area2 = self._standard_collision(cp1, cp2, collision_margin, contact_type, trajectory1, trajectory2, mesh1_faces, mesh2_faces, sample_idx)
            
            # Update score data
            if vertices1_idx is not None and vertices2_idx is not None:
                self._update_score_data(sample_idx, impact_end, contact_type, vertices1_idx, vertices2_idx, face_area1, face_area2, score_type1, score_type2, score_vertex_ids1, score_vertex_ids2, score_contact_area1, score_contact_area2, vertex1_id_list, vertex2_id_list, config_obj1, config_obj2)

                debug_print(f"facing faces between {config_obj1.name} and {config_obj2.name} at frame {sample_idx}: {np.count_nonzero(vertices1_idx)} {np.count_nonzero(vertices2_idx)} at distance {collision_margin} for {ContactType(contact_type).name.lower()}")

        # Finalize score tracks
        self._finalize_score_tracks(score_track1, score_track2, config_obj1, config_obj2, start_samples, stop_samples, score_type1, score_type2, score_vertex_ids1, score_vertex_ids2, score_contact_area1, score_contact_area2)
        
        # Update modal vertices
        self._update_modal_vertices(obj1_idx, obj2_idx, vertex1_id_list, vertex2_id_list, trajectory1, trajectory2, mesh1_faces, mesh2_faces)

    def _standard_collision(self, cp1, cp2, collision_margin, contact_type, trajectory1, trajectory2, mesh1_faces, mesh2_faces, sample_idx):
        """Original collision detection for non-proxy meshes."""
        mesh1_vertices = trajectory1.get_vertices(sample_idx)
        mesh2_vertices = trajectory2.get_vertices(sample_idx)
        
        tree1 = cKDTree(mesh1_vertices)
        tree2 = cKDTree(mesh2_vertices)
        
        radius = collision_margin * 2.0
        if contact_type in [4, 5]:
            radius = collision_margin * 4.0
        
        vertices1_idx = np.array(tree1.query_ball_point(cp1, radius, workers=-1))
        vertices2_idx = np.array(tree2.query_ball_point(cp2, radius, workers=-1))
        
        if len(vertices1_idx) > 0 and len(vertices2_idx) > 0:
            mesh1_faces_idx = np.where(np.any(np.isin(mesh1_faces, vertices1_idx), axis=1))[0]
            mesh2_faces_idx = np.where(np.any(np.isin(mesh2_faces, vertices2_idx), axis=1))[0]
            
            face_area1 = len(mesh1_faces_idx) / len(mesh1_faces)
            face_area2 = len(mesh2_faces_idx) / len(mesh2_faces)
        else:
            face_area1 = 0
            face_area2 = 0
            mesh1_faces_idx = np.array([])
            mesh2_faces_idx = np.array([])

        return vertices1_idx, vertices2_idx, face_area1, face_area2

    def _get_object_indices(self, collision, forces):
        """Determine primary and secondary object indices."""
        for f_idx in forces.keys():
            if forces[f_idx].obj_idx == collision.obj1_idx and forces[f_idx].other_obj_idx == collision.obj2_idx:
                return collision.obj1_idx, collision.obj2_idx
            elif forces[f_idx].obj_idx == collision.obj2_idx and forces[f_idx].other_obj_idx == collision.obj1_idx:
                return collision.obj2_idx, collision.obj1_idx
        return collision.obj1_idx, collision.obj2_idx

    def _get_object_configs(self, obj1_idx, obj2_idx):
        """Get object configurations."""
        config_obj1 = config_obj2 = None
        for conf_obj in self.config.objects:
            if conf_obj.idx == obj1_idx:
                config_obj1 = conf_obj
            if conf_obj.idx == obj2_idx:
                config_obj2 = conf_obj
        return config_obj1, config_obj2

    def _get_trajectories(self, obj1_idx, obj2_idx):
        """Get trajectory objects."""
        trajectory1 = trajectory2 = None
        trajectories = self.entity_manager.get('trajectories')
        for t_idx in trajectories.keys():
            if trajectories[t_idx].obj_idx == obj1_idx:
                trajectory1 = trajectories[t_idx]
            if trajectories[t_idx].obj_idx == obj2_idx:
                trajectory2 = trajectories[t_idx]
        return trajectory1, trajectory2

    def _is_connected(self, config_obj1, config_obj2):
        """Check if objects are connected."""
        if config_obj1 is None or config_obj2 is None:
            return False
        return (isinstance(config_obj2.connected, np.ndarray) and 
                config_obj1.idx in config_obj2.connected[:, 0] and
                isinstance(config_obj1.connected, np.ndarray) and 
                config_obj2.idx in config_obj1.connected[:, 0])

    def _calculate_sample_range(self, collision, total_samples, sample_rate, sfps, config_obj1, config_obj2):
        """Calculate the sample range for collision processing."""
        start_samples = int(collision.frame - collision.impulse_range / 2)
        start_samples = max(0, start_samples)
        stop_samples = int(collision.frame + collision.impulse_range)
        impact_end = stop_samples
        
        if collision.type.value == 'contact':
            stop_samples = int(collision.frame + collision.frame_range + collision.impulse_range)
        stop_samples = min(stop_samples, total_samples)
        
        # Handle fracture and shard frames
        start_samples, stop_samples = self._adjust_for_fracture_shard(stop_samples, start_samples, sample_rate, sfps, config_obj1, config_obj2)
        
        return start_samples, stop_samples, impact_end

    def _adjust_for_fracture_shard(self, stop_samples, start_samples, sample_rate, sfps, config_obj1, config_obj2):
        """Adjust sample range for fracture and shard events."""
        fracture_frame1 = -1
        if not config_obj1.fractured == False:
            if stop_samples >= config_obj1.fractured >= start_samples:
                fracture_frame1 = config_obj1.fractured - 1
                fracture_frame1 *= sample_rate / sfps

        fracture_frame2 = -1
        if not config_obj2.fractured == False:
            if stop_samples >= config_obj2.fractured >= start_samples:
                fracture_frame2 = config_obj2.fractured - 1
                fracture_frame2 *= sample_rate / sfps

        is_shard_frame1 = -1
        if not config_obj1.is_shard == False:
            if stop_samples >= config_obj1.is_shard >= start_samples:
                is_shard_frame1 = config_obj1.is_shard
                is_shard_frame1 *= sample_rate / sfps

        is_shard_frame2 = -1
        if not config_obj2.is_shard == False:
            if stop_samples >= config_obj2.is_shard  >= start_samples:
                is_shard_frame2 = config_obj2.is_shard
                is_shard_frame2 *= sample_rate / sfps

        fracture_samples = min(fracture_frame1, fracture_frame2)
        stop_samples = min(stop_samples, fracture_samples) if not fracture_samples == -1 else stop_samples

        shard_samples = max(is_shard_frame1, is_shard_frame2)
        start_samples = max(start_samples, shard_samples) if not shard_samples == -1 else start_samples

        return start_samples, stop_samples

    def _load_distance_data(self, collision):
        """Load pre-computed distance data."""
        distances_dir = f"{self.config.system.cache_path}/distances"
        distance_file = f"{distances_dir}/{collision.obj1_idx}_{collision.obj2_idx}.npz"
        
        if os.path.exists(distance_file):
            distance_data = np.load(distance_file)
            return (
                distance_data[distance_data.files[0]],
                distance_data[distance_data.files[1]],
                distance_data[distance_data.files[2]]
            )
        return None

    def _init_score_tracks(self, obj1_idx, obj2_idx, config_obj1, config_obj2):
        """Initialize or retrieve score tracks."""
        score_tracks = self.entity_manager.get('score_tracks')
        score_track1 = score_track2 = None
        
        for st_idx in score_tracks.keys():
            if score_tracks[st_idx].obj_idx == obj1_idx:
                score_track1 = score_tracks[st_idx]
            elif score_tracks[st_idx].obj_idx == obj2_idx:
                score_track2 = score_tracks[st_idx]
        
        if score_track1 is None:
            score_track1 = ScoreTrack(obj_idx=obj1_idx, obj_name=config_obj1.name)
            self.entity_manager.register('score_tracks', score_track1)
        
        if score_track2 is None:
            score_track2 = ScoreTrack(obj_idx=obj2_idx, obj_name=config_obj2.name)
            self.entity_manager.register('score_tracks', score_track2)
        
        return score_track1, score_track2

    def _get_contact_type(self, forces, obj1_idx, obj2_idx, sample_idx):
        """Get contact type for a given sample."""
        for f_idx in forces.keys():
            if forces[f_idx].obj_idx == obj1_idx and forces[f_idx].other_obj_idx == obj2_idx:
                force = forces[f_idx]
                force_frames = force.frames
                ctf = force_frames[np.where(force_frames <= sample_idx)]
                
                if ctf.shape[0] > 0:
                    if ctf[-1] != force_frames[-1]:
                        return force.get_contact_type(ctf[-1])
                    else:
                        return force.get_contact_type(force_frames[-2])
                else:
                    return force.get_contact_type(force_frames[0])
        return 0  # Default to no contact

    def _update_score_data(self, sample_idx, impact_end, contact_type, vertices1_idx, vertices2_idx, face_area1, face_area2, score_type1, score_type2, score_vertex_ids1, score_vertex_ids2, score_contact_area1, score_contact_area2, vertex1_id_list, vertex2_id_list, config_obj1, config_obj2):
        """Update score data arrays for a sample."""
        if len(vertices1_idx) > 0 and len(vertices2_idx) > 0:
            vertex1_id_list.extend(vertices1_idx.tolist())
            vertex2_id_list.extend(vertices2_idx.tolist())
            
            if sample_idx <= impact_end:
                score_type1[sample_idx] = 1
                score_type2[sample_idx] = 1
            else:
                score_type1[sample_idx] = contact_type
                score_type2[sample_idx] = contact_type
            
            score_contact_area1[sample_idx] = face_area1
            score_contact_area2[sample_idx] = face_area2

            # score_vertex_ids1[sample_idx, vertices1_idx] = True
            tmp_vertex_ids1 = score_vertex_ids1[sample_idx]
            tmp_vertex_ids1[vertices1_idx] = True
            score_vertex_ids1[sample_idx] = tmp_vertex_ids1

            # score_vertex_ids2[sample_idx, vertices2_idx] = True
            tmp_vertex_ids2 = score_vertex_ids2[sample_idx]
            tmp_vertex_ids2[vertices2_idx] = True
            score_vertex_ids2[sample_idx] = tmp_vertex_ids2


    def _finalize_score_tracks(self, score_track1, score_track2, config_obj1, config_obj2, start_sample, stop_sample,score_type1, score_type2, score_vertex_ids1, score_vertex_ids2, score_contact_area1, score_contact_area2):
        """Add events to to score tracks."""
        score_track1.add_event(ScoreEvent(coll_obj=config_obj2.idx, start_sample=start_sample, stop_sample=stop_sample, type=score_type1, contact_area=score_contact_area1, vertex_ids=score_vertex_ids1))
        
        score_track2.add_event(ScoreEvent(coll_obj=config_obj1.idx, start_sample=start_sample, stop_sample=stop_sample, type=score_type2, contact_area=score_contact_area2, vertex_ids=score_vertex_ids2))

    def _update_modal_vertices(self, obj1_idx, obj2_idx, vertex1_id_list, vertex2_id_list, trajectory1, trajectory2, mesh1_faces, mesh2_faces):
        """Update modal vertices with collision data."""
        if len(vertex1_id_list) > 0 and len(vertex2_id_list) > 0:
            vertex1_id_list = np.unique(np.array(vertex1_id_list))
            vertex2_id_list = np.unique(np.array(vertex2_id_list))
            
            modal_vertices = self.entity_manager.get('modal_vertices')
            mod_v1 = mod_v2 = None
            
            for mv_idx in modal_vertices.keys():
                if modal_vertices[mv_idx].obj_idx == obj1_idx:
                    mod_v1 = modal_vertices[mv_idx]
                elif modal_vertices[mv_idx].obj_idx == obj2_idx:
                    mod_v2 = modal_vertices[mv_idx]
            
            if mod_v1 is not None:
                mod_v1.add_vertices(vertex1_id_list)
            else:
                modal_vertices1 = ModalVertices(
                    obj_idx=obj1_idx,
                    vertices=vertex1_id_list,
                    connected_area=len(vertex1_id_list) / len(mesh1_faces)
                )
                self.entity_manager.register('modal_vertices', modal_vertices1)
            
            if mod_v2 is not None:
                mod_v2.add_vertices(vertex2_id_list)
            else:
                modal_vertices2 = ModalVertices(
                    obj_idx=obj2_idx,
                    vertices=vertex2_id_list,
                    connected_area=len(vertex2_id_list) / len(mesh2_faces)
                )
                self.entity_manager.register('modal_vertices', modal_vertices2)

    def _connected_facing_face(self, obj1_idx, obj2_idx, trajectory1, trajectory2):
        """Ultra-optimized connected objects handling using pre-computed data."""
        distances_dir = f"{self.config.system.cache_path}/distances"
        distance_file = f"{distances_dir}/connected_{obj1_idx}_{obj2_idx}.npz"
    
        if not os.path.exists(distance_file):
            return
    
        distance_data = np.load(distance_file)
        collision_margin = distance_data[distance_data.files[0]]
        cp1 = distance_data[distance_data.files[1]]
        cp2 = distance_data[distance_data.files[2]]
    
        mesh1_faces = trajectory1.get_faces()
        mesh2_faces = trajectory2.get_faces()
    
        for config_obj in self.config.objects:
            if config_obj.idx == obj1_idx:
                proxy1 = config_obj.proxy_type
                is_proxy1 = config_obj.proxy_type is not False
            if config_obj.idx == obj2_idx:
                proxy2 = config_obj.proxy_type 
                is_proxy2 = config_obj.proxy_type is not False
    
        if is_proxy1 and is_proxy2:
            # Both are proxies - use analytical collision
        
            # Get mesh vertices (only need first frame for connected objects)
            mesh1_vertices = trajectory1.get_vertices(0)
            mesh2_vertices = trajectory2.get_vertices(0)
        
            # Use analytical collision for both
            vertices1_idx, _ = self.proxy_physics.analytical_proxy_collision(proxy1, mesh1_vertices, mesh1_faces, cp1, np.mean(mesh1_vertices, axis=0), collision_margin)
            vertices2_idx, _ = self.proxy_physics.analytical_proxy_collision(proxy2, mesh2_vertices, mesh2_faces, cp2, np.mean(mesh2_vertices, axis=0), collision_margin)
        elif (is_proxy1 and not is_proxy2) or (is_proxy2 and not is_proxy1):
            # Only one object is proxy
            proxed_mesh_vertices = trajectory1.get_vertices(0) if is_proxy1 else trajectory2.get_vertices(0)
        
            proxy = proxy1 if is_proxy1 else proxy2
            proxy_mesh_faces = mesh1_faces if is_proxy1 else mesh2_faces
            proxy_cp = cp1 if is_proxy1 else cp2
            proxed_vertices_idx, _ = self.proxy_physics.analytical_proxy_collision(proxy, proxed_mesh_vertices, proxy_mesh_faces, proxy_cp, np.mean(proxy_mesh_faces, axis=0), collision_margin)
        
            # Standard KDTree for object 2
            noproxy_mesh_vertices = trajectory2.get_vertices(0) if is_proxy1 else trajectory1.get_vertices(0)
            noproxy_tree = cKDTree(noproxy_mesh_vertices)
            noproxy_radius = collision_margin * 2.0
            noproxy_cp = cp2 if is_proxy1 else cp1
            noproxy_vertices_idx = np.array(noproxy_tree.query_ball_point(noproxy_cp, radius, workers=-1))
            vertices1_idx = proxed_vertices_idx if is_proxy1 else noproxy_vertices_idx
            vertices2_idx = noproxy_vertices_idx if is_proxy1 else proxed_vertices_idx
        else:
            # Standard detection
            mesh1_vertices = trajectory1.get_vertices(0)
            mesh2_vertices = trajectory2.get_vertices(0)
            
            tree1 = cKDTree(mesh1_vertices)
            tree2 = cKDTree(mesh2_vertices)
            
            radius = collision_margin * 2.0
            vertices1_idx = tree1.query_ball_point(cp1, radius, workers=-1)
            vertices2_idx = tree2.query_ball_point(cp2, radius, workers=-1)
        
        if vertices1_idx and vertices2_idx:
            vertices1_idx = np.array(vertices1_idx)
            vertices2_idx = np.array(vertices2_idx)
            
            mesh1_faces_idx = np.where(np.any(np.isin(mesh1_faces, vertices1_idx), axis=1))[0]
            mesh2_faces_idx = np.where(np.any(np.isin(mesh2_faces, vertices2_idx), axis=1))[0]
            
            cvidx1 = np.unique(mesh1_faces[mesh1_faces_idx].flatten())
            cvidx2 = np.unique(mesh2_faces[mesh2_faces_idx].flatten())
            
            face_area1 = len(mesh1_faces_idx) / len(mesh1_faces)
            face_area2 = len(mesh2_faces_idx) / len(mesh2_faces)
            
            self._update_modal_vertices(obj1_idx, obj2_idx, cvidx1.tolist(), cvidx2.tolist(), trajectory1, trajectory2, mesh1_faces, mesh2_faces)
