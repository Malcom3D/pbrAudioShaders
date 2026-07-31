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

from ..core.entity_manager import EntityManager
from ..lib.collision_data import CollisionData
from ..lib.functions import _parse_lib
from ..lib.modal_vertices import ModalVertices
from ..lib.score_data import ScoreEvent, ScoreTrack
from ..lib.force_data import ContactType

@dataclass
class CollisionSolver:
    entity_manager: EntityManager
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')

    def _compute_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute normals of all faces."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return normals / norms

    def compute(self, collision: CollisionData) -> None:
        """Optimized collision solver with proxy mesh special handling."""
        config = self.entity_manager.get('config')
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
                vertices1_idx, vertices2_idx, face_area1, face_area2 = self._optimized_proxy_collision(
                    obj1_idx if is_proxy1 else obj1_idx,
                    obj2_idx if is_proxy2 else obj2_idx,
                    cp1, cp2, collision_margin, contact_type,
                    trajectory1, trajectory2,
                    mesh1_faces, mesh2_faces,
                    sample_idx
                )
            else:
                # Original method for non-proxy meshes
                vertices1_idx, vertices2_idx, face_area1, face_area2 = self._standard_collision(
                    cp1, cp2, collision_margin, contact_type,
                    trajectory1, trajectory2,
                    mesh1_faces, mesh2_faces,
                    sample_idx
                )
            
            # Update score data
            if vertices1_idx is not None and vertices2_idx is not None:
                self._update_score_data(
                    sample_idx, impact_end, contact_type,
                    vertices1_idx, vertices2_idx,
                    face_area1, face_area2,
                    score_type1, score_type2,
                    score_vertex_ids1, score_vertex_ids2,
                    score_contact_area1, score_contact_area2,
                    vertex1_id_list, vertex2_id_list,
                    config_obj1, config_obj2
                )

                print(f"facing faces between {config_obj1.name} and {config_obj2.name} at frame {sample_idx}: {np.count_nonzero(vertices1_idx)} {np.count_nonzero(vertices2_idx)} at distance {collision_margin} for {ContactType(contact_type).name.lower()}")

        # Finalize score tracks
        self._finalize_score_tracks(score_track1, score_track2, config_obj1, config_obj2, start_samples, stop_samples, score_type1, score_type2, score_vertex_ids1, score_vertex_ids2, score_contact_area1, score_contact_area2)
        
        # Update modal vertices
        self._update_modal_vertices(obj1_idx, obj2_idx, vertex1_id_list, vertex2_id_list, trajectory1, trajectory2, mesh1_faces, mesh2_faces)

    def _optimized_proxy_collision(self, obj1_idx, obj2_idx, cp1, cp2, collision_margin, contact_type, trajectory1, trajectory2, mesh1_faces, mesh2_faces, sample_idx):
        """
        Ultra-optimized collision detection for proxy meshes types 0, 1, 2.
        Uses analytical geometry instead of KDTree queries.
        """
        # Get proxy mesh properties
        for config_obj in self.config.objects:
            if config_obj.idx == obj1_idx:
                proxy1 = config_obj.proxy_type
            elif config_obj.idx == obj2_idx: 
                proxy2 = config_obj.proxy_type
    
        # Get current mesh vertices
        mesh1_vertices = trajectory1.get_vertices(sample_idx)
        mesh2_vertices = trajectory2.get_vertices(sample_idx)
    
        # Pre-compute centers once
        center1 = np.mean(mesh1_vertices, axis=0)
        center2 = np.mean(mesh2_vertices, axis=0)
    
        # Use analytical collision for proxy types 0, 1, 2
        if proxy1 is not False and proxy1 in [0,1,2]:
            vertices1_idx, face_area1 = self._analytical_proxy_collision(proxy1, mesh1_vertices, mesh1_faces, cp1, center1, collision_margin)
        else:
            # Fallback to KDTree for other types
            tree1 = cKDTree(mesh1_vertices)
            radius = collision_margin * (4.0 if contact_type in [4, 5] else 2.0)
            vertices1_idx = np.array(tree1.query_ball_point(cp1, radius, workers=-1))
            face_area1 = self._compute_face_area(vertices1_idx, mesh1_faces)
    
        if proxy2 is not False and proxy2 in [0,1,2]:
            vertices2_idx, face_area2 = self._analytical_proxy_collision(proxy2, mesh2_vertices, mesh2_faces, cp2, center2, collision_margin)
        else:
            tree2 = cKDTree(mesh2_vertices)
            radius = collision_margin * (4.0 if contact_type in [4, 5] else 2.0)
            vertices2_idx = np.array(tree2.query_ball_point(cp2, radius, workers=-1))
            face_area2 = self._compute_face_area(vertices2_idx, mesh2_faces)
    
        return vertices1_idx, vertices2_idx, face_area1, face_area2

    def _analytical_proxy_collision(self, proxy_type, vertices, faces, contact_point, center, collision_margin):
        """
        Analytical collision detection for proxy meshes.
        Uses geometric relationships instead of KDTree.
        """
        if proxy_type == 0:  # Pyramid (4 vertices)
            return self._pyramid_collision(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 1:  # Octahedron (6 vertices)
            return self._octahedron_collision(vertices, faces, contact_point, center, collision_margin)
        elif proxy_type == 2:  # Cube (8 vertices)
            return self._cube_collision(vertices, faces, contact_point, center, collision_margin)
    
        return np.array([], dtype=np.int32), 0.0

    def _pyramid_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast pyramid collision detection using barycentric coordinates.
        Pyramid has 4 vertices and 4 faces.
        """
        # Pyramid structure: apex at index 0, base at indices 1,2,3
        # The contact point direction from center tells us which face is hit
    
        direction = contact_point - center
        direction_norm = np.linalg.norm(direction)
    
        if direction_norm < 1e-10:
            return np.array([0, 1, 2, 3], dtype=np.int32), 1.0
    
        direction = direction / direction_norm
    
        # Pre-computed face normals for pyramid (from cache or compute on the fly)
        # Face 0: apex to base front (vertices 0,1,2)
        # Face 1: apex to base right (vertices 0,2,3)
        # Face 2: apex to base left (vertices 0,3,1)
        # Face 3: base (vertices 1,3,2)
    
        # Find which face the contact point is closest to using dot product
        face_normals = self._compute_face_normals(vertices, faces)
        dot_products = np.dot(face_normals, direction)
    
        # The closest face has the highest dot product
        closest_face = np.argmax(dot_products)
    
        # Get vertices for this face
        face_vertices = faces[closest_face]
        unique_vertices = np.unique(face_vertices)
    
        # Add neighboring vertices for smooth transition
        # This prevents artifacts at face boundaries
        if collision_margin > 0.01:
            # Add vertices from adjacent faces
            for i, face in enumerate(faces):
                if i != closest_face:
                    if len(np.intersect1d(face, face_vertices)) > 0:
                        unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
    
        face_area = len(unique_vertices) / len(vertices)
    
        return unique_vertices, face_area

    def _octahedron_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast octahedron collision detection using sign-based face selection.
        Octahedron has 6 vertices and 8 faces.
        """
        # Octahedron: vertices at (±1,0,0), (0,±1,0), (0,0,±1)
        # The contact point's octant tells us which face is hit
    
        direction = contact_point - center
        octant_signs = np.sign(direction)
    
        # Map octant to face index (pre-computed mapping for octahedron)
        # Faces are indexed by their normal direction
        octant_to_face = {
            (1, 1, 1): 0,   # Face normal: (1,1,1)/sqrt(3)
            (1, 1, -1): 1,  # Face normal: (1,1,-1)/sqrt(3)
            (1, -1, 1): 2,  # etc.
            (1, -1, -1): 3,
            (-1, 1, 1): 4,
            (-1, 1, -1): 5,
            (-1, -1, 1): 6,
            (-1, -1, -1): 7
        }
    
        octant_key = (octant_signs[0], octant_signs[1], octant_signs[2])
        face_idx = octant_to_face.get(octant_key, 0)
    
        # Get vertices for this face
        face_vertices = faces[face_idx]
        unique_vertices = np.unique(face_vertices)
    
        # Add adjacent face vertices for smooth transition
        if collision_margin > 0.01:
            for i, face in enumerate(faces):
                if i != face_idx:
                    if len(np.intersect1d(face, face_vertices)) > 0:
                        unique_vertices = np.unique(np.concatenate([unique_vertices, face]))
    
        face_area = len(unique_vertices) / len(vertices)
    
        return unique_vertices, face_area

    def _cube_collision(self, vertices, faces, contact_point, center, collision_margin):
        """
        Fast cube collision detection using axis-aligned bounding box.
        Cube has 8 vertices and 6 faces (12 triangles).
        """
        # Cube is axis-aligned, so we can use simple coordinate checks
        direction = contact_point - center
    
        # Find the dominant axis (which face is closest)
        abs_direction = np.abs(direction)
        dominant_axis = np.argmax(abs_direction)
        face_sign = np.sign(direction[dominant_axis])
    
        # For a cube, each face has 2 triangles
        # Face indices are grouped: [0,1] for +x, [2,3] for -x, etc.
        if dominant_axis == 0:  # x-axis
            face_indices = np.array([0, 1]) if face_sign > 0 else np.array([2, 3])
        elif dominant_axis == 1:  # y-axis
            face_indices = np.array([4, 5]) if face_sign > 0 else np.array([6, 7])
        else:  # z-axis
            face_indices = np.array([8, 9]) if face_sign > 0 else np.array([10, 11])
    
        # Get all vertices from these faces
        face_vertices = faces[face_indices].flatten()
        unique_vertices = np.unique(face_vertices)
    
        # Add edge vertices for smooth transition
        if collision_margin > 0.01:
            # Add vertices from adjacent faces (sharing an edge)
            edge_faces = self._get_adjacent_faces(faces, face_indices)
            for edge_face in edge_faces:
                unique_vertices = np.unique(np.concatenate([unique_vertices, faces[edge_face]]))
    
        face_area = len(unique_vertices) / len(vertices)
    
        return unique_vertices, face_area

    def _compute_face_area(self, vertices_idx, faces):
        """Compute face area ratio from vertex indices."""
        if len(vertices_idx) == 0:
            return 0.0
    
        mesh_faces_idx = np.where(np.any(np.isin(faces, vertices_idx), axis=1))[0]
        return len(mesh_faces_idx) / len(faces)

    def _get_adjacent_faces(self, faces, face_indices):
        """Get faces adjacent to the given face indices."""
        adjacent = set()
        face_vertices_set = set(faces[face_indices].flatten())
    
        for i, face in enumerate(faces):
            if i not in face_indices:
                if len(set(face.flatten()) & face_vertices_set) >= 2:  # Share an edge
                    adjacent.add(i)
    
        return list(adjacent)

    def _octant_to_face_indices(self, octant: np.ndarray, proxy_type: int = 0) -> np.ndarray:
        """
        Map octant sign to face indices for proxy meshes.
        
        For pyramid (proxy_type=0): 4 vertex, 4 faces
        For octahedron (proxy_type=1): 6 vertex, 8 faces
        For octant/cube (proxy_type=2): 8 vertex, 6 faces
        
        Returns face indices for the given octant.
        """
        if proxy_type == 0:
            # Pyramid has 4 faces
            # Faces are: [apex, base1, base2], [apex, base2, base3], [apex, base3, base1], [base1, base3, base2]
            # Apex is at +x, base is at -x
            # Map based on which quadrant the contact point is in
            if octant[1] >= 0 and octant[2] >= 0:
                return np.array([0])  # Front face
            elif octant[1] >= 0 and octant[2] < 0:
                return np.array([1])  # Right face
            elif octant[1] < 0 and octant[2] >= 00:
                return np.array([2])  # Left face
            else:
                return np.array([3])  # Base face
        elif proxy_type == 1:
            # Octahedron has 8 faces
            octant_key = (octant[0] >= 0, octant[1] >= 0, octant[2] >= 0)
            face_map = {
                (True, True, True): 0,
                (True, True, False): 1,
                (True, False, True): 2,
                (True, False, False): 3,
                (False, True, True): 4,
                (False, True, False): 5,
                (False, False, True): 6,
                (False, False, False): 7
            }
            return np.array([face_map.get(octant_key, 0)])
        elif proxy_type == 2:
            # Cube has 6 faces
            # Determine which face is closest based on the dominant axis
            abs_octant = np.abs(octant)
            dominant_axis = np.argmax(abs_octant)
            if dominant_axis == 0:  # x-axis
                return np.array([0, 1]) if octant[0] >= 0 else np.array([2, 3])
            elif dominant_axis == 1:  # y-axis
                return np.array([4, 5]) if octant[1] >= 0 else np.array([6, 7])
            else:  # z-axis
                return np.array([8, 9]) if octant[2] >= 0 else np.array([10, 11])
        else:
            # Default to octahedron behavior
            return np.array([0])

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
        stop_samples = self._adjust_for_fracture_shard(
            stop_samples, start_samples, sample_rate, sfps,
            config_obj1, config_obj2
        )
        
        return start_samples, stop_samples, impact_end

    def _adjust_for_fracture_shard(self, stop_samples, start_samples, sample_rate, sfps,
                                    config_obj1, config_obj2):
        """Adjust sample range for fracture and shard events."""
        fracture_frames = []
        shard_frames = []
        
        for config_obj in [config_obj1, config_obj2]:
            if config_obj is not None:
                if config_obj.fractured is not False:
                    fracture_frames.append(config_obj.fractured * sample_rate / sfps)
                if config_obj.is_shard is not False:
                    shard_frames.append(config_obj.is_shard * sample_rate / sfps)
        
        # Adjust for shard frames
        valid_shard_frames = [f for f in shard_frames if start_samples <= f <= stop_samples]
        if valid_shard_frames:
            start_samples = max(valid_shard_frames)
        
        # Adjust for fracture frames
        valid_fracture_frames = [f for f in fracture_frames if start_samples <= f <= stop_samples]
        if valid_fracture_frames:
            stop_samples = min(valid_fracture_frames)
        
        return stop_samples

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

    def _update_score_data(self, sample_idx, impact_end, contact_type,
                            vertices1_idx, vertices2_idx,
                            face_area1, face_area2,
                            score_type1, score_type2,
                            score_vertex_ids1, score_vertex_ids2,
                            score_contact_area1, score_contact_area2,
                            vertex1_id_list, vertex2_id_list,
                            config_obj1, config_obj2):
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
                is_proxy1 = True if config_obj.proxy_type is not False and config_obj.proxy_type in [0,1,2] else False
            if config_obj.idx == obj2_idx:
                proxy2 = config_obj.proxy_type 
                is_proxy2 = True if config_obj.proxy_type is not False and config_obj.proxy_type in [0,1,2] else False
    
        if is_proxy1 and is_proxy2:
            # Both are proxies - use analytical collision
        
            # Get mesh vertices (only need first frame for connected objects)
            mesh1_vertices = trajectory1.get_vertices(0)
            mesh2_vertices = trajectory2.get_vertices(0)
        
            # Use analytical collision for both
            vertices1_idx, _ = self._analytical_proxy_collision(proxy1, mesh1_vertices, mesh1_faces, cp1, np.mean(mesh1_vertices, axis=0), collision_margin)
            vertices2_idx, _ = self._analytical_proxy_collision(proxy2, mesh2_vertices, mesh2_faces, cp2, np.mean(mesh2_vertices, axis=0), collision_margin)
        elif is_proxy1:
            # Only object 1 is proxy
            mesh1_vertices = trajectory1.get_vertices(0)
        
            vertices1_idx, _ = self._analytical_proxy_collision(proxy1, mesh1_vertices, mesh1_faces, cp1, np.mean(mesh1_vertices, axis=0), collision_margin)
        
            # Standard KDTree for object 2
            mesh2_vertices = trajectory2.get_vertices(0)
            tree2 = cKDTree(mesh2_vertices)
            radius = collision_margin * 2.0
            vertices2_idx = np.array(tree2.query_ball_point(cp2, radius, workers=-1))
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
            
            self._update_modal_vertices(
                obj1_idx, obj2_idx,
                cvidx1.tolist(), cvidx2.tolist(),
                trajectory1, trajectory2,
                mesh1_faces, mesh2_faces
            )
