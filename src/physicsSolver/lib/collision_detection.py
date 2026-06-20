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

from pbrAudioCommon import np
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple, List, Optional

def parallel_query_ball(tree_data: Tuple[np.ndarray, np.ndarray, float, int]) -> List[List[int]]:
    """
    Worker function for parallel KDTree query.
    
    Args:
        tree_data_data: Tuple containing (tree_points, query_points, radius, chunk_start)
    
    Returns:
        List of vertex indices within radius for each query point
    """
    tree_points, query_points, radius, chunk_start = tree_data
    tree = cKDTree(tree_points)
    results = tree.query_ball_point(query_points, radius, workers=-1)
    return results

def parallel_batch_query(tree_points: np.ndarray, 
                         query_points: np.ndarray, 
                         radius: float,
                         n_jobs: Optional[int] = None) -> List[List[int]]:
    """
    Parallel batch query using multiprocessing.
    
    Args:
        tree_points: Points to build the KDTree from
        query_points: Points to query
        radius: Search radius
        n_jobs: Number of parallel jobs (default: CPU count)
    
    Returns:
        List of vertex indices for each query point
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Split query points into chunks for parallel processing
    chunk_size = max(1, len(query_points) // (n_jobs * 2))  # 2x chunks for better load balancing
    chunks = []
    
    for i in range(0, len(query_points), chunk_size):
        chunk = query_points[i:i + chunk_size]
        chunks.append((tree_points, chunk, radius, i))
    
    # Process chunks in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(parallel_query_ball, chunks)
    
    # Flatten results
    all_results = []
    for chunk_result in results:
        all_results.extend(chunk_result)
    
    return all_results

class ParallelCollisionDetector:
    """
    Parallel collision detector using multiprocessing for KDTree queries.
    """
    
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs if n_jobs else cpu_count()
        self._pool = None
    
    def __enter__(self):
        self._pool = Pool(processes=self.n_jobs)
        return self
    
    def __exit__(self, exc_type, exc_val,, exc_tb):
        if self._pool:
            self._pool.close()
            self._pool.join()
    
    def detect_collisions_batch(self,
                                mesh1_vertices: np.ndarray,
                                mesh2_vertices: np.ndarray,
                                query_points1: np.ndarray,
                                query_points2: np.ndarray,
                                radius: float) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Detect collisions between two meshes in parallel.
        
        Args:
            mesh1_vertices: Vertices of mesh 1
            mesh2_vertices: Vertices of mesh 2
            query_points1: Query points on mesh 1
            query_points2: Query points on mesh 2
            radius: Search radius
        
        Returns:
            Tuple of (indices1, indices2) for each query point
        """
        # Build trees once (these are fast)
        tree1 = cKDTree(mesh1_vertices)
        tree2 = cKDTree(mesh2_vertices)
        
        # Prepare chunks for parallel processing
        chunk_size = max(1, len(query_points1) // (self.n_jobs * 2))
        chunks1 = []
        chunks2 = []
        
        for i in range(0, len(query_points1), chunk_size):
            chunk1 = query_points1[i:i + chunk_size]
            chunk2 = query_points2[i:i + chunk_size]
            chunks1.append((mesh2_vertices, chunk1, radius, i))
            chunks2.append((mesh1_vertices, chunk2, radius, i))
        
        # Process both directions in parallel
        with Pool(processes=self.n_jobs) as pool:
            results1 = pool.map(parallel_query_ball, chunks1)
            results2 = pool.map(parallel_query_ball, chunks2)
        
        # Flatten results
        indices1 = []
        indices2 = []
        for chunk_result in results1:
            indices1.extend(chunk_result)
        for chunk_result in results2:
            indices2.extend(chunk_result)
        
        return indices1, indices2

