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

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple, Optional, Any
import json

class VoxelMeshModalAnalysis:
    def __init__(self, voxel_grid, voxel_size, material_properties):
        """
        Initialize voxel mesh modal analysis
        
        Parameters:
        -----------
        voxel_grid : 3D numpy array (boolean or int)
            Binary array where True/1 indicates solid material
        voxel_size : float or tuple
            Size of each voxel (can be uniform or anisotropic)
        material_properties : dict
            Dictionary containing material properties
        """
        self.voxel_grid = np.array(voxel_grid, dtype=bool)
        self.nx, self.ny, self.nz = self.voxel_grid.shape
        
        # Handle voxel size (uniform or anisotropic)
        if isinstance(voxel_size, (int, float)):
            self.dx = self.dy = self.dz = float(voxel_size)
        else:
            self.dx, self.dy, self.dz = voxel_size
        
        # Material properties
        self.E = material_properties['young_modulus']
        self.nu = material_properties['poisson_ratio']
        self.rho = material_properties['density']
        self.alpha = material_properties.get('rayleigh_alpha', 0.0)
        self.beta = material_properties.get('rayleigh_beta', 0.0)
        
        # Compute Lamé parameters
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        
        # Initialize system matrices
        self.K = None  # Stiffness matrix
        self.M = None  # Mass matrix
        self.C = None  # Damping matrix

        # For interpolation: node index grid and origin
        self.node_index_grid = None
        self.origin = None
        
    def build_system_matrices(self):
        """
        Build stiffness and mass matrices using voxel-based FEM
        Uses trilinear hexahedral elements (8-node bricks)
        """
        # Get active voxels
        active_voxels = np.argwhere(self.voxel_grid)
        n_voxels = len(active_voxels)
        
        if n_voxels == 0:
            raise ValueError("No active voxels found in the grid")

        # Determine bounding box for node indexing
        min_idx = active_voxels.min(axis=0)
        max_idx = active_voxels.max(axis=0)
        self.origin = min_idx * np.array([self.dx, self.dy, self.dz])
        
        # Map voxel indices to global DOF indices
        self.voxel_to_dof = {}
        self.dof_to_voxel = {}
        n_nodes = 0

        # Initialize node_index_grid with -1
        self.node_index_grid = -np.ones((self.nx+1, self.ny+1, self.nz+1), dtype=int)
        
        # Each voxel has 8 corner nodes, each with 3 DOFs
        for voxel_idx, (i, j, k) in enumerate(active_voxels):
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        node = (i + di, j + dj, k + dk)
                        if node not in self.voxel_to_dof:
                            self.voxel_to_dof[node] = n_nodes
                            self.dof_to_voxel[n_nodes] = node
                            # Fill node_index_grid
                            ni, nj, nk = node
                            self.node_index_grid[ni, nj, nk] = n_nodes
                            n_nodes += 1
        
        n_dofs = n_nodes * 3
        
        # Initialize sparse matrices
        rows_K, cols_K, data_K = [], [], []
        rows_M, cols_M, data_M = [], [], []
        
        # Element stiffness and mass matrices for 8-node brick
        Ke, Me = self._get_element_matrices()
        
        # Assemble global matrices
        for voxel_idx, (i, j, k) in enumerate(active_voxels):
            # Get global node indices for this voxel's corners
            element_dofs = []
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        node = (i + di, j + dj, k + dk)
                        node_idx = self.voxel_to_dof[node]
                        for dof in range(3):
                            element_dofs.append(node_idx * 3 + dof)
            
            # Add to global matrices
            for a in range(24):  # 8 nodes * 3 DOFs
                for b in range(24):
                    if abs(Ke[a, b]) > 1e-15:
                        rows_K.append(element_dofs[a])
                        cols_K.append(element_dofs[b])
                        data_K.append(Ke[a, b])
                    
                    if abs(Me[a, b]) > 1e-15:
                        rows_M.append(element_dofs[a])
                        cols_M.append(element_dofs[b])
                        data_M.append(Me[a, b])
        
        # Build sparse matrices
        self.K = csr_matrix((data_K, (rows_K, cols_K)), shape=(n_dofs, n_dofs))
        self.M = csr_matrix((data_M, (rows_M, cols_M)), shape=(n_dofs, n_dofs))
        
        # Build damping matrix if needed
        if self.alpha > 0 or self.beta > 0:
            self.C = self.alpha * self.M + self.beta * self.K
        
        return n_dofs
    
    def _get_element_matrices(self):
        """
        Get element stiffness and mass matrices for 8-node hexahedral element
        Using standard trilinear shape functions
        """
        # Gauss quadrature points (2x2x2)
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        
        # Material matrix
        D = self._get_constitutive_matrix()
        
        # Initialize element matrices
        Ke = np.zeros((24, 24))
        Me = np.zeros((24, 24))
        
        # Integration loop
        for xi in gp:
            for eta in gp:
                for zeta in gp:
                    # Shape function derivatives at Gauss point
                    dN_dxi = 0.125 * np.array([
                        -(1-eta)*(1-zeta), (1-eta)*(1-zeta),
                        (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
                        -(1-eta)*(1+zeta), (1-eta)*(1+zeta),
                        (1+eta)*(1+zeta), -(1+eta)*(1+zeta)
                    ])
                    dN_deta = 0.125 * np.array([
                        -(1-xi)*(1-zeta), -(1+xi)*(1-zeta),
                        (1+xi)*(1-zeta), (1-xi)*(1-zeta),
                        -(1-xi)*(1+zeta), -(1+xi)*(1+zeta),
                        (1+xi)*(1+zeta), (1-xi)*(1+zeta)
                    ])
                    dN_dzeta = 0.125 * np.array([
                        -(1-xi)*(1-eta), -(1+xi)*(1-eta),
                        -(1+xi)*(1+eta), -(1-xi)*(1+eta),
                        (1-xi)*(1-eta), (1+xi)*(1-eta),
                        (1+xi)*(1+eta), (1-xi)*(1+eta)
                    ])
                    
                    # Jacobian determinant
                    detJ = self.dx * self.dy * self.dz / 8
                    
                    # Build B matrix (strain-displacement)
                    B = np.zeros((6, 24))
                    for i in range(8):
                        B[0, 3*i] = dN_dxi[i] * 2/self.dx
                        B[1, 3*i+1] = dN_deta[i] * 2/self.dy
                        B[2, 3*i+2] = dN_dzeta[i] * 2/self.dz
                        B[3, 3*i] = dN_deta[i] * 2/self.dy
                        B[3, 3*i+1] = dN_dxi[i] * 2/self.dx
                        B[4, 3*i+1] = dN_dzeta[i] * 2/self.dz
                        B[4, 3*i+2] = dN_deta[i] * 2/self.dy
                        B[5, 3*i] = dN_dzeta[i] * 2/self.dz
                        B[5, 3*i+2] = dN_dxi[i] * 2/self.dx
                    
                    # Add to element matrices
                    Ke += B.T @ D @ B * detJ
                    
                    # Mass matrix (consistent)
                    N = 0.125 * np.array([
                        (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
                        (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
                        (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
                        (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta)
                    ])
                    
                    for a in range(8):
                        for b in range(8):
                            for dof in range(3):
                                Me[3*a+dof, 3*b+dof] += self.rho * N[a] * N[b] * detJ
        
        return Ke, Me
    
    def _get_constitutive_matrix(self):
        """
        Get constitutive matrix for isotropic material
        """
        D = np.zeros((6, 6))
        c = self.lambda_ + 2 * self.mu
        
        D[0, 0] = D[1, 1] = D[2, 2] = c
        D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = self.lambda_
        D[3, 3] = D[4, 4] = D[5, 5] = self.mu
        
        return D
    
    def compute_modes(self, n_modes=10, min_freq=0, max_freq=None):
        """
        Compute modal modes (eigenvalues and eigenvectors)
        """
        if self.K is None or self.M is None:
            self.build_system_matrices()
        
        # Compute eigenvalues and eigenvectors
        if max_freq is not None:
            sigma = (2 * np.pi * max_freq) ** 2
        else:
            sigma = 0
        
        k = min(n_modes, self.K.shape[0] - 2)
        if k <= 0:
            return np.array([]), np.array([])
        
        eigenvalues, eigenvectors = eigsh(
            self.K, 
            k=k,
            M=self.M,
            sigma=sigma,
            which='LM'
        )
        
        # Sort by frequency
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert to frequencies (Hz)
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        
        # Filter by frequency range
        mask = (frequencies >= min_freq)
        if max_freq is not None:
            mask &= (frequencies <= max_freq)
        
        return frequencies[mask], eigenvectors[:, mask]

    def interpolate_mode_shapes_to_points(self, points: np.ndarray, mode_shapes: np.ndarray) -> np.ndarray:
        """
        Interpolate modal displacement magnitudes to arbitrary points (e.g., original mesh vertices).

        Parameters:
        -----------
        points : np.ndarray, shape (N, 3)
            Coordinates of points where to evaluate mode shapes.
        mode_shapes : np.ndarray, shape (n_dofs, n_modes)
            Mode shape matrix (each column is a mode, each row is a DOF).

        Returns:
        --------
        gains : np.ndarray, shape (N, n_modes)
            Magnitude of displacement at each point for each mode.
        """
        if self.node_index_grid is None:
            raise RuntimeError("Must call build_system_matrices first.")

        N = points.shape[0]
        n_modes = mode_shapes.shape[1]
        gains = np.zeros((N, n_modes), dtype=np.float64)

        # Precompute node positions from index grid (only for nodes that exist)
        # We'll use the grid to get node indices; positions are computed on the fly.

        for p_idx, pt in enumerate(points):
            # Compute voxel cell indices (clamp to grid)
            # origin is the minimum corner of the grid
            i = (pt[0] - self.origin[0]) / self.dx
            j = (pt[1] - self.origin[1]) / self.dy
            k = (pt[2] - self.origin[2]) / self.dz

            # Clamp to valid range for interpolation
            i = np.clip(i, 0, self.nx - 1)
            j = np.clip(j, 0, self.ny - 1)
            k = np.clip(k, 0, self.nz - 1)

            # Get surrounding node indices
            i0, i1 = int(np.floor(i)), int(np.ceil(i))
            j0, j1 = int(np.floor(j)), int(np.ceil(j))
            k0, k1 = int(np.floor(k)), int(np.ceil(k))

            # Clamp to valid indices (ensuring we stay within grid)
            i0 = max(0, min(i0, self.nx))
            i1 = max(0, min(i1, self.nx))
            j0 = max(0, min(j0, self.ny))
            j1 = max(0, min(j1, self.ny))
            k0 = max(0, min(k0, self.nz))
            k1 = max(0, min(k1, self.nz))

            # Gather node indices for the 8 corners
            corners = [
                (i0, j0, k0), (i1, j0, k0),
                (i0, j1, k0), (i1, j1, k0),
                (i0, j0, k1), (i1, j0, k1),
                (i0, j1, k1), (i1, j1, k1)
            ]
            node_indices = []
            for (ci, cj, ck) in corners:
                nidx = self.node_index_grid[ci, cj, ck]
                if nidx == -1:
                    found = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                ni2 = ci + di
                                nj2 = cj + dj
                                nk2 = ck + dk
                                if 0 <= ni2 <= self.nx and 0 <= nj2 <= self.ny and 0 <= nk2 <= self.nz:
                                    alt_idx = self.node_index_grid[ni2, nj2, nk2]
                                    if alt_idx != -1:
                                        node_indices.append(alt_idx)
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        # Point is outside valid domain.
                        break  # will leave gain as 0
                else:
                    node_indices.append(nidx)

            if len(node_indices) != 8:
                # If we couldn't get 8 nodes, set gains to 0 for this point
                continue

            # Compute interpolation weights for trilinear interpolation
            di = i - i0
            dj = j - j0
            dk = k - k0

            # Compute the displacement at the point for each mode.
            for mode_idx in range(n_modes):
                # Gather displacement vectors for the 8 corner nodes
                disp_vectors = []
                for nidx in node_indices:
                    # Each node has 3 DOFs: nidx*3, nidx*3+1, nidx*3+2
                    dx = mode_shapes[nidx*3, mode_idx]
                    dy = mode_shapes[nidx*3+1, mode_idx]
                    dz = mode_shapes[nidx*3+2, mode_idx]
                    disp_vectors.append(np.array([dx, dy, dz]))

                # Trilinear interpolation on each component
                # c000, c100, c010, c110, c001, c101, c011, c111
                c000, c100, c010, c110, c001, c101, c011, c111 = disp_vectors

                # Interpolate x component
                x00 = c000[0] * (1 - di) + c100[0] * di
                x01 = c001[0] * (1 - di) + c101[0] * di
                x10 = c010[0] * (1 - di) + c110[0] * di
                x11 = c011[0] * (1 - di) + c111[0] * di
                x0 = x00 * (1 - dj) + x10 * dj
                x1 = x01 * (1 - dj) + x11 * dj
                x = x0 * (1 - dk) + x1 * dk

                # y component
                y00 = c000[1] * (1 - di) + c100[1] * di
                y01 = c001[1] * (1 - di) + c101[1] * di
                y10 = c010[1] * (1 - di) + c110[1] * di
                y11 = c011[1] * (1 - di) + c111[1] * di
                y0 = y00 * (1 - dj) + y10 * dj
                y1 = y01 * (1 - dj) + y11 * dj
                y = y0 * (1 - dk) + y1 * dk

                # z component
                z00 = c000[2] * (1 - di) + c100[2] * di
                z01 = c001[2] * (1 - di) + c101[2] * di
                z10 = c010[2] * (1 - di) + c110[2] * di
                z11 = c011[2] * (1 - di) + c111[2] * di
                z0 = z00 * (1 - dj) + z10 * dj
                z1 = z01 * (1 - dj) + z11 * dj
                z = z0 * (1 - dk) + z1 * dk

                # Magnitude (Euclidean norm)
                magnitude = np.sqrt(x*x + y*y + z*z)
                gains[p_idx, mode_idx] = magnitude

        return gains









