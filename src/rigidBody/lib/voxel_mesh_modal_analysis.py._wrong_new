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
        
        # Map voxel indices to global DOF indices
        self.voxel_to_dof = {}
        self.dof_to_voxel = {}
        n_nodes = 0
        
        # Each voxel has 8 corner nodes, each with 3 DOFs
        for voxel_idx, (i, j, k) in enumerate(active_voxels):
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        node = (i + di, j + dj, k + dk)
                        if node not in self.voxel_to_dof:
                            self.voxel_to_dof[node] = n_nodes
                            self.dof_to_voxel[n_nodes] = node
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
