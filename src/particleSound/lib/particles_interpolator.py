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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import numba as nb
from scipy.spatial.transform import Rotation, Slerp
import warnings

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _load_particle
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix


@dataclass
class ParticlesInterpolator:
    """
    High-performance interpolator for exported particle animations.
    
    Handles interpolation of particle positions, rotations, and sizes
    between exported frames. Optimized for large particle counts.
    
    This class is adapted to work within the pbrAudio architecture,
    using the EntityManager for configuration and registration.
    """
    
    entity_manager: EntityManager
    particles_idx: int = None
    
    # Interpolation parameters
    use_numba: bool = True
    cache_frames: bool = True
    max_cache_size: int = 100  # Maximum number of frames to cache
    
    # Internal state (not part of dataclass init)
    _initialized: bool = False
    _frames: List[int] = field(default_factory=list)
    _frame_cache: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    _particle_count: int = 0
    _particle_data: Dict[str, np.ndarray] = field(default_factory=dict)
    _is_static: bool = False
    _obj_path: str = None
    _particle_name: str = None
    
    def __post_init__(self):
        """Initialize the ParticleInterpolator."""
        if not self._initialized:
            config = self.entity_manager.get('config')
            
            set_debug(config.system.debug)
            set_debug_prefix(self.__class__.__name__)
            
            # Get particle configuration
            self._setup_particle_config()
            
            # Initialize Numba if requested
            if self.use_numba:
                self._setup_numba_kernels()
            
            # Discover available frames
            self._discover_frames()
            
            # Pre-allocate arrays
            self._preallocate_arrays()
            
            self._initialized = True
    
    def _setup_particle_config(self):
        """Setup particle configuration from entity manager."""
        config = self.entity_manager.get('config')
        
        # Find particle configuration
        particle_cfg = None
        for p in config.particles:
            if p.idx == self.particles_idx:
                particle_cfg = p
                break
        
        if particle_cfg is None:
            raise ValueError(f"Particle system {self.particles_idx} not found in config")
        
        # Store particle configuration
        self._particle_cfg = particle_cfg
        self._particle_name = particle_cfg.name
        self._obj_path = particle_cfg.obj_path
        self._is_static = particle_cfg.static
    
    def _setup_numba_kernels(self):
        """Setup Numba JIT compiled kernels for interpolation."""
        try:
            import numba as nb
            
            @nb.jit(nopython=True, parallel=True, cache=True)
            def _interpolate_positions_numba(pos0, pos1, t):
                """Linear interpolation of positions using Numba."""
                result = np.empty_like(pos0)
                for i in nb.prange(pos0.shape[0]):
                    for j in range(3):
                        result[i, j] = pos0[i, j] * (1.0 - t) + pos1[i, j] * t
                return result
            
            @nb.jit(nopython=True, parallel=True, cache=True)
            def _interpolate_sizes_numba(size0, size1, t):
                """Linear interpolation of sizes using Numba."""
                result = np.empty_like(size0)
                for i in nb.prange(size0.shape[0]):
                    result[i] = size0[i] * (1.0 - t) + size1[i] * t
                return result
            
            @nb.jit(nopython=True, parallel=True, cache=True)
            def _interpolate_quaternions_numba(q0, q1, t):
                """Spherical linear interpolation (slerp) of quaternions using Numba."""
                result = np.empty_like(q0)
                for i in nb.prange(q0.shape[0]):
                    # Normalize quaternions
                    w0 = q0[i, 0]
                    x0 = q0[i, 1]
                    y0 = q0[i, 2]
                    z0 = q0[i, 3]
                    w1 = q1[i, 0]
                    x1 = q1[i, 1]
                    y1 = q1[i, 2]
                    z1 = q1[i, 3]
                    
                    # Calculate dot product
                    dot = w0*w1 + x0*x1 + y0*y1 + z0*z1
                    
                    # If dot is negative, negate one quaternion
                    if dot < 0.0:
                        w1 = -w1
                        x1 = -x1
                        y1 = -y1
                        z1 = -z1
                        dot = -dot
                    
                    # Clamp dot to valid range
                    if dot > 1.0:
                        dot = 1.0
                    elif dot < -1.0:
                        dot = -1.0
                    
                    # Calculate interpolation factors
                    theta_0 = np.arccos(dot)
                    sin_theta_0 = np.sin(theta_0)
                    
                    if sin_theta_0 > 1e-6:
                        # Standard slerp
                        theta = theta_0 * t
                        sin_theta = np.sin(theta)
                        
                        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
                        s1 = sin_theta / sin_theta_0
                    else:
                        # Handle near-zero angle case
                        s0 = 1.0 - t
                        s1 = t
                    
                    # Interpolate
                    result[i, 0] = s0 * w0 + s1 * w1
                    result[i, 1] = s0 * x0 + s1 * x1
                    result[i, 2] = s0 * y0 + s1 * y1
                    result[i, 3] = s0 * z0 + s1 * z1
                    
                    # Normalize result
                    norm = np.sqrt(result[i, 0]**2 + result[i, 1]**2 + 
                                  result[i, 2]**2 + result[i, 3]**2)
                    if norm > 0:
                        result[i, 0] /= norm
                        result[i, 1] /= norm
                        result[i, 2] /= norm
                        result[i, 3] /= norm
                
                return result
            
            self._interp_pos_numba = _interpolate_positions_numba
            self._interp_size_numba = _interpolate_sizes_numba
            self._interp_quat_numba = _interpolate_quaternions_numba
            
        except ImportError:
            warnings.warn("Numba not available, using NumPy only")
            self.use_numba = False
    
    def _discover_frames(self):
        """Discover available frame files in the data directory."""
        if self._is_static:
            # Static particle system
            pattern = f"{self._particle_name}.npz"
            file_path = Path(self._obj_path) / pattern
            if file_path.exists():
                self._frames = [0]  # Single frame
        else:
            # Animated particle system
            pattern = f"{self._particle_name}_*.npz"
            obj_dir = Path(self._obj_path)
            frame_files = sorted(obj_dir.glob(pattern))
            
            # Extract frame numbers from fil filenames
            for file_path in frame_files:
                try:
                    # Extract number from filename like "name_00001.npz"
                    frame_str = file_path.stem.split('_')[-1]
                    frame_num = int(frame_str)
                    self._frames.append(frame_num)
                except ValueError:
                    continue
            
            if not self._frames:
                # Try loading all npz files if pattern doesn't match
                npz_files = sorted(obj_dir.glob("*.npz"))
                for file_path in npz_files:
                    try:
                        # Extract number from filename
                        frame_str = file_path.stem.split('_')[-1]
                        frame_num = int(frame_str)
                        self._frames.append(frame_num)
                    except ValueError:
                        # If can't extract frame number, try to use index
                        pass
                
                # If still no frames, try sequential numbering
                if not self._frames and npz_files:
                    self._frames = list(range(len(npz_files)))
            
            if not self._frames:
                raise ValueError(f"No particle data found in {self._obj_path}")
    
    def _preallocate_arrays(self):
        """Pre-allocate arrays based on particle count from first frame."""
        if not self._frames:
            return
        
        # Load first frame to get dimensions
        first_frame = self.load_frame(self._frames[0])
        if first_frame is not None:
            self._particle_count = first_frame['particle_count']
            debug_print(f"ParticleInterpolator: Found {self._particle_count} particles across {len(self._frames)} frames")
    
    def load_frame(self, frame: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load particle data for a specific frame.
        
        Args:
            frame: Frame number to load
            
        Returns:
            Dictionary containing particle data or None if frame not found
        """
        # Check cache first
        if frame in self._frame_cache:
            return self._frame_cache[frame]
        
        # Determine file path
        if self._is_static or len(self._frames) == 1:
            file_path = Path(self._obj_path) / f"{self._particle_name}.npz"
        else:
            # Try multiple naming conventions
            file_path = Path(self._obj_path) / f"{self._particle_name}_{frame:05d}.npz"
            if not file_path.exists():
                file_path = Path(self._obj_path) / f"{self._particle_name}_{frame}.npz"
            if not file_path.exists():
                # Try finding the frame in the list
                npz_files = sorted(Path(self._obj_path).glob("*.npnpz"))
                if frame < len(npz_files):
                    file_path = npz_files[frame]
                else:
                    return None
        
        if not file_path.exists():
            return None
        
        # Load data
        try:
            with np.load(file_path) as data:
                frame_data = {
                    'positions': data['positions'].astype(np.float64),
                    'rotations': data['rotations'].astype(np.float64),
                    'sizes': data['sizes'].astype(np.float64),
                    'states': data['states'].astype(np.int8) if 'states' in data else np.ones(data['positions'].shape[0], dtype=np.int8),
                    'particle_count': int(data['particle_count']) if 'particle_count' in data else data['positions'].shape[0]
                }
                
                # Convert euler rotations to quaternions for better interpolation
                frame_data['quaternions'] = self._euler_to_quat(frame_data['rotations'])
                
                # Cache if requested and cache isn't too large
                if self.cache_frames:
                    if len(self._frame_cache) >= self.max_cache_size:
                        # Remove oldest entry (simple FIFO)
                        oldest_key = next(iter(self._frame_cache))
                        del self._frame_cache[oldest_key]
                    self._frame_cache[frame] = frame_data
                
                return frame_data
                
        except Exception as e:
            debug_print(f"Error loading frame {frame}: {e}")
            return None
    
    def _euler_to_quat(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert euler angles (XYZ) to quaternions.
        
        Args:
            euler_angles: Array of shape (N, 3) containing euler angles
            
        Returns:
            Array of shape (N, 4) containing quaternions (w, x, y, z)
        """
        # Use scipy for vectorized conversion
        try:
            from scipy.spatial.transform import Rotation
            quats = Rotation.from_euler('XYZ', euler_angles).as_quat()
            # scipy returns (x, y, z, w), convert to (w, x, y, z)
            return np.roll(quats, 1, axis=1)
        except ImportError:
            # Manual implementation as fallback
            quats = np.zeros((euler_angles.shape[0], 4))
            
            # Pre-calculate half angles
            half = euler_angles * 0.5
            cx, cy, cz = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
            sx, sy, sz = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])
            
            # Quaternion components (w, x, y, z)
            quats[:, 0] = cx * cy * cz + sx * sy * sz  # w
            quats[:, 1] = sx * cy * cz - cx * sy * sz  # x
            quats[:, 2] = cx * sy * cz + sx * cy * sz  # y
            quats[:, 3] = cx * cy * sz - sx * sy * cz  # z
            
            # Normalize
            norms = np.linalg.norm(quats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            quats = quats / norms
            
            return quats
    
    def _quat_to_euler(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Convert quaternions back to euler angles (XYZ).
        
        Args:
            quaternions: Array of shape (N, 4) containing quaternionsions (w, x, y, z)
            
        Returns:
            Array of shape (N, 3) containing euler angles
        """
        try:
            from scipy.spatial.transform import Rotation
            # Convert from (w, x, y, z) to (x, y, z, w)
            quats_xyz_w = np.roll(quaternions, -1, axis=1)
            euler = Rotation.from_quat(quats_xyz_w).as_euler('XYZ')
            return euler
        except ImportError:
            # Manual implementation as fallback
            euler = np.zeros((quaternions.shape[0], 3))
            
            # Extract quaternion components
            w = quaternions[:, 0]
            x = quaternions[:, 1]
            y = quaternions[:, 2]
            z = quaternions[:, 3]
            
            # Compute euler angles (XYZ convention)
            # Roll (x-axis rotation)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            euler[:, 0] = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2.0 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            euler[:, 1] = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            euler[:, 2] = np.arctan2(siny_cosp, cosy_cosp)
            
            return euler
    
    def interpolate(self, sample_idx: float, attributes: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Interpolate particle data at arbitrary time point.
        
        Args:
            sample_idx: Sample index (can be fractional)
            attributes: List of attributes to interpolate ('positions', 'rotations', 'sizes')
            
        Returns:
            Dictionary with interpolated particle data
        """
        if attributes is None:
            attributes = ['positions', 'rotations', 'sizes']
        
        # Convert sample_idx to frame number
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = (fps / fps_base) * subframes  # subframes per second
        
        frame = sample_idx * sfps / sample_rate
        
        # Handle static case
        if self._is_static or len(self._frames) == 1:
            frame_data = self.load_frame(self._frames[0])
            if frame_data is None:
                return None
            result = {}
            for attr in attributes:
                if attr == 'rotations':
                    # Return in original format (euler)
                    result[attr] = self._quat_to_euler(frame_data['quaternions'])
                elif attr in frame_data:
                    result[attr] = frame_data[attr].copy()
            return result
        
        # Find surrounding frames
        frame = float(frame)
        lower_frame = int(np.floor(frame))
        upper_frame = int(np.ceil(frame))
        
        # Clamp to valid range
        lower_frame = max(self._frames[0], min(lower_frame, self._frames[-1]))
        upper_frame = max(self._frames[0], min(upper_frame, self._frames[-1]))
        
        if lower_frame == upper_frame:
            # Exact frame match
            frame_data = self.load_frame(lower_frame)
            if frame_data is None:
                return None
            result = {}
            for attr in attributes:
                if attr == 'rotations':
                    result[attr] = self._quat_to_euler(frame_data['quaternions'])
                elif attr in frame_data:
                    result[attr] = frame_data[attr].copy()
            return result
        
        # Load surrounding frames
        frame_lower = self.load_frame(lower_frame)
        frame_upper = self.load_frame(upper_frame)
        
        if frame_lower is None or frame_upper is None:
            debug_print(f"Cannot interpolate: missing frame data for frames {lower_frame}-{upper_frame}")
            return None
        
        # Calculate interpolation factor
        t = (frame - lower_frame) / (upper_frame - lower_frame)
        
        # Initialize result
        result = {}
        
        # Interpolate positions
        if 'positions' in attributes:
            if self.use_numba and hasattr(self, '_interp_pos_numba'):
                result['positions'] = self._interp_pos_numba(frame_lower['positions'], frame_upper['positions'], t)
            else:
                result['positions'] = (frame_lower['positions'] * (1.0 - t) + frame_upper['positions'] * t)
        
        # Interpolate quaternions and convert back to euler
        if 'rotations' in attributes:
            if self.use_numba and hasattr(self, '_interp_quat_numba'):
                quats = self._interp_quat_numba(frame_lower['quaternions'], frame_upper['quaternions'], t)
            else:
                # Manual slerp
                quats = self._manual_slerp(frame_lower['quaternions'], frame_upper['quaternions'], t)
            
            # Convert back to euler
            result['rotations'] = self._quat_to_euler(quats)
        
        # Interpolate sizes
        if 'sizes' in attributes:
            if self.use_numba and hasattr(self, '_interp_size_numba'):
                result['sizes'] = self._interp_size_numba(frame_lower['sizes'], frame_upper['sizes'], t)
            else:
                result['sizes'] = (frame_lower['sizes'] * (1.0 - t) + frame_upper['sizes'] * t)
        
        # Interpolate states (nearest neighbor for integer states)
        if 'states' in attributes and 'states' in frame_lower and 'states' in frame_upper:
            # Use nearest frame's states
            if t < 0.5:
                result['states'] = frame_lower['states'].copy()
            else:
                result['states'] = frame_upper['states'].copy()
        
        return result
    
    def _manual_slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Manual vectorized slerp implementation."""
        # Ensure quaternions are normalized
        q0_norm = q0 / (np.linalg.norm(q0, axis=1, keepdims=True) + 1e-10)
        q1_norm = q1 / (np.linalg.norm(q1, axis=1, keepdims=True) + 1e-10)
        
        # Calculate dot product
        dot = np.sum(q0_norm * q1_norm, axis=1)
        
        # Handle negative dot product
        neg_mask = dot < 0
        q1_norm[neg_mask] = -q1_norm[neg_mask]
        dot[neg_mask] = -dot[neg_mask]
        
        # Clamp dot to valid range
        dot = np.clip(dot, -1.0, 1.0)
        
        # Calculate interpolation
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        
        # Handle near-zero angle case
        near_zero = sin_theta_0 < 1e-6
        
        # Standard slerp
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        
        # Avoid division by zero
        s0 = np.zeros_like(dot)
        s1 = np.zeros_like(dot)
        
        valid = ~near_zero
        s0[valid] = np.cos(theta[valid]) - dot[valid] * sin_theta[valid] / sin_theta_0[valid]
        s1[valid] = sin_theta[valid] / sin_theta_0[valid]
        
        # Handle near-zero case
        s0[near_zero] = 1.0 - t
        s1[near_zero] = t
        
        # Interpolate
        result = s0[:, np.newaxis] * q0_norm + s1[:, np.newaxis] * q1_norm
        
        # Normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = result / norms
        
        return result
    
    def interpolate_batch(self, sample_indices: np.ndarray, attributes: List[str] = None) -> List[Dict[str, np.ndarray]]:
        """
        Interpolate particle data for multiple sample indices efficiently.
        
        Args:
            sample_indices: Array of sample indices (can be fractional)
            attributes: List of attributes to interpolate
            
        Returns:
            List of dictionaries with interpolated particle data
        """
        results = []
        for sample_idx in sample_indices:
            result = self.interpolate(sample_idx, attributes)
            results.append(result)
        return results
    
    def get_particle_count(self) -> int:
        """Get the number of particles in the system."""
        return self._particle_count
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Get the valid frame range."""
        if len(self._frames) == 1:
            return (self._frames[0], self._frames[0])
        return (self._frames[0], self._frames[-1])
    
    def get_sample_range(self) -> Tuple[int, int]:
        """Get the valid sample range."""
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = (fps / fps_base) * subframes
        
        frame_range = self.get_frame_range()
        start_sample = int(frame_range[0] * sample_rate / sfps)
        end_sample = int(frame_range[1] * sample_rate / sfps)
        return (start_sample, end_sample)
    
    def clear_cache(self):
        """Clear the frame cache to free memory."""
        self._frame_cache.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()
