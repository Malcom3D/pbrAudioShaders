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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import signal
from scipy.interpolate import RegularGridInterpolator

from pbrAudioCommon import EntityManager
from pbrAudioCommon import ShapeType, ShapeProperties
from pbrAudioCommon import PrimitiveGeometry
from pbrAudioCommon import _compute_rayleigh_damping, _load_mesh

@dataclass
class ProxyIRTable:
    """
    Precomputed Impulse Response table for proxy meshes.
    
    The table is indexed by:
    - Size scale (normalized 0-1)
    - Contact type (impact, sliding, scraping, rolling)
    - Frequency band
    
    Uses SIMD-optimized numpy operations for fast convolution.
    """
    entity_manager: EntityManager
    
    # IR parameters
    sample_rate: int = 48000
    max_ir_length: int = 8192  # Maximum IR length in samples
    n_frequency_bands: int = 8  # Number of frequency bands for equalization
    
    # Table parameters
    n_size_steps: int = 16  # Number of size interpolation steps
    min_size: float = 0.01  # Minimum proxy size (m)
    max_size: float = 1.0   # Maximum proxy size (m)
    
    # Internal state
    ir_table: np.ndarray = None  # Shape: (n_size_steps, n_contact_types, n_freq_bands, max_ir_length)
    size_steps: np.ndarray = None  # Size values for interpolation
    freq_bands: np.ndarray = None  # Frequency band edges
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = config.system.sample_rate
        
        # Compute frequency bands (logarithmic spacing)
        nyquist = self.sample_rate / 2
        self.freq_bands = np.logspace(np.log10(20), np.log10(nyquist), self.n_frequency_bands + 1)
        
        # Initialize size steps
        self.size_steps = np.linspace(0, 1, self.n_size_steps)
        
        # Initialize IR table
#        event_types = 6 # no-contact, impact, sliding, scraping, rolling, static
        event_types = 4 # impact, sliding, scraping, rolling
        self.ir_table = np.zeros((self.n_size_steps, event_types, self.n_frequency_bands, self.max_ir_length), dtype=np.float32)
    
    def compute_ir_table(self, proxy_meshes: List[Any]) -> None:
        """
        Compute the IR table from proxy meshes.
        
        Parameters:
        -----------
        proxy_meshes : List of proxy mesh configurations
        """
        # Group meshes by acoustic material
        material_groups = self._group_by_material(proxy_meshes)
        
        for material_key, meshes in material_groups.items():
            # Compute size range for this material
            sizes = [self._compute_mesh_size(mesh) for mesh in meshes]
            self.min_size = min(sizes)
            self.max_size = max(sizes)
            
            # Get material properties
            material_props = self._get_material_properties(meshes[0])
            
            # Compute IRs for each size step
            for size_idx, size_scale in enumerate(self.size_steps):
                size = self.min_size + size_scale * (self.max_size - self.min_size)
                
                # Compute modal parameters for this size
                modal_params = self._compute_modal_params(material_props=material_props, size=size, proxy_type=meshes[0].proxy_type)
                
                # Generate IRs for each contact type
                for contact_type in range(4):
                    ir = self._generate_ir(modal_params=modal_params, contact_type=contact_type, size=size)
                    debug_print('generated ir for contact_type:', contact_type, 'ir:', ir.shape, np.count_nonzero(ir)) 
                    
                    # Split into frequency bands
                    banded_ir = self._split_into_frequency_bands(ir)
                    
                    # Store in table
                    self.ir_table[size_idx, contact_type] = banded_ir
    
    def _group_by_material(self, proxy_meshes: List[Any]) -> Dict[Tuple, List[Any]]:
        """Group proxy meshes by their acoustic material properties."""
        groups = {}
        for mesh in proxy_meshes:
            if mesh.acoustic_shader:
                key = (mesh.acoustic_shader.young_modulus, mesh.acoustic_shader.poisson_ratio, mesh.acoustic_shader.density, mesh.acoustic_shader.damping)
            else:
                key = (None, None, None, None)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(mesh)
        return groups
    
    def _compute_mesh_size(self, mesh: Any) -> float:
        """Compute characteristic size of a mesh."""
        # Use bounding box diagonal as size measure
        if hasattr(mesh, 'obj_path'):
            # Load mesh to compute size
            vertices, _, _ = _load_mesh(mesh, 0, use_proxy_path=True)
            if len(vertices) > 0:
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                return float(np.linalg.norm(max_coords - min_coords))
        return 0.1  # Default size
    
    def _get_material_properties(self, mesh: Any) -> Dict[str, float]:
        """Extract material properties from mesh config."""
        if mesh.acoustic_shader:
            return {
                'young_modulus': mesh.acoustic_shader.young_modulus or 1e9,
                'poisson_ratio': mesh.acoustic_shader.poisson_ratio or 0.3,
                'density': mesh.acoustic_shader.density or 1000.0,
                'damping': mesh.acoustic_shader.damping or 0.02
            }
        return {
            'young_modulus': 1e9,
            'poisson_ratio': 0.3,
            'density': 1000.0,
            'damping': 0.02
        }
    
    def _compute_modal_params(self, material_props: Dict[str, float], size: float, proxy_type: int) -> Dict[str, np.ndarray]:
        """
        Compute approximate modal parameters for a proxy shape.
        
        Uses analytical solutions for platonic solids.
        """
        # Wave speeds
        E = material_props['young_modulus']
        nu = material_props['poisson_ratio']
        rho = material_props['density']
        
        c_long = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))
        c_shear = np.sqrt(E / (2 * rho * (1 + nu)))
        
        # Number of modes based on proxy type
        n_modes = proxy_type + 2  # 2, 3, 4 modes for pyramid, octahedron, cube
        
        # Generate mode frequencies based on shape
        if proxy_type == 0:  # Pyramid
            frequencies = self._pyramid_frequencies(c_shear, size, n_modes)
        elif proxy_type == 1:  # Octahedron
            frequencies = self._octahedron_frequencies(c_shear, size, n_modes)
        else:  # Cube
            frequencies = self._cube_frequencies(c_long, size, n_modes)
        
        # Compute T60 values
        damping = material_props['damping']
        t60s = 3 * np.log(10) / (np.pi * damping * np.maximum(frequencies, 1))
        t60s = np.clip(t60s, 0.001, 10.0)
        
        # Compute gains (simplified - uniform for now)
        gains = np.ones(n_modes) * 0.1
        
        return {
            'frequencies': frequencies,
            't60s': t60s,
            'gains': gains
        }
    
    def _pyramid_frequencies(self, c_shear: float, size: float, n_modes: int) -> np.ndarray:
        """Compute mode frequencies for pyramid."""
        # Mode families for tetrahedral symmetry
        factors = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
        frequencies = factors[:n_modes] * c_shear / (2 * np.pi * size / 2)
        return frequencies
    
    def _octahedron_frequencies(self, c_shear: float, size: float, n_modes: int) -> np.ndarray:
        """Compute mode frequencies for octahedron."""
        factors = np.array([1.0, 1.5, 2.0, 2.4, 2.8, 3.2])
        frequencies = factors[:n_modes] * c_shear / (2 * np.pi * size / 2)
        return frequencies
    
    def _cube_frequencies(self, c_long: float, size: float, n_modes: int) -> np.ndarray:
        """Compute mode frequencies for cube."""
        side = size / np.sqrt(3)  # Approximate side length
        # Standing wave modes
        modes = np.array([
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1)
        ])
        frequencies = np.zeros(n_modes)
        for i in range(min(n_modes, len(modes))):
            m = modes[i]
            frequencies[i] = c_long / 2 * np.sqrt(
                (m[0]/side)**2 + (m[1]/side)**2 + (m[2]/side)**2
            )
        return frequencies
    
    def _generate_ir(self, modal_params: Dict[str, np.ndarray], contact_type: int, size: float) -> np.ndarray:
        """
        Generate impulse response for a specific contact type.
        
        Uses SIMD-optimized modal synthesis.
        """
        frequencies = modal_params['frequencies']
        t60s = modal_params['t60s']
        gains = modal_params['gains']
        
        # Determine IR length based on max T60
        max_t60 = np.max(t60s)
        ir_length = min(int(max_t60 * self.sample_rate * 2), self.max_ir_length)
        
        # Generate time axis
        t = np.arange(ir_length) / self.sample_rate
        
        # Initialize IR
        ir = np.zeros(ir_length, dtype=np.float32)
        
        # Contact type specific excitation
        if contact_type == 0:  # Impact - impulse excitation
            excitation = np.zeros(ir_length)
            excitation[0] = 1.0
        elif contact_type == 1:  # Sliding - filtered noise
            excitation = self._generate_noise_excitation(ir_length, 'lowpass')
        elif contact_type == 2:  # Scraping - bandpass noise
            excitation = self._generate_noise_excitation(ir_length, 'bandpass')
        else:  # Rolling - periodic pulses
            excitation = self._generate_rolling_excitation(ir_length, size)
        
        # Vectorized modal synthesis
        # For each mode, compute damped sinusoid and add to IR
        omega = 2 * np.pi * frequencies
        decay = np.log(10) * 3 / t60s  # Exponential decay rate
        
        # Create mode matrix: (n_modes, ir_length)
        mode_matrix = np.zeros((len(frequencies), ir_length), dtype=np.float32)
        
        for i in range(len(frequencies)):
            # Damped sinusoid
            mode_matrix[i] = gains[i] * np.exp(-decay[i] * t) * np.sin(omega[i] * t)
        
        # Convolve with excitation (using FFT for speed)
        excitation_fft = np.fft.rfft(excitation, n=ir_length * 2)
        mode_fft = np.fft.rfft(mode_matrix, n=ir_length * 2, axis=1)
        
        # Sum all modes
        combined_fft = np.sum(mode_fft, axis=0)
        
        # Multiply with excitation
        result_fft = excitation_fft * combined_fft
        
        # Inverse FFT
        ir = np.fft.irfft(result_fft, n=ir_length * 2)[:ir_length]
        
        # Normalize
        max_val = np.max(np.abs(ir))
        if max_val > 0:
            ir = ir / max_val * 0.9
        
        return ir.astype(np.float32)
    
    def _generate_noise_excitation(self, length: int, filter_type: str) -> np.ndarray:
        """Generate noise excitation with filtering."""
        noise = np.random.randn(length)
        
        # Design filter
        nyquist = self.sample_rate / 2
        if filter_type == 'lowpass':
            cutoff = 2000 / nyquist
            b, a = signal.butter(4, cutoff, btype='low')
        else:  # bandpass
            low = 500 / nyquist
            high = 5000 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = signal.lfilter(b, a, noise)
        
        # Normalize
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        
        return filtered
    
    def _generate_rolling_excitation(self, length: int, size: float) -> np.ndarray:
        """Generate rolling excitation with periodic pulses."""
        # Pulse rate based on size (smaller = faster rolling)
        pulse_rate = 10.0 / size
        
        # Generate pulse train
        t = np.arange(length) / self.sample_rate
        pulse_phase = np.mod(t * pulse_rate, 1.0)
        
        # Gaussian pulses
        pulse_width = 0.01  # 10ms pulses
        excitation = np.exp(-((pulse_phase - 0.5) / pulse_width)**2)
        
        # Add some noise
        excitation += 0.1 * np.random.randn(length)
        
        # Normalize
        max_val = np.max(np.abs(excitation))
        if max_val > 0:
            excitation = excitation / max_val
        
        return excitation
    
    def _split_into_frequency_bands(self, ir: np.ndarray) -> np.ndarray:
        """
        Split IR into frequency bands using FFT.
        
        Returns:
            Array of shape (n_frequency_bands, ir_length)
        """
        n_bands = self.n_frequency_bands
        ir_length = len(ir)
        
        # FFT of IR
        ir_fft = np.fft.rfft(ir, n=ir_length * 2)
        freqs = np.fft.rfftfreq(ir_length * 2, 1/self.sample_rate)
        
        # Initialize banded IR
        banded_ir = np.zeros((n_bands, self.max_ir_length), dtype=np.float32)
        
        # For each band, apply bandpass filter in frequency domain
        for band_idx in range(n_bands):
            low_freq = self.freq_bands[band_idx]
            high_freq = self.freq_bands[band_idx + 1]
            
            # Create frequency mask
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Apply mask
            band_fft = ir_fft * mask
            
            # Inverse FFT
            band_ir = np.fft.irfft(band_fft, n=self.max_ir_length)
            
            banded_ir[band_idx] = band_ir
        
        return banded_ir
    
    def get_ir(self, size_scale: float, contact_type: int) -> np.ndarray:
        """
        Get interpolated IR for a given size scale and contact type.
        
        Parameters:
        -----------
        size_scale : float
            Normalized size (0-1)
        contact_type : int
            Contact type (0=impact, 1=sliding, 2=scraping, 3=rolling)
        
        Returns:
        --------
        np.ndarray : Interpolated IR (n_frequency_bands, ir_length)
        """
        # Clamp size scale
        size_scale = np.clip(size_scale, 0, 1)
        
        # Find interpolation indices
        size_idx = size_scale * (self.n_size_steps - 1)
        idx_low = int(np.floor(size_idx))
        idx_high = min(idx_low + 1, self.n_size_steps - 1)
        frac = size_idx - idx_low
        
        # Linear interpolation between size steps
        ir_low = self.ir_table[idx_low, contact_type]
        ir_high = self.ir_table[idx_high, contact_type]
        
        # Vectorized interpolation
        ir = ir_low * (1 - frac) + ir_high * frac
        
        return ir
    
    def save(self, filepath: str) -> None:
        """Save IR table to file."""
        np.savez_compressed(filepath, ir_table=self.ir_table, size_steps=self.size_steps, freq_bands=self.freq_bands, sample_rate=self.sample_rate, min_size=self.min_size, max_size=self.max_size)
    
    def load(self, filepath: str) -> None:
        """Load IR table from file."""
        data = np.load(filepath)
        self.ir_table = data['ir_table']
        self.size_steps = data['size_steps']
        self.freq_bands = data['freq_bands']
        self.sample_rate = data['sample_rate']
        self.min_size = data['min_size']
        self.max_size = data['max_size']
        self.n_size_steps = len(self.size_steps)
        self.n_frequency_bands = len(self.freq_bands) - 1

