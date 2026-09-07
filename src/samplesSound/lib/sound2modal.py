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
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert
import soundfile as sf
from scipy.interpolate import RegularGridInterpolator

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _generate_lib, _generate_stochastic_lib
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from pbrAudioCommon import ShapeType, ShapeProperties, PrimitiveGeometry

@dataclass
class ModalParameters:
    """Container for extracted modal parameters"""
    frequencies: np.ndarray  # Modal frequencies (Hz)
    damping_ratios: np.ndarray  # Damping ratios
    t60s: np.ndarray  # T60 reverberation times (seconds)
    gains: np.ndarray  # Modal gains (n_modes, n_vertices)
    quality_factors: np.ndarray  # Q factors
    mode_shapes: np.ndarray = None  # Optional mode shapes (n_vertices, n_modes)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StochasticResidual:
    """Container for stochastic modal residuals"""
    residual_spectrum: np.ndarray  # Residual spectrum after modal extraction
    residual_gains: np.ndarray  # Position-specific residual gains
    stochastic_seed: int  # Seed for reproducibility
    confidence: float  # Confidence in modal extraction (0-1)

class Sound2Modal:
    """
    Extract modal parameters from audio samples for modal synthesis.
    
    This class performs modal analysis on audio signals to extract:
    - Modal frequencies
    - Damping factors (T60, Q factors)
    - Position-specific gains using shape-aware stochastic fuzzy logic
    - Stochastic modal residuals for compensation
    
    The extracted parameters are formatted as Faust .lib files compatible
    with the existing modal synthesis pipeline.
    """
    
    def __init__(self, entity_manager: EntityManager):
        """
        Initialize Sound2Modal.
        
        Parameters:
        -----------
        entity_manager : EntityManager
            Entity manager instance for accessing configuration
        """
        self.entity_manager = entity_manager
        self.config = entity_manager.get('config')
        
        set_debug(self.config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        # Initialize primitive geometry classifier for shape-aware gains
        self.primitive_geometry = PrimitiveGeometry()
        
        # Analysis parameters
        self.fft_size = 4096
        self.hop_size = 1024
        self.min_freq = self.config.system.lowest_frequency
        self.max_freq = self.config.system.higher_frequency
        self.n_modes = self.config.system.modal_modes
        self.sample_rate = self.config.system.sample_rate
        
        # Residual compensation parameters
        self.residual_fft_size = 2048
        self.residual_hop_size = 512
        self.residual_threshold_db = -40.0  # Threshold for residual detection
        
    def compute(self, audio_file: str, obj_idx: int, vertices: np.ndarray = None, 
                faces: np.ndarray = None, output_name: str = None) -> ModalParameters:
        """
        Extract modal parameters from an audio file.
        
        Parameters:
        -----------
        audio_file : str
            Path to audio file (WAV, FLAC, etc.)
        obj_idx : int
            Object index for configuration lookup
        vertices : np.ndarray, optional
            Vertex positions (N, 3) for position-specific gains
        faces : np.ndarray, optional
            Face indices (M, 3) for shape classification
        output_name : str, optional
            Name for the output (defaults to object name)
            
        Returns:
        --------
        ModalParameters
            Extracted modal parameters
        """
        # Load audio file
        audio_data, sr = self._load_audio(audio_file)
        
        # Resample if necessary
        if sr != self.sample_rate:
            from resampy import resample
            audio_data = resample(audio_data, sr, self.sample_rate)
        
        # Get object configuration
        config_obj = None
        for obj in self.config.objects:
            if obj.idx == obj_idx:
                config_obj = obj
                break
        
        if config_obj is None:
            raise ValueError(f"Object {obj_idx} not found in configuration")
        
        # Determine output name
        if output_name is None:
            output_name = config_obj.name
        
        # Extract modal parameters from audio
        modal_params = self._extract_modal_parameters(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            n_modes=self.n_modes,
            min_freq=self.min_freq,
            max_freq=self.max_freq
        )
        
        # Compute stochastic residuals
        residual = self._compute_stochastic_residuals(
            audio_data=audio_data,
            frequencies=modal_params['frequencies'],
            t60s=modal_params['t60s'],
            sample_rate=self.sample_rate
        )
        
        # Generate position-specific gains if vertices are provided
        if vertices is not None and len(vertices) > 0:
            gains = self._compute_position_gains(
                vertices=vertices,
                faces=faces,
                frequencies=modal_params['frequencies'],
                config_obj=config_obj,
                residual=residual
            )
            modal_params['gains'] = gains
        else:
            # Use uniform gains if no vertices provided
            modal_params['gains'] = np.ones((self.n_modes, 1))
        
        # Add residual information to metadata
        modal_params['metadata'] = {
            'source': 'Sound2Modal',
            'audio_file': os.path.basename(audio_file),
            'sample_rate': self.sample_rate,
            'n_vertices': vertices.shape[0] if vertices is not None else 1,
            'n_modes': len(modal_params['frequencies']),
            'residual_energy': float(np.sum(residual.residual_spectrum**2)),
            'residual_confidence': float(residual.confidence),
            'stochastic_seed': residual.stochastic_seed
        }
        
        return ModalParameters(**modal_params)
    
    def _load_audio(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono.
        
        Returns:
        --------
        Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sr = sf.read(audio_file)
        except Exception as e:
            raise FileNotFoundError(f"Could not read audio file {audio_file}: {e}")
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            if audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
                debug_print(f"Warning: Multi-channel audio converted to mono")
        
        return audio_data, sr
    
    def _extract_modal_parameters(self, audio_data: np.ndarray, sample_rate: int,
                                  n_modes: int, min_freq: float, max_freq: float) -> Dict[str, np.ndarray]:
        """
        Extract modal frequencies, damping, and T60 from audio signal.
        
        Uses ESPRIT-like method with adaptive modal selection.
        """
        # Ensure audio is float32
        audio_data = audio_data.astype(np.float32)
        
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Find peaks in spectrum for initial mode detection
        frequencies, spectrum = self._compute_spectrum(audio_data, sample_rate)
        
        # Find peaks in the frequency range of interest
        mask = (frequencies >= min_freq) & (frequenciesencies <= max_freq)
        peak_freqs, peak_props = find_peaks(
            spectrum[mask],
            height=np.max(spectrum[mask]) * 0.01,
            distance=5
        )
        
        # Convert to actual frequencies
        peak_frequencies = frequencies[mask][peak_freqs]
        
        # Sort by magnitude and take top n_modes
        peak_magnitudes = spectrum[mask][peak_freqs]
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        
        # Limit to n_modes
        n_peaks = min(len(sorted_idx), n_modes)
        selected_freqs = peak_frequencies[sorted_idx[:n_peaks]]
        
        if len(selected_freqs) == 0:
            debug_print("No peaks found in frequency range")
            return {
                'frequencies': np.array([]),
                'damping_ratios': np.array([]),
                't60s': np.array([]),
                'gains': np.array([])
            }
        
        # Extract damping and T60 for each peak
        frequencies = []
        damping_ratios = []
        t60s = []
        
        for freq in selected_freqs:
            # Find the exact frequency and damping using local analysis
            freq_exact, damping, t60 = self._extract_single_mode(
                audio_data=audio_data,
                sample_rate_rate=sample_rate,
                center_freq=freq,
                bandwidth=max(freq * 0.1, 10.0)  # ±10% bandwidth
            )
            
            if freq_exact is not None:
                frequencies.append(freq_exact)
                damping_ratios.append(damping)
                t60s.append(t60)
        
        if len(frequencies) == 0:
            debug_print("No valid modes extracted")
            return {
                'frequencies': np.array([]),
                'damping_ratios': np.array([]),
                't60s': np.array([]),
                'gains': np.array([])
            }
        
        # Convert to arrays
        frequencies = np.array(frequencies)
        damping_ratios = np.array(damping_ratios)
        t60s = np.array(t60s)
        
        # Sort by frequency
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx]
        damping_ratios = damping_ratios[sort_idx]
        t60s = t60s[sort_idx]
        
        # Calculate Q factors
        quality_factors = 1.0 / (2.0 * damping_ratios)
        
        # Ensure we have exactly n_modes (pad with synthetic modes if needed)
        if len(frequencies) < n_modes:
            frequencies, damping_ratios, t60s = self._pad_modes(
                frequencies, damping_ratios, t60s, n_modes, min_freq, max_freq
            )
        
        return {
            'frequencies': frequencies,
            'damping_ratios': damping_ratios,
            't60s': t60s,
            'quality_factors': quality_factors
        }
    
    def _compute_spectrum(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum using Welch's method.
        """
        frequencies, spectrum = signal.welch(
            audio_data,
                       fs=sample_rate,
            nperseg=self.fft_size,
            noverlap=self.hop_size,
            window='hann',
            scaling='spectrum'
        )
        
        # Convert to dB for peak detection
        spectrum_db = 20 * np.log10(spectrum + 1e-12)
        
        return frequencies, spectrum_db
    
    def _extract_single_mode(self, audio_data: np.ndarray, sample_rate: int,
                            center_freq: float, bandwidth: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Extract a single mode's parameters using bandpass filtering and envelope analysis.
        """
        # Design bandpass filter around the center frequency
        nyquist = sample_rate / 2
        low_freq = max(center_freq - bandwidth/2, 1.0)
        high_freq = min(center_freq + bandwidth/2, nyquist - 1)
        
        if low_freq >= high_freq:
            return None, None, None
        
        # Apply bandpass filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band', fs=sample_rate)
        filtered = signal.lfilter(b, a, audio_data)
        
        # Get envelope using Hilbert transform
        analytic_signal = hilbert(filtered)
        envelope = np.abs(analytic_signal)
        
        # Find the peak of the envelope
        peak_idx = np.argmax(envelope)
        
        # Check if the peak is significant
        if envelope[peak_idx] < 0.001 * np.max(envelope):
            return None, None, None
        
        # Extract decay portion after peak
        decay_start = peak_idx
        decay_end = min(len(envelope), peak_idx + int(sample_rate * 2))  # 2 seconds max
        
        if decay_end - decay_start < 100:  # Need at least 100 samples
            return None, None, None
        
        decay_envelope = envelope[decay_start:decay_end]
        time_axis = np.arange(len(decay_envelope)) / sample_rate
        
        # Fit exponential decay: y = A * exp(-t/tau)
        try:
            # Initial guess
            A0 = decay_envelope[0]
            tau0 = 0.1  # 100ms initial guess
            
            popt, _ = curve_fit(
                lambda t, A, tau: A * np.exp(-t/tau),
                time_axis,
                decay_envelope,
                p0=[A0, tau0],
                bounds=([0, 0.001], [np.max(decay_envelope)*10, 10.0])
            )
            
            A, tau = popt
            
            if A <= 0 or tau <= 0:
                return None, None, None
            
            # Calculate exact frequency from the filtered signal
            # Use zero-crossing detection
            crossings = np.where(np.diff(np.signbit(filtered[peak_idx:decay_end])))[0]
            if len(crossings) > 2:
                # Estimate frequency from zero crossings
                time_crossings = (peak_idx + crossings) / sample_rate
                freq_estimate = (len(crossings) - 1) / (2 * (time_crossings[-1] - time_crossings[0]))
            else:
                freq_estimate = center_freq
            
            # Calculate damping ratio
            # For exponential decay: x(t) = A * exp(-ζ*ω_n*t) * sin(ω_d*t)
            # ζ = 1/(ωω_n * tau)
            omega_n = 2 * np.pi * freq_estimate
            damping_ratio = 1.0 / (omega_n * tau)
            
            # T60 = 6.9078 * tau
            t60 = 6.9078 * tau
            
            # Validate results
            if damping_ratio < 0 or damping_ratio > 1:
                return None, None, None
            
            return freq_estimate, damping_ratio, t60
            
        except Exception as e:
            debug_print(f"Error fitting mode at {center_freq} Hz: {e}")
            return None, None, None
    
    def _pad_modes(self, frequencies: np.ndarray, damping_ratios: np.ndarray,
                  t60s: np.ndarray, n_modes: int, min_freq: float, max_freq: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pad modal parameters with synthetic modes to reach n_modes.
        """
        n_existing = len(frequencies)
        n_to_add = n_modes - n_existing
        
        if n_to_add <= 0:
            return frequencies, damping_ratios, t60s
        
        # Generate synthetic modes in the frequency range
        rng = np.random.default_rng(42)
        
        # Find gaps in existing frequencies
        all_freqs = np.sort(np.concatenate([frequencies, 
                                           np.linspace(min_freq, max_freq, n_modes + 10)]))
        
        # Select frequencies for synthetic modes
        synth_freqs = np.array([])
        while len(synth_freqs) < n_to_add:
            # Pick random frequencies that don't overlap with existing modes
            candidate = rng.uniform(min_freq, max_freq)
            if not np.any(np.abs(frequencies - candidate) < 0.1 * candidate):
                synth_freqs = np.append(synth_freqs, candidate)
        
        # Generate synthetic damping and T60
        synth_damping = rng.uniform(0.01, 0.1, n_to_add)
        synth_t60 = 6.9078 / (2 * np.pi * synth_freqs * synth_damping)
        
        # Combine and sort
        frequencies = np.sort(np.concatenate([frequencies, synth_freqs]))
        damping_ratios = np.concatenate([damping_ratios, synth_damping])
        t60s = np.concatenate([t60s, synth_t60])
        
        # Sort by frequency
        sort_idx = np.argsort(frequencies)
        
        return frequencies[sort_idx], damping_ratios[sort_idx], t60s[sort_idx]
    
    def _compute_stochastic_residuals(self, audio_data: np.ndarray, 
                                     frequencies: np.ndarray,
                                     t60s: np.ndarray,
                                     sample_rate: int) -> StochasticResidual:
        """
        Compute stochastic modal residuals for compensation.
        
        The residual represents the part of the audio signal not captured
        by the extracted modes. This is used to add realistic stochastic
        variations to the modal synthesis.
        """
        # Generate synthetic modal response
        synthetic_audio = self._synthesize_modal_response(
            audio_data=audio_data,
            frequencies=frequencies,
            t60s=t60s,
            sample_rate=sample_rate
        )
        
        # Compute residual
        residual_audio = audio_data - synthetic_audio
        
        # Compute residual spectrum
        frequencies_res, residual_spectrum = signal.welch(
            residual_audio,
            fs=sample_rate,
            nperseg=self.residual_fft_size,
            noverlap=self.residual_hop_size,
            window='hann'
        )
        
        # Convert to dB
        residual_db = 20 * np.log10(residual_spectrum + 1e-12)
        
        # Determine confidence based on how much energy is in the residual
        total_energy = np.sum(audio_data**2)
        residual_energy = np.sum(residual_audio**2)
        
        if total_energy > 0:
            confidence = 1.0 - min(residual_energy / total_energy, 1.0)
        else:
            confidence = 0.0
        
        # Generate stochastic seed for reproducibility
        rng = np.random.default_rng()
        stochastic_seed = rng.integers(0, 2**32 - 1)
        
        return StochasticResidual(
            residual_spectrum=residual_spectrum,
            residual_gains=np.abs(residual_spectrum),  # Use magnitude as gains
            stochastic_seed=int(stochastic_seed),
            confidence=float(confidence)
        )
    
    def _synthesize_modal_response(self, audio_data: np.ndarray,
                                  frequencies: np.ndarray,
                                  t60s: np.ndarray,
                                  sample_rate: int) -> np.ndarray:
        """
        Synthesize modal response from extracted parameters.
        """
        n_samples = len(audio_data)
        output = np.zeros(n_samples)
        
        # For each mode, generate a decaying sinusoid
        for i, freq in enumerate(frequencies):
            if i >= len(t60s):
                break
            
            t60 = t60s[i]
            
            # Generate decaying sinusoid
            t = np.arange(n_samples) / sample_rate
            decay = np.exp(-6.9078 * t / max(t60, 0.001))
            sinusoid = np.sin(2 * np.pi * freq * t)
            
            # Estimate amplitude from audio data
            # Use correlation with the sinusoid
            amplitude = np.dot(audio_data, sinusoid) / (np.dot(sinusoid, sinusoid) + 1e-10)
            
            output += amplitude * decay * sinusoid
        
        return output
    
    def _compute_position_gains(self, vertices: np.ndarray, faces: np.ndarray,
                               frequencies: np.ndarray, config_obj,
                               residual: StochasticResidual = None) -> np.ndarray:
        """
        Compute position-specific gains using shape-aware stochastic fuzzy logic.
        
        This method:
        1. Classifies the mesh shape
        2. Computes shape-specific gain patterns
        3. Applies stochastic variations based on confidence and residuals
        """
        n_vertices = vertices.shape[0]
        n_modes = len(frequencies)
        
        # Initialize gains matrix
        gains = np.zeros((n_modes, n_vertices))
        
        # Classify shape
        if faces is not None and len(faces) > 0:
            shape_props = self.primitive_geometry.classify(vertices, faces)
        else:
            # Create dummy shape properties for point cloud
            shape_props = self._create_dummy_shape_properties(vertices)
        
        # Get shape-specific parameters
        shape_type = shape_props.shape_type
        confidence = shape_props.confidence
        centroid = shape_props.centroid
        bbox = shape_props.bounding_box
        
        # Compute vertex features
        # Normalized position in bounding box (0..1)
        bbox_min = bbox.min(axis=0)
        bbox_max = bbox.max(axis=0)
        range_ = bbox_max - bbox_min
        range_[range_ == 0] = 1.0  # Avoid division by zero
        norm_pos = (vertices - bbox_min) / range_
        
        # Distance from centroid (normalized)
        dist_from_centroid = np.linalg.norm(vertices - centroid, axis=1)
        max_dist = np.max(dist_from_centroid) if np.max(dist_from_centroid) > 0 else 1.0
        norm_dist = dist_from_centroid / max_dist
        
        # Get material properties for stochastic variation
        young_modulus = config_obj.acoustic_shader.young_modulus if config_obj.acoustic_shader else 1e9
        density = config_obj.acoustic_shader.density if config_obj.acoustic_shader else 1000.0
        
        # Compute shape-specific gain patterns
        if shape_type == ShapeType.SPHERE:
            # Uniform gains with slight variation
            shape_scale = 1.0 + 0.1 * np.sin(np.pi * norm_dist)
        elif shape_type == ShapeType.PLATE:
            # Higher gains at edges
            edge_factor = np.minimum(norm_pos, 1 - norm_pos).min(axis=1)
            shape_scale = 1.0 + 0.5 * (1.0 - 2.0 * edge_factor)
        elif shape_type == ShapeType.BEAM:
            # Higher gains at ends
            extents = bbox_max - bbox_min
            longest_axis = np.argmax(extents)
            axis_pos = norm_pos[:, longest_axis]
            shape_scale = 1.0 + 0.4 * np.sin(np.pi * axis_pos)
        elif shape_type == ShapeType.CUBE:
            # Higher gains at corners
            corner_dist = np.max(np.abs(norm_pos - 0.5), axis=1)
            shape_scale = 1.0 + 0.3 * corner_dist
        elif shape_type == ShapeType.CYLINDER:
            # Higher gains at ends
            shape_scale = 1.0 + 0.3 * np.sin(np.pi * norm_pos[:, 2])
        elif shape_type == ShapeType.CONE:
            # Higher gains at base
            shape_scale = 1.0 + 0.3 * (1.0 - norm_pos[:, 2])
        else:
            # Irregular: use distance from centroid
            shape_scale = 1.0 + 0.2 * (1.0 - norm_dist)
        
        # Apply stochastic variations based on confidence and residuals
        rng = np.random.default_rng(seed=residual.stochastic_seed if residual else 42)
        noise_amplitude = 1.0 - confidence
        
        # If residual is available, use its spectrum for frequency-dependent variation
        if residual is not None:
            residual_spectrum = residual.residual_spectspectrum
            # Normalize residual spectrum
            if np.max(residual_spectrum) > 0:
                residual_norm = residual_spectrum / np.max(residual_spectrum)
            else:
                residual_norm = np.ones_like(residual_spectrum)
        else:
            residual_norm = np.ones(n_modes)
        
        for mode_idx in range(n_modes):
            # Mode-specific frequency-dependent variation
            freq = frequencies[mode_idx]
            
            # Material-dependent variation
            material_factor = np.sqrt(young_modulus / 1e9) * (density / 1000.0)
            
            # Frequency-dependent spatial pattern
            wavelength_factor = 1.0 / (1.0 + freq / 10000.0)
            
            # Apply residual-based variation
            residual_factor = 1.0 + 0.2 * residual_norm[mode_idx % len(residual_norm)]
            
            # Combine shape scale with frequency-dependent factor
            mode_phase = rng.uniform(0, 2 * np.pi)
            mode_factor = 1.0 + 0.2 * np.sin(2 * np.pi * norm_dist * wavelength_factor + mode_phase)
            
            # Apply to gains
            gains[mode_idx, :] = shape_scale * mode_factor * material_factor * residual_factor
            
            # Add stochastic noise
            noise = rng.normal(0, noise_amplitude * np.abs(gains[mode_idx, :]))
            gains[mode_idx, :] += noise
        
        # Normalize gains
        max_gain = np.max(np.abs(gains))
        if max_gain > 0:
            gains /= max_gain
        
        # Ensure minimum gain for all vertices
        gains = np.maximum(gains, 1e-6)
        
        return gains.T  # Return (n_vertices, n_modes) for compatibility
    
    def _create_dummy_shape_properties(self, vertices: np.ndarray) -> ShapeProperties:
        """
        Create dummy shape properties for point clouds without mesh data.
        """
        centroid = np.mean(vertices, axis=0)
        bbox = np.array([
            [np.min(vertices[:, 0]), np.min(vertices[:, 1]), np.min(vertices[:, 2])],
            [np.max(vertices[:, 0]), np.min(vertices[:, 1]), np.min(vertices[:, 2])],
            [np.min(vertices[:, 0]), np.max(vertices[:, 1]), np.min(vertices[:, 2])],
            [np.max(vertices[:, 0]), np.max(vertices[:, 1]), np.min(vertices[:, 2])],
            [np.min(vertices[:, 0]), np.min(vertices[:, 1]), np.max(vertices[:, 2])],
            [np.max(vertices[:, 0]), np.min(vertices[:, 1]), np.max(vertices[:, 2])],
            [np.min(vertices[:, 0]), np.max(vertices[:, 1]), np.max(vertices[:, 2])],
            [np.max(vertices[:, 0]), np.max(vertices[:, 1]), np.max(vertices[:, 2])]
        ])
        
        return Shape ShapeProperties(
            shape_type=ShapeType.IRREGULAR,
            dimensions={'extent': np.max(bbox.max(axis=0) - bbox.min(axis=0))},
            volume=np.prod(bbox.max(axis=0) - bbox.min(axis=0)),
            surface_area=0.0,
            aspect_ratio=1.0,
            compactness=0.3,
            confidence=0.3,
            bounding_box=bbox,
            centroid=centroid
        )
    
    def to_faust_lib(self, modal_params: ModalParameters, output_name: str,
                     min_freq: float = None, max_freq: float = None) -> str:
        """
        Generate Faust .lib file content from modal parameters.
        
        Parameters:
        -----------
        modal_params : ModalParameters
            Extracted modal parameters
        output_name : str
            Name for the Faust output
        min_freq, max_freq : float
            Frequency range (defaults to system settings)
            
        Returns:
        --------
        str
            Faust .lib file content
        """
        # Set default frequency range
        if min_freq is None:
            min_freq = self.config.system.lowest_frequency
        if max_freq is None:
            max_freq = self.config.system.higher_frequency
        
        # Get modal data
        frequencies = modal_params.frequencies
        gains = modal_params.gains
        t60s = modal_params.t60s
        metadata = modal_params.metadata
        
        n_modes = len(frequencies)
        n_vertices = gains.shape[1] if gains.ndim > 1 else 1
        
        if n_modes == 0 or n_vertices == 0:
            return _generate_stochastic_lib(
                output_name=output_name,
                min_freq=min_freq,
                max_freq=max_freq,
                n_expos=n_vertices,
                n_modes=1,
                young_modulus=metadata.get('young_modulus', 1e9),
                poisson_ratio=metadata.get('poisson_ratio', 0.3),
                density=metadata.get('density', 1000.0),
                damping=metadata.get('damping', 0.02)
            )
        
        # Filter modes by frequency range
        valid_idx = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]
        if len(valid_idx) < n_modes:
            frequencies = frequencies[valid_idx]
            gains = gains[valid_idx, :] if gains.ndim > 1 else gains[valid_idx]
            t60s = t60s[valid_idx]
            n_modes = len(frequencies)
            
            if n_modes == 0:
                return _generate_stochastic_lib(
                    output_name=output_name,
                    min_freq=min_freq,
                    max_freq=max_freq,
                    n_expos=n_vertices,
                    n_modes=1,
                    young_modulus=metadata.get('young_modulus', 1e9),
                    poisson_ratio=metadata.get('poisson_ratio', 0.3),
                    density=metadata.get('density', 1000.0),
                    damping=metadata.get('damping', 0.02)
                )
        
        # Prepare frequency string
        freq_values = [f"{freq:.6f}" for freq in frequencies]
        freq_str = ", ".join(freq_values)
        
        # Prepare T60 strings
        t60_values = [f"{t:.6f}" for t in t60s]
        t60_str = ", ".join(t60_values)
        
        # Generate gains matrix as waveform
        # Each row row corresponds to a mode, each column to a vertex
        gain_lines = []
        for mode_idx in range(n_modes):
            if gains.ndim > 1:
                mode_gains = gains[mode_idx, :]
            else:
                mode_gains = np.full(n_vertices, gains[mode_idx])
])
            gain_str = ", ".join([f"{g:.6f}" for g in mode_gains])
            gain_lines.append(f"{gain_str}")
        gain_waveform = ", ".join(gain_lines)
        
        # Get material properties from metadata
        young_modulus = metadata.get('young_modulus',  1e9)
        poisson_ratio = metadata.get('poisson_ratio', 0.3)
        density = metadata.get('density', 1000.0)
        
        header = f"Sound2Modal extracted modal model for {output_name}"
        generator = self.__class__.__name__
        
        return _generate_lib(
            header=header,
            generator=generator,
            output_name=output_name,
            n_modes=n_modes,
            n_vertices=n_vertices,
            min_freq=min_freq,
            max_freq=max_freq,
            freq_str=freq_str,
            t60_str=t60_str,
            gain_waveform=gain_waveform,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            density=density
        )
    
    def save_to_file(self, modal_params: ModalParameters, output_path: str,
                     output_name: str, min_freq: float = None, max_freq: float = None) -> str:
        """
        Generate and save Faust .lib file.
        
        Returns:
        --------
        str
            Path to the saved file
        """
        lib_content = self.to_faust_lib(modal_params, output_name, min_freq, max_freq)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(lib_content)
        
        debug_print(f"Saved modal model to {output_path}")
        return output_path

