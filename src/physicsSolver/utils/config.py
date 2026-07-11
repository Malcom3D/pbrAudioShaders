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

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..lib.acoustic_shader import AcousticShader, AcousticProperties, AcousticCoefficients

@dataclass
class SystemConfig:
    sample_rate: int = 48000
    bit_depth: int = 32
    fps: int = 24 # video fps
    fps_base: int = 1
    file_format: str = 'RAW'
    subframes: int = 1 # video subframes
    modal_modes: int = 20
    collision_margin: float = 0.05
    samples_per_object: int = 1000
    cache_path: str = "./pbrAudioCache/"
    enable_denoiser: bool = False
    enable_postprocess: bool = False
    proxy_size_threshold: float = 0.1

@dataclass
class DenoiserConfig:
    # DC Offset Removal parameters
    enable_dc_blocker: bool = False
    dc_blocker_alpha: float = 0.999  # DC blocker coefficient (higher = more aggressive)
    # Adaptive Noise Gate parameters
    enable_noise_gate: bool = False
    gate_threshold_db: float = -60.0  # Noise gate threshold in dB
    gate_attack_ms: float = 2.0       # Attack time in ms
    gate_release_ms: float = 50.0     # Release time in ms
    gate_hold_ms: float = 10.0        # Hold time in ms
    # Temporal Smoothing parameters
    enable_temporal_smoothing: bool = False
    temporal_smoothing_window: int = 5  # Window size for temporal smoothing (samples)
    # Spectral Noise Reduction parameters
    enable_spectral_noise_reduction: bool = False
    spectral_fft_size: int = 2048      # FFT size for spectral processing
    spectral_hop_size: int = 512       # Hop size for spectral processing
    spectral_noise_floor_db: float = -80.0  # Noise floor estimate in dB
    spectral_reduction_strength: float = 0.8  # Reduction strength (0-1)
    spectral_smoothing: float = 0.3    # Spectral smoothing factor
    # Envelope Shaping parameters
    enable_envelope_shaping: bool = False
    envelope_attack_ms: float = 1.0    # Attack time for envelope
    envelope_release_ms: float = 20.0  # Release time for envelope
    envelope_smoothing: float = 0.5    # Envelope smoothing factor
    # Gaussian Adaptive Smoothing parameters
    enable_gaussian_adaptive_smoothing: bool = False
    gaussian_sigma_min: float = 0.5    # Minimum Gaussian sigma
    gaussian_sigma_max: float = 3.0    # Maximum Gaussian sigma
    gaussian_force_threshold: float = 0.1  # Force threshold for adaptive smoothing

@dataclass
class PostProcessConfig:
    """Configuration for post-processing parameters."""
    # Denoising parameters
    dynamic_denoise_enabled: bool = True
    noise_gate_threshold_db: float = -60.0
    noise_floor_estimate_db: float = -80.0
    spectral_reduction_strength: float = 0.7
    temporal_smoothing_window: int = 5
    # Dynamic reference weighting
    force_reference_weight: float = 0.3  # How much to trust force signals
    min_force_threshold: float = 1e-6    # Minimum force to consider active
    # Smoothing parameters
    smoothing_enabled: bool = True
    smoothing_window_ms: float = 2.0     # Window in milliseconds
    adaptive_smoothing: bool = True      # Smooth more during quiet sections
    # Phase alignment
    phase_align_enabled: bool = True
    crossfade_samples: int = 256         # Crossfade length for blending
    # Amplification
    target_rms: float = 0.15             # Target RMS level
    max_gain_db: float = 20.0            # Maximum gain in dB
    dynamic_range_compression: float = 0.5  # 0=no compression, 1=full
    # Blending
    blend_enabled: bool = True
    dry_wet_mix: float = 0.85            # 0=dry only, 1=wet only
    # Output
    normalize_output: bool = True
    # Debug
    verbose: bool = False

@dataclass
class ObjectConfig:
    idx: int
    name: str
    obj_path: str
    pose_path: str
    static: bool
    proxy: Union[bool, int] = False # 0 = octahedron, 1 = icosahedron for < proxy_size_threshold, 2,3,4 for low,mid,hi manual selection and icosahedron subdivision
    ground: bool = False
    resonance: bool = False
    resonance_modes: int = 10
    is_shard: Union[bool, int] = False
    fractured: Union[bool, int] = False
    shard: Union[bool, np.ndarray] = False # for shards of fractured object [obj_idx]
    connected: Union[bool, np.ndarray] = False # for static coupled systems [[obj_idx, coupling_strength]]
    stochastic_variation: bool = False
    acoustic_shader: Optional[AcousticShader] = None

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.data = json.load(f)

        self.system = SystemConfig(**self.data.get('system', {}))
        self.denoiser = DenoiserConfig(**self.data.get('denoiser', {}))
        self.postprocess = PostProcessConfig(**self.data.get('postprocess', {}))

        # Handle objects with nested acoustic_shader
        self.objects = []
        for o in self.data.get('objects', []):
            acoustic_shader_data = o.get('acoustic_shader', {})

            object_config = ObjectConfig(
                **{k: v for k, v in o.items() if k != 'acoustic_shader'},
                acoustic_shader=self._create_acoustic_shader(acoustic_shader_data) if acoustic_shader_data else None
            )
            self.objects.append(object_config)

    def _create_acoustic_shader(self, shader_data: Dict[str, Any]) -> AcousticShader:
        """Create AcousticShader instance from dictionary data"""
        acoustic_props_data = shader_data.get('acoustic_properties', {})

        # Create AcousticCoefficients for each property
        acoustic_properties = AcousticProperties()

        if 'absorption' in acoustic_props_data:
            abs_data = acoustic_props_data['absorption']
            if isinstance(abs_data, float):
                frequencies = np.linspace(shader_data.get('low_frequency', 1.0), shader_data.get('high_frequency', 24000.0), 5)
                abs_data = [abs_data for _ in range(5)]
                abs_data = {"frequencies": frequencies, "coefficients": abs_data, "phases": []}
            acoustic_properties.absorption = AcousticCoefficients(
                frequencies=np.array(abs_data['frequencies']),
                coefficients=np.array(abs_data['coefficients'])
            )

        if 'refraction' in acoustic_props_data:
            refr_data = acoustic_props_data['refraction']
            if isinstance(refr_data, float):
                frequencies = np.linspace(shader_data.get('low_frequency', 1.0), shader_data.get('high_frequency', 24000.0), 5)
                refr_data = [refr_data for _ in range(5)]
                refr_data = {"frequencies": frequencies, "coefficients": refr_data, "phases": []}
            acoustic_properties.refraction = AcousticCoefficients(
                frequencies=np.array(refr_data['frequencies']),
                coefficients=np.array(refr_data['coefficients'])
            )

        if 'reflection' in acoustic_props_data:
            refl_data = acoustic_props_data['reflection']
            if isinstance(refl_data, float):
                frequencies = np.linspace(shader_data.get('low_frequency', 1.0), shader_data.get('high_frequency', 24000.0), 5)
                refl_data = [refl_data for _ in range(5)]
                refl_data = {"frequencies": frequencies, "coefficients": refl_data, "phases": []}
            acoustic_properties.reflection = AcousticCoefficients(
                frequencies=np.array(refl_data['frequencies']),
                coefficients=np.array(refl_data['coefficients'])
            )

        if 'scattering' in acoustic_props_data:
            scat_data = acoustic_props_data['scattering']
            if isinstance(scat_data, float):
                frequencies = np.linspace(shader_data.get('low_frequency', 1.0), shader_data.get('high_frequency', 24000.0), 5)
                scat_data = [scat_data for _ in range(5)]
                scat_data = {"frequencies": frequencies, "coefficients": scat_data, "phases": []}
            acoustic_properties.scattering = AcousticCoefficients(
                frequencies=np.array(scat_data['frequencies']),
                coefficients=np.array(scat_data['coefficients'])
            )

        # Create AcousticShader
        return AcousticShader(
            sound_speed=shader_data.get('sound_speed', 343.0),
            young_modulus=shader_data.get('young_modulus', []),
            poisson_ratio=shader_data.get('poisson_ratio', []),
            density=shader_data.get('density', 1.225),
            damping=shader_data.get('damping', []),
            friction=shader_data.get('friction', []),
            roughness=shader_data.get('roughness', []),
            low_frequency=shader_data.get('low_frequency', 1.0),
            high_frequency=shader_data.get('high_frequency', 24000.0),
            acoustic_properties=acoustic_properties
        )
