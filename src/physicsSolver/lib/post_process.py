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
import json
import numpy as np
from typing typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine
from pbrAudioSignals import (
    ChainProcessor,
    DCBlocker,
    AdaptiveNoiseGate,
    TemporalSmoothing,
    SpectralDenoise,
    EnvelopeShaper,
    AdaptiveGaussianSmoothing,
    ForceWeightedGain,
    GlobalNormalize,
    PerTrackNormalize,
    build_chain,
)

@dataclass
class postProcess:
    """
    Non-destructive post-processing engine for synthesized audio forces.

    Instead of writing processed audio to disk, this engine writes processing
    chain metadata to the StorageEngine. The processing chains are then applied
    at read time by the DAG evaluator.

    This replaces the deprecated AudioForcePostProcessEngine that used the
    old postProcess module.
    """
    entity_manager: EntityManager
    storage_engine: StorageEngine = None

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.config = config
        self.denoiser_config = config.denoiser
        self.postprocess_config = config.postprocess

        # Initialize or reuse StorageEngine
        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

    def process(self) -> None:
        """
        Configure processing chains for all objects.
        Writes chain metadata to StorageEngine for read-time processing.
        """
        config = self.config

        debug_print("Configuring processing chains...")

        # First, compute global normalization value across all objects
        global_peak = self._compute_global_peak()
        debug_print(f"Global peak value: {global_peak}")

        # Configure chains for each object
        obj_count = 0
        for config_obj in config.objects:
            if config_obj.static:
                continue

            # Check if object has audio-force tracks
            if not self._has_audio_force_tracks(config_obj.idx):
                continue

            # Build and register processing chain
            self._configure_object_chains(config_obj, global_peak)
            obj_count += 1

        debug_print(f"Configured processing chains for {obj_count} objects")

    def _has_audio_force_tracks(self, obj_idx: int) -> bool:
        """Check if an object has audio-force tracks in storage."""
        try:
            # Try to read a track to see if it exists
            track_data = self.storage_engine.read(
                engine='physicsSolver',
                obj_idx=obj_idx,
                track_name='impact'
            )
            return track_data is not None and track_data.size > 0
        except Exception:
            return False

    def _compute_global_peak(self) -> float:
        """
        Compute the global peak value across all audio-force tracks.
        Used for global normalization.
        """
        config = self.config
        max_peak = 0.0

        track_names = ['non_collision', 'impact', 'sliding', 'scraping', 'rolling',
                       'coupling_strength']
        if config.system.enable_noise_enhancement:
            track_names += ['sliding_sound', 'scraping_sound', 'rolling_sound']

        for config_obj_obj in config.objects:
            if config_obj.static:
                continue

            for track_name in track_names:
                try:
                    track_data = self.storage_engine.read(
                        engine='physicsSolver',
                        obj_idx=config_obj.idx,
                        track_name=track_name
                    )
                    if track_data is not None and track_data.size > 0:
                        peak = float(np.max(np.abs(track_data)))
                        max_peak = max(max_peak, peak)
                except Exception:
                    continue

        return max_peak if max_peak > 0 else 1.0

    def _configure_object_chains(self, config_obj: Any, global_peak: float) -> None:
        """
        Configure processing chains for all tracks of a single object.

        Args:
            config_obj: Object configuration
            global_peak: Global peak value for normalization
        """
        config = self.config

        # Determine which tracks exist and their types
        track_types = {
            'impact': 'impact',
            'sliding': 'continuous',
            'scraping': 'continuous',
            'rolling': 'continuous',
            'non_collision': 'ambient',
            'coupling_strength': 'control',
        }

        if config.system.enable_noise_enhancement:
            track_types.update({
                'sliding_sound': 'noise',
                'scraping_sound': 'noise',
                'rolling_sound': 'noise',
            })

        for track_name, track_type in track_types.items():
            # Build chain specification for this track
            chain_spec = self._build_chain_spec(
                track_name=track_name,
                track_type=track_type,
                config_obj=config_obj,
                global_peak=global_peak
            )

            if chain_spec:
                # Write chain metadata to storage
                self._write_chain_metadata(
                    obj_idx=config_obj.idx,
                    track_name=track_name,
                    chain_spec=chain_spec
                )
                debug_print(f"Configured chain for {config_obj.name}/{track_name}: "
                           f"{len(chain_spec)} processors")

    def _build_chain_spec(self, track_name: str, track_type: str,
                          config_obj: Any, global_peak: float) -> List[Dict[str, Any]]:
        """
        Build the chain specification for a specific track.

        Args:
            track_name: Name of the track
            track_type: Type of track ('impact', 'continuous', 'noise', 'ambient', 'control')
            config_obj: Object configuration
            global_peak: Global peak value

        Returns:
            List of processor specifications
        """
        chain = []
        denoiser = self.denoiser_config
        postprocess = self.postprocess_config

        # coupling_strength is a control signal, minimal processing
        if track_type == 'control':
            return []  # No processing for control signals

        # DC blocker (always first if enabled)
        if denoiser.enable_dc_blocker or postprocess.dynamic_denoise_enabled:
            chain.append({
                'type': 'dc_blocker',
                'params': {
                    'alpha': denoiser.dc_blocker_alpha if denoiser.enable_dc_blocker else 0.999
                }
            })

        # Adaptive noise gate with force correlation
        if denoiser.enable_noise_gate or postprocess.dynamic_denoise_enabled:
            # Different thresholds for different track types
            if track_type == 'impact':
                threshold_db = -70.0  # Preserve more for impacts
                force_weight = 0.7    # Strong force correlation
            elif track_type == 'continuous':
                threshold_db = -60.0
                force_weight = 0.5
            elif track_type == 'noise':
                threshold_db = -50.0
                force_weight = 0.3
            else:  # ambient
                threshold_db = -40.0
                force_weight = 0.1

            chain.append({
                'type': 'adaptive_noise_gate',
                'params': {
                    'sample_rate': self.config.system.sample_rate,
                    'threshold_db': threshold_db,
                    'attack_ms': denoiser.gate_attack_ms,
                    'release_ms': denoiser.gate_release_ms,
                    'hold_ms': denoiser.gate_hold_ms,
                    'force_weight': force_weight
                }
            })

        # Temporal smoothing (preserves transients)
        if denoiser.enable_temporal_smoothing or postprocess.smoothing_enabled:
            # Different windows for different tracks
            if track_type == 'impact':
                window = 3  # Minimal smoothing for impacts
            elif track_type == 'continuous':
                window = denoiser.temporal_smoothing_window
            else:
                window = 5

            chain.append({
                'type': 'temporal_smoothing',
                'params': {
                    'window': window,
                    'transient_threshold': 0.15
                }
            })

        # Spectral denoising
        if denoiser.enable_spectral_noise_reduction or postprocess.dynamic_denoise_enabled:
            # Different reduction strengths for different tracks
            if track_type == 'impact':
                reduction_quiet = 0.5   # Less reduction for impacts
                reduction_loud = 0.1
            elif track_type == 'continuous':
                reduction_quiet = postprocess.spectral_reduction_strength
                reduction_loud = 0.2
            elif track_type == 'noise':
                reduction_quiet = 0.9   # More reduction for noise tracks
                reduction_loud = 0.3
            else:
                reduction_quiet = 1.0
                reduction_loud = 0.4

            chain.append({
                'type': 'spectral_denoise',
                'params': {
                    'n_fft': denoiser.spectral_fft_size,
                    'hop_length': denoiser.spectral_hop_size,
                    'reduction_strength_quiet': reduction_quiet,
                    'reduction_strength_loud': reduction_loud,
                    'noise_floor_percentile': 10.0
                }
            })

        # Envelope shaping
        if denoiser.enable_envelope_shaping:
            chain.append({
                'type': 'envelope_shaper',
                'params': {
                    'depth': 0.5 if track_type == 'continuous' else 0.3
                }
            })

        # Gaussian adaptive smoothing
        if denoiser.enable_gaussian_adaptive_smoothing:
            chain.append({
                'type': 'adaptive_gaussian',
                'params': {
                    'sigma_min': denoiser.gaussian_sigma_min,
                    'sigma_max': denoiser.gaussian_sigma_max,
                    'force_threshold': denoiser.gaussian_force_threshold
                }
            })

        # Force-weighted gain
        if postprocess.dynamic_denoise_enabled or postprocess.smoothing_enabled:
            # Different target RMS for different track types
            if track_type == 'impact':
                target_rms = 0.2
                gain_low = 0.5
                gain_high = 1.2
            elif track_type == 'continuous':
                target_rms = 0.1
                gain_low = 0.3
                gain_high = 1.0
            elif track_type == 'noise':
                target_rms = 0.05
                gain_low = 0.2
                gain_high = 0.8
            else:
                target_rms = 0.03
                gain_low = 0.1
                gain_high = 0.5

            chain.append({
({
                'type': 'force_weighted_gain',
                'params': {
                    'target_rms': target_rms,
                    'gain_low': gain_low,
                    'gain_high': gain_high,
                    'max_gain_db': postprocess.max_gain_db,
                    'compression': postprocess.dynamic_range_compression,
                    'smooth_ms': postprocess.smoothing_window_ms,
                    'sample_rate': self.config.system.sample_rate
                }
            })

        # Global normalization
        if postprocess.normalize_output:
            chain.append({
                'type': 'global_normalize',
                'params': {
                    'target_peak': 0.95,
                    'peak_key': 'global_peak'
                }
            })

        return chain

    def _write_chain_metadata(self, obj_idx: int, track_name: str,
                               chain_spec: List[Dict[str, Any]]) -> None:
        """
        Write chain metadata to StorageEngine for a specific track.

        The metadata is stored in the signal attributes and is used by the
        DAG evaluator at read time to apply the processing chain.

        Args:
            obj_idx: Object index
            track_name: Track name
            chain_spec: List of processor specifications
        """
        from pbrAudioStorage.lib.backend import Blosc2Backend

        # Get the backend instance
        backend = self.storage_engine.backend
        if backend is None:
            backend = Blosc2Backend.get(self.storage_engine.tree_store_path, mode="a")

        # Prepare metadata with chain specification
        metadata = {
            'chains': {
                'op': 'sequential',
                'processors': chain_spec
            },
            'post_processed': True,
            'post_process_engine': 'physicsSolver'
        }

        # Update the track metadata in the storage
        with backend._lock:
            store = backend.store
            node_path = backend._node_path('physicsSolver', obj_idx)

            if node_path not in store:
                debug_print(f"Warning: Node {node_path} not found in storage")
                return

            arr = store[node_path]

            # Get track metadata
            obj_meta = arr.attrs.get('object', {})
            track_names = obj_meta.get('track_names', [])

            if track_name not in track_names:
                debug_print(f"Warning: Track {track_name} not found for object {obj_idx}")
                return

            track_index = track_names.index(track_name)

            # Get and update track metadata
            tracks_meta = arr.attrs.get('tracks', [])
            if track_index < len(tracks_meta):
                track_meta = tracks_meta[track_index]

                # Store chain in signal metadata
                # If signals exist, update the first one; otherwise create a placeholder
                signals = track_meta.get('signals', [])

                if signals:
                    # Add chain to each signal's metadata
                    for signal in signals:
                        signal['chains'] = metadata['chains']
                        signal['post_processed'] = True
                        signal['post_process_engine'] = 'physicsSolver'
                else:
                    # No signals yet - store chain in track metadata for later
                    track_meta['pending_chains'] = metadata['chains']
                    track_meta['post_processed'] = True

                # Update the array attributes
                arr.attrs['tracks'] = tracks_meta

        debug_print(f"Wrote chain metadata for {obj_idx}/{track_name}")

    def get_chain_spec(self, obj_idx: int, track_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the chain specification for a specific track.

        Args:
            obj_idx: Object index
            track_name: Track name

        Returns:
            Chain specification dictionary or None if not found
        """
        from pbrAudioStorage.lib.backend import Blosc2Backend

        backend = self.storage_engine.backend
        if backend is None:
            backend = Blosc2Backend.get(self.storage_engine.tree_store_path, mode="r")

        try:
            with backend._lock:
                store = backend.store
                node_path = backend._node_path('physicsSolver', obj_idx)

                if node_path not in store:
                    return None

                arr = store[node_path]
                obj_meta = arr.attrs.get('object', {})
                track_names = obj_meta.get('track_names', [])

                if track_name not in track_names:
                    return None

                track_index = track_names.index(track_name)
                tracks_meta = arr.attrs.get('tracks', [])

                if track_index < len(tracks_meta):
                    track_meta = tracks_meta[track_index]

                    # Check for pending chains (no signals yet)
                    if 'pending_chains' in track_meta:
                        return track_meta['pending_chains']

                    # Check signal-level chains
                    signals = track_meta.get('signals', [])
                    if signals and 'chains' in signals[0]:
                        return signals[0]['chains']

        except Exception as e:
            debug_print(f"Error reading chain spec: {e}")

        return None

    def clear_chains(self, obj_idx: int = None, track_name: str = None) -> None:
        """
        Clear processing chains from storage.

        Args:
            obj_idx: Object index (if None, clear all)
            track_name: Track name (if None, clear all tracks for object)
        """
        from pbrAudioStorage.lib.backend import Blosc2Backend

        backend = self.storage_engine.backend
        if backend is None:
            backend = Blosc2Backend.get(self.storage_engine.tree_store_path, mode="a")

        config = self.config

        with backend._lock:
            store = backend.store

            if obj_idx is not None:
                obj_indices = [obj_idx]
            else:
                obj_indices = [obj.idx for obj in config.objects if not obj.static]

            for o_idx in obj_indices:
                node_path = backend._node_path('physicsSolver', o_idx)

                if node_path not in store:
                    continue

                arr = store[node_path]
                obj_meta = arr.attrs.get('object', {})
                track_names = obj_meta.get('track_names', [])

                if track_name is not None:
                    track_indices = [track_names.index(track_name)] if track_name in track_names else []
                else:
                    track_indices = range(len(track_names))

                tracks_meta = arr.attrs.get('tracks', [])

                for t_idx in track_indices:
                    if t_idx < len(tracks_meta):
                        track_meta = tracks_meta[t_idx]

                        # Remove pending chains
                        track_meta.pop('pending_chains', None)
                        track_meta.pop('post_processed', None)

                        # Remove signal-level chains
                        signals = track_meta.get('signals', [])
                        for signal in signals:
                            signal.pop('chains', None)
                            signal.pop('post_processed', None)
                            signal.pop('post_process_engine', None)

                arr.attrs['tracks'] = tracks_meta

        debug_print(f"Cleared chains for obj_idx={obj_idx}, track_name={track_name}")
