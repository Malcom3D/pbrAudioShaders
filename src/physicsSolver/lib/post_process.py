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
from typing import Any, Dict, List, Optional

from dataclasses import dataclass

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine
from pbrAudioSignals.core.groups import AUDIO_FORCE, GENERIC


# Track name -> (semantic track type, default processor params)
# The semantics drive the choice of processor params below.
_TRACK_TYPES: Dict[str, str] = {
    'impact':            'impact',
    'sliding':           'continuous',
    'scraping':          'continuous',
    'rolling':           'continuous',
    'non_collision':     'ambient',
    'coupling_strength': 'control',
}

_NOISE_TRACKS: Dict[str, str] = {
    'sliding_sound':  'noise',
    'scraping_sound': 'noise',
    'rolling_sound':  'noise',
}

@dataclass
class postProcess:
    """
    Non-destructive post-processing engine for synthesized audio forces.

    Writes processor *chain metadata* onto each track's AudioDataType via the
    StorageEngine. Actual DSP runs at read time through the DAG evaluator in
    pbrAudioStorage, so nothing is ever re-computed on disk.

    This replaces the deprecated AudioForcePostProcessEngine (which used the
    old postProcess module and wrote processed WAVs).
    """

    entity_manager: EntityManager
    storage_engine: Optional[StorageEngine] = None

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.config = config
        self.denoiser = config.denoiser
        self.postprocess = config.postprocess

        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

        # Engine key as registered in physicsEngine._register_storage_schema
        self.engine = "physicsEngine" # -> "postProcess"
        # Fallback engine name used by ForceSynth. If your ForceSynth writes
        # under "physicsSolver" keep the constant below in sync.
        self.force_engine = "physicsEngine"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self) -> None:
        """Configure read-time processing chains for every dynamic object."""
        config = self.config
        debug_print("Configuring processing chains...")

        # Global normalisation target: peak across all *raw* tracks.
        global_peak = self._compute_global_peak()
        debug_print(f"Global peak value: {global_peak}")

        configured = 0
        for config_obj in config.objects:
            if not self._object_has_tracks(config_obj.idx):
                continue
            self._configure_object_chains(config_obj, global_peak)
            configured += 1

        debug_print(f"Configured processing chains for {configured} objects")

    def clear_chains(self, obj_idx: Optional[int] = None, track_name: Optional[str] = None) -> None:
        """
        Clear processing chains via the StorageEngine.

        Args:
            obj_idx    : if None, clear all dynamic objects.
            track_name : if None, clear every track of the object(s).
        """
        config = self.config

        if obj_idx is not None:
            obj_indices = [obj_idx]
        else:
            obj_indices = [o.idx for o in config.objects if not o.static]

        track_names = list(_TRACK_TYPES.keys())
        if config.system.enable_noise_enhancement:
            track_names += list(_NOISE_TRACKS.keys())

        cleared = 0
        for o_idx in obj_indices:
            if not self._object_has_tracks(o_idx):
                continue

            targets = [track_name] if track_name else track_names
            for t_name in targets:
                try:
                    n = self.storage_engine.clear_chains(
                        collection=config.system.collection,
                        engine=self.force_engine,
                        obj_idx=o_idx,
                        track_name=t_name,
                    )
                    cleared += n
                except Exception as e:
                    debug_print(
                        f"clear_chains failed for {o_idx}/{t_name}: {e}"
                    )

        debug_print(f"Cleared {cleared} chain-bearing nodes")

    def get_chain_spec(self, obj_idx: int, track_name: str) -> Optional[Dict[str, Any]]:
        """Read the current chain spec for an object's track (metadata only)."""
        try:
            meta = self.storage_engine.get_track_meta(
                collection=self.config.system.collection,
                engine=self.force_engine,
                obj_idx=obj_idx,
                track_name=track_name,
            )
        except Exception as e:
            debug_print(f"get_chain_spec: {e}")
            return None

        # Track-level chains are the mirror that set_chains() writes.
        return meta.chains

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def _object_has_tracks(self, obj_idx: int) -> bool:
        """
        True if *any* physicsEngine track node exists for this object.

        We use the StorageEngine.read() with unprocessed=True so the DAG
        evaluator never tries to run a chain that isn't installed yet.
        """
        for t_name in _TRACK_TYPES:
            try:
                data = self.storage_engine.read(
                    collection=self.config.system.collection,
                    engine=self.force_engine,
                    obj_idx=obj_idx,
                    track_name=t_name,
                    signal_name=None,
                    unprocessed=True,
                )
                if data is not None and data.size > 0 and np.any(data):
                    return True
            except Exception:
                continue
        return False

    def _compute_global_peak(self) -> float:
        """Peak across all raw (unprocessed) physicsSolver tracks."""
        config = self.config
        all_tracks = list(_TRACK_TYPES.keys())
        if config.system.enable_noise_enhancement:
            all_tracks += list(_NOISE_TRACKS.keys())

        max_peak = 0.0
        for config_obj in config.objects:
            if config_obj.static:
                continue
            for t_name in all_tracks:
                try:
                    data = self.storage_engine.read(
                        collection=config.system.collection,
                        engine=self.force_engine,
                        obj_idx=config_obj.idx,
                        track_name=t_name,
                        signal_name=None,
                        unprocessed=True,
                    )
                    if data is not None and data.size > 0:
                        max_peak = max(max_peak, float(np.max(np.abs(data))))
                except Exception:
                    continue

        return max_peak if max_peak > 0 else 1.0

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------
    def _configure_object_chains(self, config_obj: Any, global_peak: float) -> None:
        """Build & attach chain specs for for every track of one object."""
        config = self.config

        track_types = dict(_TRACK_TYPES)
        if config.system.enable_noise_enhancement:
            track_types.update(_NOISE_TRACKS)

        for track_name, track_type in track_types.items():
            # Only touch tracks that actually contain audio.
            if not self._track_has_audio(config_obj.idx, track_name):
                continue

            chain_spec = self._build_chain_spec(
                track_name=track_name,
                track_type=track_type,
                config_obj=config_obj,
                global_peak=global_peak,
            )

            if not chain_spec:
                continue

            try:
                n = self.storage_engine.set_chains(
                    collection=config.system.collection,
                    engine=self.force_engine,
                    obj_idx=config_obj.idx,
                    track_name=track_name,
                    chains={
                        "op": "sequential",
                        "processors": chain_spec,
                    },
                )
                debug_print(
                    f"Configured chain for {config_obj.name}/{track_name}: "
                    f"{len(chain_spec)} processors ({n} nodes updated)"
                )
            except Exception as e:
                debug_print(
                    f"set_chains failed for {config_obj.name}/{track_name}: {e}"
                )

    def _track_has_audio(self, obj_idx: int, track_name: str) -> bool:
        try:
            data = self.storage_engine.read(
                collection=self.config.system.system.collection,
                engine=self.force_engine,
                obj_idx=obj_idx,
                track_name=track_name,
                signal_name=None,
                unprocessed=True,
            )
            return data is not None and data.size > 0 and bool(np.any(data))
        except Exception:
            return False

    def _build_chain_spec(self, track_name: str, track_type: str,
                          config_obj: Any, global_peak: float) -> List[Dict[str, Any]]:
        """Build a list of {type, params} dicts for one track."""
        denoiser = self.denoiser
        pp = self.postprocess
        sr = self.config.system.sample_rate

        # Control signal: leave alone.
        if track_type == 'control':
            return []

        chain: List[Dict[str, Any]] = []

        # --- 1. DC blocker (always first if enabled) ---
        if denoiser.enable_dc_blocker or pp.dynamic_denoise_enabled:
            chain.append({
                "type": "dc_blocker",
                "params": {
                    "alpha": denoiser.dc_blocker_alpha
                    if denoiser.enable_dc_blocker else 0.999,
                },
            })

        # --- 2. Adaptive noise gate (force-correlated) ---
        if denoiser.enable_noise_gate or pp.dynamic_denoise_enabled:
            threshold_db, force_weight = self._gate_params_for(track_type)
            chain.append({
                "type": "adaptive_noise_gate",
                "params": {
                    "sample_rate": sr,
                    "threshold_db": threshold_db,
                    "attack_ms": denoiser.gate_attack_ms,
                    "release_ms": denoiser.gate_release_ms,
                    "hold_ms": denoiser.gate_hold_ms,
                    "force_weight": force_weight,
                },
            })

        # --- 3. Temporal smoothing ---
        if denoiser.enable_temporal_smoothing or pp.smoothing_enabled:
            window = 3 if track_type == 'impact' else (
                denoiser.temporal_smoothing_window if track_type == 'continuous' else 5
            )
            chain.append({
                "type": "temporal_smoothing",
                "params": {
                    "window": window,
                    "transient_threshold": 0.15,
                },
            })

        # --- 4. Spectral denoising ---
        if denoiser.enable_spectral_noise_reduction or pp.dynamic_denoise_enabled:
            quiet, loud = self._spectral_params_for(track_type, pp)
            chain.append({
                "type": "spectral_denoise",
                "params": {
                    "n_fft": denoiser.spectral_fft_size,
                    "hop_length": denoiser.spectral_hop_size,
                    "reduction_strength_quiet": quiet,
                    "reduction_strength_loud": loud,
                    "noise_floor_percentile": 10.0,
                },
            })

        # --- 5. Envelope shaping ---
        if denoiser.enable_envelope_shaping:
            chain.append({
                "type": "envelope_shaper",
                "params": {
                    "depth": 0.5 if track_type == 'continuous' else 0.3,
                },
            })

        # --- 6. Gaussian adaptive smoothing ---
        if denoiser.enable_gaussian_adaptive_smoothing:
            chain.append({
                "type": "adaptive_gaussian",
                "params": {
                    "sigma_min": denoiser.gaussian_sigma_min,
                    "sigma_max": denoiser.gaussian_sigma_max,
                    "force_threshold": denoiser.gaussian_force_threshold,
                },
            })

        # --- 7. Force-weighted gain ---
        if pp.dynamic_denoise_enabled or pp.smoothing_enabled:
            target_rms, gain_low, gain_high = self._gain_params_for(track_type)
            chain.append({
                "type": "force_weighted_gain",
                "params": {
                    "target_rms": target_rms,
                    "gain_low": gain_low,
                    "gain_high": gain_high,
                    "max_gain_db": pp.max_gain_db,
                    "compression": pp.dynamic_range_compression,
                    "smooth_ms": pp.smoothing_window_ms,
                    "sample_rate": sr,
                },
            })

        # --- 8. Scene-wide normalisation ---
        if pp.normalize_output:
            chain.append({
                "type": "global_normalize",
                "params": {
                    "target_peak": 0.95,
                    "peak_key": "global_peak",
                },
            })

        return chain

    # ------------------------------------------------------------------
    # Parameter palette per track type
    # ------------------------------------------------------------------
    @staticmethod
    def _gate_params_for(track_type: str):
        if track_type == 'impact':
            return -70.0, 0.7
        if track_type == 'continuous':
            return -60.0, 0.5
        if track_type == 'noise':
            return -50.0, 0.3
        return -40.0, 0.1  # ambient / anything else

    @staticmethod
    def _spectral_params_for(track_type: str, pp: Any):
        if track_type == 'impact':
            return 0.5, 0.1
        if track_type == 'continuous':
            return pp.spectral_reduction_strength, 0.2
        if track_type == 'noise':
            return 0.9, 0.3
        return 1.0, 0.4  # # ambient

    @staticmethod
    def _gain_params_for(track_type: str):
        if track_type == 'impact':
            return 0.20, 0.5, 1.2
        if track_type == 'continuous':
            return 0.10, 0.3, 1.0
        if track_type == 'noise':
            return 0.05, 0.2, 0.8
        return 0.03, 0.1, 0.5  # ambient

