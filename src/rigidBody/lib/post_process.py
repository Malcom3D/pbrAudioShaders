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

""" 
Non-destructive chain writer for rigidBodyEngine tracks.

This replaces the old pbrAudioPostProcess.postProcess. Instead of
loading/writing audio, it writes `chains` metadata onto the AudioDataType
nodes produced by rigidBodyEngine in pbrAudioStorage. The DAG evaluator
inside StorageEngine.read() applies them.

Signal processing is delegated to pbrAudioSignals processors (looked up
through `pbrAudioSignals.get_processor(type)`).
"""

import os
from typing import Dict, List, Optional, Any

from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine, AudioDataType, Blosc2Backend

# ---------------------------------------------------------------------------
# track  ->  force-track mapping
# ---------------------------------------------------------------------------
# The old code mapped each rendered track name to an audio_force raw file.
# In the new architecture, force tracks live under engine="physicsSolver"
# and are read via StorageEngine.read(). We expose the mapping so callers
# can override it if needed.
DEFAULT_TRACK_FORCE_MAP: Dict[str, str] = {
    'rigidbody':  'impact',               # rigidbody + resonance use impact
    'resonance':  'impact',
    'sliding':    'sliding',
    'scraping':   'scraping',
    'rolling':    'rolling',
}

# Modal Player also writes these noise-enhancement tracks - the old
# postProcess did not process them, so we skip them here. If you want
# to post-process them too, add them to TRACKS_TO_POST.
DEFAULT_TRACKS_TO_POST: List[str] = ['rigidbody', 'resonance', 'sliding', 'scraping', 'rolling']


# ---------------------------------------------------------------------------
# DEFAULT CHAIN FACTORY
# ---------------------------------------------------------------------------
# Chain step format is what StorageEngine._build_chains_for_node expects:
#   {"type": "<processor name>", "params": {...}}
# and validated by pbrAudioSignals.get_processor().
#
# The old postProcess applied these steps in order:
#   1. _apply_noise_gate           (force-correlated)
#   2. _spectral_denoise           (force-correlated, STFT)
#   3. _adaptive_smoothing         (force-correlated EMA)
#   4. _phase_align                (transient alignment vs force)
#   5. _dynamic_amplify            (RMS + force envelope gain)
#   6. normalize_output            (peak normalize to 0.95)
#
# We must map each to a pbrAudioSignals processor type. The names below
# are the canonical names; if your pbrAudioSignals registry differs,
# adjust the `type` strings (or add aliases in pbrAudioSignals).

def _build_track_chain(track_name: str, cfg: Any) -> Dict[str, Any]:
    """
    Build the `chains` dict for one rigidBodyEngine track.

    The chain expects the current + lookahead sample window passed by the
    DAG evaluator, and reads `props.custom["force"]` (a ForceDataSequence)
    as a sidechain - that is injected by StorageEngine._get_props because
    the AudioDataType.chains[*].props.custom has {"entity": "forces",
    "entity_idx": <obj_idx>}.
    """
    steps: List[Dict[str, Any]] = []

    # --- 1. Force-correlated noise gate --------------------------------
    if cfg.postprocess.dynamic_denoise_enabled:
        steps.append({
            "type": "adaptive_noise_gate",
            "params": {
                "threshold_db": cfg.postprocess.noise_gate_threshold_db,
                "attack_ms": 2.0,
                "release_ms": 50.0,
                "hold_ms": 10.0,
#                "force_track": DEFAULT_TRACK_FORCE_MAP.get(track_name, 'impact'),
#                "min_force_threshold": cfg.postprocess.min_force_threshold,
                "force_weight": cfg.postprocess.force_reference_weight,
                "sample_rate": cfg.system.sample_rate,
            },
        })

        # --- 2. Spectral denoise --------------------------------------
        steps.append({
            "type": "force_spectral_denoise",
            "params": {
                "fft_size": 2048,
                "hop_size": 512,
                "noise_floor_db": cfg.postprocess.noise_floor_estimate_db,
                "reduction_strength": cfg.postprocess.spectral_reduction_strength,
                "sample_rate": cfg.system.sample_rate,
#                "force_track": DEFAULT_TRACK_FORCE_MAP.get(track_name, 'impact'),
            },
        })

    # --- 3. Adaptive smoothing ----------------------------------------
    if cfg.postprocess.smoothing_enabled:
        steps.append({
            "type": "force_adaptive_smoothing",
            "params": {
                "alpha_min": 0.1,
                "alpha_max": 0.9 if cfg.postprocess.adaptive_smoothing else 0.5,
                "smooth_ms": cfg.postprocess.smoothing_window_ms,
                "sample_rate": cfg.system.sample_rate,
#                "window_ms": cfg.postprocess.smoothing_window_ms,
#                "adaptive": cfg.postprocess.adaptive_smoothing,
#                "force_track": DEFAULT_TRACK_FORCE_MAP.get(track_name, 'impact'),
            },
        })

    # --- 4. Phase alignment -------------------------------------------
    if cfg.postprocess.phase_align_enabled:
        steps.append({
            "type": "force_phase_align",
            "params": {
                "crossfade_samples": cfg.postprocess.crossfade_samples,
                "sample_rate": cfg.system.sample_rate,
#                "force_track": DEFAULT_TRACK_FORCE_MAP.get(track_name, 'impact'),
            },
        })

    # --- 5. Dynamic amplify -------------------------------------------
    # Different tracks historically had different target RMS values.
    track_rms_overrides = {
        'rigidbody': 0.15,
        'resonance': 0.08,
        'sliding':   0.04,
        'scraping':  0.04,
        'rolling':   0.032,
    }
    target_rms = track_rms_overrides.get(track_name, cfg.postprocess.target_rms)
    steps.append({
        "type": "force_dynamic_amplify",
        "params": {
            "target_rms": target_rms,
            "max_gain_db": cfg.postprocess.max_gain_db,
            "dynamic_range_compression": cfg.postprocess.dynamic_range_compression,
            "sample_rate": cfg.system.sample_rate,
#            "force_track": DEFAULT_TRACK_FORCE_MAP.get(track_name, 'impact'),
        },
    })

    # --- 6. Peak normalize -------------------------------------------
    if cfg.postprocess.normalize_output:
        steps.append({
            "type": "per_track_normalize",
            "params": {"target_peak": 0.95},
#            "type": "peak_normalize",
#            "params": {"peak": 0.95},
        })

    return {
        "op": "sequential",
        "processors": steps,
        # Sidechain injection: StorageEngine._get_props reads this and
        # appends the matching ForceDataSequence to `props` before
        # applying the chain.
        "props": {
            "custom": {
                "entity": "forces",
                "entity_idx": None,  # filled in per-object in apply()
            },
        },
    }


@dataclass
class postProcess:
    """
    Writes non-destructive chains metadata for rigidBodyEngine's tracks.

    In the new architecture this class is NOT responsible for applying the
    chains - StorageEngine.read() does that lazily via its DAG evaluator.
    We only describe *what* processing should happen on each track.
    """

    entity_manager: EntityManager
    storage_engine: StorageEngine = None

    # Per-object chain overrides. Keys are obj_idx, values are dict[track_name, chains].
    overrides: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # Optional: restrict processing to a subset of tracks. Defaults to
    # rigidbody/resonance/sliding/scraping/rolling.
    tracks_to_process: List[str] = field(default_factory=lambda: list(DEFAULT_TRACKS_TO_POST))

    # Force-track mapping overrides (patchable per-instance).
    force_track_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TRACK_FORCE_MAP))

    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.config = config
        self.collection = config.system.collection
        self.engine_name = "rigidBodyEngine"  # storage engine dir name

        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self) -> Dict[str, int]:
        """
        Write chains metadata for every dynamic object's rigidBodyEngine tracks.

        Returns:
            Dict[obj_name, count_of_tracks_patched]
        """
        results: Dict[str, int] = {}
        for obj in self.config.objects:
            n = self.apply_for_object(obj_idx=obj.idx, obj_name=obj.name if hasattr(obj, 'name') else str(obj.idx))
            if n is not None:
                results[obj.name if hasattr(obj, 'name') else str(obj.idx)] = n
        return results

    def apply_for_object(self, obj_idx: int, obj_name: str = None) -> int:
        """
        Write chains for one object's tracks. Returns the number of tracks
        that were patched (0 if the object has no tracks).
        """
        cfg_obj = self._find_config_obj(obj_idx)
        if cfg_obj is None:
            return 0

        patched = 0
        for track_name in self.tracks_to_process:
            chain = self._chain_for(obj_idx, track_name)
            if chain is None:
                continue
            if self._write_chain_to_storage(obj_idx, track_name, chain):
                patched += 1
        if patched:
            debug_print(
                f"postProcess: patched {patched} tracks for "
                f"{obj_name or obj_idx}"
            )
        return patched

    def revert_for_object(self, obj_idx: int) -> int:
        """Remove chains from an object's rigidBodyEngine tracks."""
        patched = 0
        for track_name in self.tracks_to_process:
            if self._clear_chain_from_storage(obj_idx, track_name):
                patched += 1
        return patched

    def revert_all(self) -> None:
        for obj in self.config.objects:
            self.revert_for_object(obj.idx)

    # ------------------------------------------------------------------
    # Internal: chain resolution
    # ------------------------------------------------------------------
    def _chain_for(self, obj_idx: int, track_name: str) -> Optional[Dict[str, Any]]:
        # Caller-supplied override wins.
        if obj_idx in self.overrides and track_name in self.overrides[obj_idx]:
            chain = dict(self.overrides[obj_idx][track_name])
            chain.setdefault("props", {}).setdefault("custom", {})["entity"] = "forces"
            chain["props"]["custom"]["entity_idx"] = obj_idx
            return chain

        if track_name not in self.tracks_to_process:
            return None

        # Build default chain from config.
        chain = _build_track_chain(track_name, self.config)
        # Inject per-object force sidechain reference.
        chain["props"] = {
            "custom": {"entity": "forces", "entity_idx": obj_idx},
        }
        return chain

    # ------------------------------------------------------------------
    # Internal: storage mutation
    # ------------------------------------------------------------------
    def _find_config_obj(self, obj_idx: int):
        for obj in self.config.objects:
            if obj.idx == obj_idx:
                return obj
        return None

    def _write_chain_to_storage(self, obj_idx: int, track_name: str, chain: Dict[str, Any]) -> bool:
        """
        Attach `chain` to `<collection>/rigidBodyEngine/<obj_idx>/<track_name>`
        in the storage backend.

        Both the track-level AudioDataType and the track array's `signals`
        metadata get the chain so the DAG evaluator and any downstream
        tree traversal both see it.
        """
        backend: Blosc2Backend = self.storage_engine._backend
        path = f"/{self.collection}/{self.engine_name}/{obj_idx}/{track_name}"

        if path not in backend._wstore:
            debug_print(
                f"postProcess: {path} not materialized - skipping "
                f"(is the object dynamic and was the schema registered?)"
            )
            return False

        try:
            track_arr = backend._wstore[path]
        except Exception as e:
            debug_print(f"postProcess: cannot open {path}: {e}")
            return False

#        # 1) Attach to the track-level metadata.
#        track_meta = AudioDataType.from_dict(track_arr.attrs['track'])
#        track_meta.chains = chain
#        track_arr.attrs['track'] = track_meta.to_dict()

        # Attach only to each signal-level AudioDataType so that
        # reading via _traverse_signal finds it after merging signals.
        signals_meta_raw = list(track_arr.attrs.get('signals', []))
        for i, sig in enumerate(signals_meta_raw):
            sig_dt = AudioDataType.from_dict(sig)
            sig_dt.chains = chain
            signals_meta_raw[i] = sig_dt.to_dict()
        track_arr.attrs['signals'] = signals_meta_raw

        # Write back - blosc2 persists attrs on reassignment.
        backend._wstore[path] = track_arr
        return True

    def _clear_chain_from_storage(self, obj_idx: int, track_name: str) -> bool:
        backend: Blosc2Backend = self.storage_engine._backend
        path = f"/{self.collection}/{self.engine_name}/{obj_idx}/{track_name}"
        if path not in backend._wstore:
            return False
        try:
            track_arr = backend._wstore[path]
        except Exception:
            return False

        track_meta = AudioDataType.from_dict(track_arr.attrs['track'])
        if not track_meta.chains:
            return False
        track_meta.chains = None
        track_arr.attrs['track'] = track_meta.to_dict()

        signals_meta_raw = list(track_arr.attrs.get('signals', []))
        for i, sig in enumerate(signals_meta_raw):
            sig_dt = AudioDataType.from_dict(sig)
            sig_dt.chains = None
            signals_meta_raw[i] = sig_dt.to_dict()
        track_arr.attrs['signals'] = signals_meta_raw

        backend._wstore[path] = track_arr
        return True
