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
inside StorageEngine.read() applies them lazily.

The class no longer talks to Blosc2Backend directly: it goes exclusively
through the StorageEngine public helpers:
    - set_chains()      -> write a chain spec on track / signals
    - clear_chains()    -> remove all chains
    - get_track_meta()  -> read AudioDataType on a track node
    - list_signals()    -> discover which signals exist
    - _signal_node_exists() -> cheap "is the track materialized?" check

All processors referenced here exist in pbrAudioSignals and accept the
side-chain key `force` injected through StorageEngine._get_props() via
`chains["props"]["custom"] = {"entity": "forces", "entity_idx": <obj>}`.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine


# Track-name -> force side-chain name. Currently every track in this engine
# reads `props.custom["force"]`, so the force-track name is used only for
# future-proofing if we ever expose multiple side channels.
DEFAULT_TRACK_FORCE_MAP: Dict[str, str] = {
    "rigidbody": "impact",
    "resonance": "impact",
    "sliding":   "sliding",
    "scraping":  "scraping",
    "rolling":   "rolling",
}

DEFAULT_TRACKS_TO_POST: List[str] = [
    "rigidbody", "resonance", "sliding", "scraping", "rolling",
]

# Per-track target RMS used by force_dynamic_amplify. Kept inline so the
# postProcess behaviour is fully described in this one file.
DEFAULT_TRACK_RMS: Dict[str, float] = {
    "rigidbody": 0.15,
    "resonance": 0.08,
    "sliding":   0.04,
    "scraping":  0.04,
    "rolling":   0.032,
}


def _build_track_chain(track_name: str, cfg: Any) -> Dict[str, Any]:
    """
    Build the `chains` dict for one rigidBodyEngine track.

    The chain assumes the DAG evaluator will inject the current object's
    ForceDataSequence into ``props.custom["force"]`` (StorageEngine reads
    ``chains["props"]["custom"]`` to do so).
    """
    steps: List[Dict[str, Any]] = []

    # --- 1. Force-correlated noise gate --------------------------------
    # adaptive_noise_gate is group=AUDIO_SIGNAL|ACOUSTIC_PRESSURE|AUDIO_FORCE|GENERIC
    # -> valid in this generic per-track chain (it's only a signal processor;
    #    the group check only runs at schema registration time).
    if cfg.postprocess.dynamic_denoise_enabled:
        steps.append({
            "type": "adaptive_noise_gate",
            "params": {
                "threshold_db": cfg.postprocess.noise_gate_threshold_db,
                "attack_ms": 2.0,
                "release_ms": 50.0,
                "hold_ms": 10.0,
                "force_weight": cfg.postprocess.force_reference_weight,
                "sample_rate": cfg.system.sample_rate,
            },
        })

        # --- 2. Spectral subtraction -----------------------------------
        # NOTE: force_spectral_denoise expects fft_size/hop_size/
        # reduction_strength/force_coupling/quiet_threshold — NOT
        # noise_floor_db.  We honour cfg.postprocess.noise_floor_estimate_db
        # by mapping it into a threshold via a soft derived parameter:
        # quiet_threshold = the linear force level below which we estimate
        # the noise floor.  The dB value is just informational on
        # cfg; we simply use the defaults that pbrAudioSignals ships
        # with, which are tuned for typical modal signals.
        steps.append({
            "type": "force_spectral_denoise",
            "params": {
                "fft_size": 2048,
                "hop_size": 512,
                "reduction_strength": cfg.postprocess.spectral_reduction_strength,
                "force_coupling": 0.8,
                "quiet_threshold": 0.1,
                "sample_rate": cfg.system.sample_rate,
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
            },
        })

    # --- 4. Phase alignment -------------------------------------------
    if cfg.postprocess.phase_align_enabled:
        steps.append({
            "type": "force_phase_align",
            "params": {
                "crossfade_samples": cfg.post.postprocess.crossfade_samples,
                "sample_rate": cfg.system.sample_rate,
            },
        })

    # --- 5. Dynamic amplify -------------------------------------------
    target_rms = DEFAULT_TRACK_RMS.get(track_name, cfg.postprocess.target_rms)
    steps.append({
        "type": "force_dynamic_amplify",
        "paramsparams": {
            "target_rms": target_rms,
            "max_gain_db": cfg.postprocess.max_gain_db,
            "dynamic_range_compression": cfg.postprocess.dynamic_range_compression,
            "sample_rate": cfg.system.sample_rate,
        },
    })

    # --- 6. Peak normalize -------------------------------------------
    if cfg.postprocess.normalize_output:
        steps.append({
            "type": "per_track_normalize",
            "params": {"target_peak": 0.95},
        })

    return {
        "op": "sequential",
        "processors": steps,
        # Sidechain injection: StorageEngine._get_props() reads this dict
        # and stuffs the matching ForceDataSequence into props.custom["force"].
        "props": {
            "custom": {
                "entity": "forces",
                "entity_idx": None,   # filled in per-object in apply_for_object()
            },
        },
    }


@dataclass
class postProcess:
    """
    Writes non-destructive chains metadata for rigidBodyEngine's tracks.

    In the new architecture this class is NOT responsible for applying the
    chains — StorageEngine.read() applies them lazily via its DAG evaluator.
    We only describe *what* processing should happen on each track, then
    hand that description to StorageEngine.set_chains().
    """

    entity_manager: EntityManager
    storage_engine: StorageEngine = None

    # Per-object chain overrides. Keys are obj_idx,
    # values are dict[track_name, chains].
    overrides: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # Optional: restrict processing to a subset of tracks.
    tracks_to_process: List[str] = field(
        default_factory=lambda: list(DEFAULT_TRACKS_TO_POST)
    )

    # Force-track mapping overrides (patchable per-instance).
    force_track_map: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TRACK_FORCE_MAP)
    )

    # ------------------------------------------------------------------
    def __post_init__(self):
        config = self.entity_manager.get("config")

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.config = config
        self.collection = config.system.collection
        self.engine_name = "rigidBodyEngine"

        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self) -> Dict[str, int]:
        """
        Write chains metadata for every dynamic object's rigidBodyEngine
        tracks.

        Returns:
            Dict[obj_name, count_of_tracks_patched]
        """
        results: Dict[str, int] = {}
        for obj in self.config.objects:
            name = getattr(obj, "name", str(obj.idx))
            n = self.apply_for_object(obj_idx=obj.idx, obj_name=name)
            results[name] = n
        return results

    def apply_for_object(self, obj_idx: int, obj_name: Optional[str] = None) -> int:
        """
        Write chains for one object's tracks.

        Returns the number of tracks that were patched (0 if the object
        has no materialized tracks).
        """
        cfg_obj = self._find_config_obj(obj_idx)
        if cfg_obj is None:
            debug_print(f"postProcess: no ObjectConfig for obj_idx={obj_idx}")
            return 0

        patched = 0
        for track_name in self.tracks_to_process:
            # Fast skip: track was never materialized (e.g. noise enhancement
            # disabled and this isn't rigidbody/resonance).
            if not self.storage_engine._signal_node_exists(
                self.collection, self.engine_name, obj_idx, track_name
            ):
                debug_print(
                    f"postProcess: "
                    f"{self.collection}/{self.engine_name}/{obj_idx}/{track_name} "
                    f"not materialized — skipped"
                )
                continue

            chain = self._chain_for(obj_idx, track_name)
            if chain is None:
                continue

            n = self._write_chain_to_storage(obj_idx, track_name, chain)
            if n > 0:
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
            if not self.storage_engine._signal_node_exists(
                self.collection, self.engine_name, obj_idx, track_name
            ):
                continue
            n = self.storage_engine.clear_chains(
                collection=self.collection,
                engine=self.engine_name,
                obj_idx=obj_idx,
                track_name=track_name,
            )
            if n > 0:
                patched += 1
        return patched

    def revert_all(self) -> None:
        for obj in self.config.objects:
            self.revert_for_object(obj.idx)

    # ------------------------------------------------------------------
    # Internal: chain resolution
    # ------------------------------------------------------------------
    def _chain_for(
        self, obj_idx: int, track_name: str
    ) -> Optional[Dict[str, Any]]:
        # Caller-supplied override wins.
        if obj_idx in self.overrides and track_name in self.overrides[obj_idx]:
            chain = dict(self.overrides[obj_idx][track_name])
            custom = chain.setdefault("props", {}).setdefault("custom", {})
            custom["entity"] = "forces"
            custom["entity_idx"] = obj_idx
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

    def _write_chain_to_storage(
        self, obj_idx: int, track_name: str, chain: Dict[str, Any]
    ) -> int:
        """
        Attach `chain` to every signal of the track via StorageEngine.

        Returns the number of AudioDataTypeDataType nodes that were updated
        (signals when present, otherwise the track itself).
        """
        try:
            updated = self.storage_engine.set_chains(
                collection=self.collection,
                engine=self.engine_name,
                obj_idx=obj_idx,
                track_name=track_name,
                chains=chain,
                signal_names=None,   # None -> apply to every signal
            )
        except Exception as e:
            debug_print(
                f"postProcess: set_chains failed for "
                f"{self.collection}/{self.engine_name}/{obj_idx}/{track_name}: {e}"
            )
            return 0

        if updated <= 1:
            # Either the track itself was updated (no signals) or
            # nothing was updated; sanity-check against existing signals
            # so log messages stay accurate.
            signals = self.storage_engine.list_signals(
                self.collection, self.engine_name, obj_idx, track_name
            )
            if not signals and updated == 1:
                debug_print(
                    f"postProcess: chain written on track node "
                    f"{track_name} (no signals present)"
                )
        else:
            debug_print(
                f"postProcess: chain written on {updated} signal node(s) "
                f"of {track_name}"
            )
        return updated
