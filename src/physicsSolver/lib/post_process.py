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
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from pbrAudioCommon import EntityManager, ForceDataSequence
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine
from pbrAudioStorage import StorageError

# ---------------------------------------------------------------------------
# Track classification
# ---------------------------------------------------------------------------
# Impact tracks are transient-heavy: DC blocker + envelope smoother, but no
# aggressive spectral work (that would kill the attack).
#
# Continuous tracks (sliding/scraping/rolling) benefit from spectral leak
# reduction because their tail rings much longer than the physical event.
#
# Periodic repeats deserve the periodicity breaker.
#
# `_sound` tracks are the noise-enhancement side tracks synthesized alongside
# the vibration signal. They are the *cleanest* target for spectral work.
#
# `coupling_strength` is a control-side channel; never processed.

_IMPACT_TRACKS    = ("impact",)
_CONTINUOUS_TRACKS = ("sliding", "scraping", "rolling")
_NOISE_TRACKS     = ("sliding_sound", "scraping_sound", "rolling_sound")
_CONTROL_TRACKS   = ("coupling_strength",)
_AMBIENT_TRACKS   = ("non_collision",)


@dataclass
class postProcess:
    """
    Non-destructive post-processor for synthesized audio-force tracks.

    What this class does
    --------------------
    For every dynamic object, walk each audio-force track and *attach* a
    processor chain to its AudioDataType via `StorageEngine.set_chains()`.

    The actual DSP is then executed lazily by `StorageEngine.read(...)` when
    downstream consumers (ModalPlayer, AudioRenderer, analytics, exports,
    etc.) pull the audio:

        StorageEngine.read(...)
            └─ _traverse_signal()
                └─ _process_audio()
                    ├─ _apply_behavior()      # single / merge / multi
                    ├─ _apply_chain()         # <-- this is what we drive
                    │      └─ DagEvaluator
                    │          └─ DCBlockerForce
                    │          └─ ForceEnvelopeSmoothing
                    │          └─ ForceSpectralLeakReduce
                    │          └─ ForcePeriodicityBreaker
                    │          └─ ForceDynamicAmplify
                    └─ _apply_kinds()         # mono / multi / ambisonic

    Key points
    ----------
    * Nothing is written to disk beyond metadata. Re-running postProcess is
      idempotent (chains are replaced, not appended).
    * The `force` side-channel (a `ForceDataSequence`) and `obj_config` are
      injected *by StorageEngine* at read time via `_get_props()`; we only
      need to make sure the chains dict requests them.
    * `sample_rate`, `sfps`, `obj_idx` are exposed to processors via
      `props.custom["params"]` so the same chain works across tracks.
    * Global normalization is *not* applied to the raw tracks; instead, the
      `PerTrackNormalize` processor is appended at the end of every chain with
      a conservative target peak. Truly global normalization belongs to the
      final render stage, which can call `StorageEngine.read(...)` on every
      track, take the max, and pass it as `props.custom["global_peak"]` to a
      `GlobalNormalize` node it appends afterward.

    Parameters
    ----------
    entity_manager : EntityManager
        Provides `config` and the `forces` collection.
    storage_engine : StorageEngine, optional
        If omitted, a new instance is constructed.
    collection : str, optional
        Storage collection. Defaults to `config.system.collection`.
    engine : str, optional
        Storage engine label under which audio-force tracks are stored.
        Defaults to `"physicsEngine"` to match `ForceSynth._write_tracks_to_storage`.
    """

    entity_manager: EntityManager
    storage_engine: Optional[StorageEngine] = None
    collection: Optional[str] = None
    engine: str = "physicsEngine"

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.config = self.entity_manager.get("config")
        set_debug(self.config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.afp = getattr(self.config, "audio_forces_postprocess", None)

        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

        if self.collection is None:
            self.collection = self.config.system.collection

        # Derived timing
        self.sample_rate = int(self.config.system.sample_rate)
        fps = self.config.system.fps
        fps_base = self.config.system.fps_base or 1
        subframes = self.config.system.subframes or 1
        self.sfps = (fps / fps_base) * subframes

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process(self) -> Dict[int, Dict[str, Any]]:
        """
        Walk every dynamic object, attach a chain to each of its tracks, and
        return a per-object report describing what was applied.

        Returns
        -------
        dict
            { obj_idx: { track_name: {"chain_types": [...], "signal_names": [...], "skipped": bool or reason} } }
        """
        debug_print("attaching non-destructive chains to audio-force tracks")

        report: Dict[int, Dict[str, Any]] = {}

        for config_obj in self.config.objects:
            if config_obj.static:
                continue

            force_side = self._resolve_force_for_object(config_obj.idx)
            if force_side is None:
                debug_print(f"skipping obj_idx={config_obj.idx} ({config_obj.name}): no ForceDataSequence found")
                report[config_obj.idx] = {"__skipped__": "no force side-channel"}
                continue

            report[config_obj.idx] = self._process_object(config_obj)

        return report

    # ------------------------------------------------------------------
    # Per-object processing
    # ------------------------------------------------------------------
    def _process_object(self, config_obj: Any) -> Dict[str, Any]:
        obj_idx = config_obj.idx
        obj_report: Dict[str, Any] = {}

        existing_tracks = self._list_existing_tracks(obj_idx)
        if not existing_tracks:
            debug_print(f"obj_idx={obj_idx} ({config_obj.name}): no materialized tracks")
            return obj_report

        # Build the shared `props.custom.params` block that every processor
        # in every chain will see at read time.
        shared_params = {"sample_rate": self.sample_rate, "sfps": self.sfps, "obj_idx": obj_idx}

        for track_name in existing_tracks:
            chain_spec = self._build_chain_for_track(track_name=track_name, config_obj=config_obj, shared_params=shared_params)
            if chain_spec is None:
                obj_report[track_name] = {"skipped": "no-op track"}
                continue
            try:
                changed = self.storage_engine.set_chains(collection=self.collection, engine=self.engine, obj_idx=obj_idx, track_name=track_name, chains=chain_spec, signal_names=None)
            except StorageError as e:
                debug_print(f"failed to set chain on {config_obj.name}/{track_name}: {e}")
                obj_report[track_name] = {"error": str(e)}
                continue

            obj_report[track_name] = {"chain_types": [s["type"] for s in chain_spec["processors"]], "signal_names": list(self.storage_engine.list_signals(self.collection, self.engine, obj_idx, track_name)), "nodes_updated": changed}
            debug_print(f"obj={config_obj.name}({obj_idx}) track={track_name}: chain={[s['type'] for s in chain_spec['processors']]}")

        return obj_report

    # ------------------------------------------------------------------
    # Chain selection
    # ------------------------------------------------------------------
    def _build_chain_for_track(self, track_name: str, config_obj: Any, shared_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return a chain spec (`{"op": "sequential", "processors": [...], "props": {...}}`)
        for the given track, or None if this track should be left untouched.

        The exact parameter defaults mirror the old `AudioForcesDenoiser`
        settings but expose them through the new individual processors.
        """
        afp = self.afp

        # -------- control tracks: never touch --------
        if track_name in _CONTROL_TRACKS:
            return None

        # -------- ambient tracks: only gentle DC + normalize --------
        if track_name in _AMBIENT_TRACKS:
            steps: List[Dict[str, Any]] = []
            if afp is not None and getattr(afp, "enable_dc_blocker", False):
                steps.append({
                    "type": "force_dc_blocker",
                    "params": {
                        "alpha_quiet": float(afp.dc_blocker_alpha),
                        "alpha_active": float(max(0.9999, afp.dc_blocker_alpha)),
                        "enable_local_mean": True,
                    },
                })
            steps.append({
                "type": "per_track_normalize",
                "params": {"target_peak": 0.95},
            })
            return self._wrap_chain(steps, config_obj, shared_params, track_name)

        # -------- impact tracks: DC blocker + envelope smoothing + amplify --------
        if track_name in _IMPACT_TRACKS:
            steps: List[Dict[str, Any]] = []

            if afp is None or getattr(afp, "enable_dc_blocker", False):
                steps.append({
                    "type": "force_dc_blocker",
                    "params": {
                        "alpha_quiet": 0.999,
                        "alpha_active": 0.9999,
                        "min_force": 0.02,
                        "enable_local_mean": True,
                    },
                })

            if afp is None or getattr(afp, "enable_noise_gate", False):
                # The force_processor `ForceEnvelopeSmoothing` dominates the
                # old noise-gate path because it doesn't kill the attack.
                steps.append({
                    "type": "force_envelope_smoothing",
                    "params": {
                        "spike_ratio": 0.35,
                        "smooth_strength_quiet": 0.9,
                        "smooth_strength_active": 0.0,
                        "attr": "normal_force_magnitude",
                    },
                })

            if afp is None or getattr(afp, "enable_temporal_smoothing", False):
                steps.append({
                    "type": "force_adaptive_smoothing",
                    "params": {
                        "alpha_min": 0.05,
                        "alpha_max": 0.6,
                        "smooth_ms": 0.5,
                    },
                })

            if getattr(self.config.synth_postprocess, "dynamic_denoise_enabled", False):
                steps.append({
                    "type": "force_dynamic_amplify",
                    "params": {
                        "target_rms": 0.15,
                        "max_gain_db": 20.0,
                        "dynamic_range_compression": 0.5,
                        "smooth_ms": 5.0,
                    },
                })

            steps.append({
                "type": "per_track_normalize",
                "params": {"target_peak": 0.95},
            })
            return self._wrap_chain(steps, config_obj, shared_params, track_name)

        # -------- continuous tracks: DC + envelope + spectral leak reduction --------
        if track_name in _CONTINUOUS_TRACKS:
            steps: List[Dict[str, Any]] = []

            if afp is None or getattr(afp, "enable_dc_blocker", False):
                steps.append({
                    "type": "force_dc_blocker",
                    "params": {
                        "alpha_quiet": 0.9995,
                        "alpha_active": 0.99995,
                        "min_force": 0.02,
                        "enable_local_mean": True,
                    },
                })

            steps.append({
                "type": "force_envelope_smoothing",
                "params": {
                    "spike_ratio": 0.25,
                    "smooth_strength_quiet": 0.7,
                    "smooth_strength_active": 0.0,
                },
            })

            # Spectral leak reduction is *the* fix for the long ForceSynth tail
            # smearing beyond the true contact duration.
            if afp is None or getattr(afp, "enable_spectral_noise_reduction", False):
                steps.append({
                    "type": "force_spectral_leak_reduce",
                    "params": {
                        "n_fft": 1024,
                        "hop_length": 256,
                        "min_gain_db": -30.0,
                        "leak_margin": 0.25,
                    },
                })

            if getattr(self.config.synth_postprocess, "dynamic_denoise_enabled", False):
                steps.append({
                    "type": "force_dynamic_amplify",
                    "params": {
                        "target_rms": 0.08,
                        "max_gain_db": 18.0,
                        "dynamic_range_compression": 0.6,
                        "smooth_ms": 8.0,
                    },
                })

            steps.append({
                "type": "per_track_normalize",
                "params": {"target_peak": 0.95},
            })
            return self._wrap_chain(steps, config_obj, shared_params, track_name)

        # -------- noise-enhancement tracks: periodicity breaker + spectral denoise --------
        if track_name in _NOISE_TRACKS:
            steps: List[Dict[str, Any]] = []

            # Noise tracks are the *cleanest* place to kill the loop feel
            # because their amplitude story is entirely synthesized.
            steps.append({
                "type": "force_periodicity_breaker",
                "params": {
                    "event_gain_jitter": 0.12,
                    "lfo_depth": 0.04,
                    "lfo_delay_depth": 0.25,
                    "lfo_rate_hz": 0.8,
                    "repetition_boost": 1.8,
                    "coupling_tol": 0.02,
                    "seed": 1337,
                },
            })

            if afp is None or getattr(afp, "enable_spectral_noise_reduction", False):
                # The generic spectral denoise (STFT) is *safe* on the noise
                # tracks because they are already noise; the spectral subtraction

                # simply tightens their band to the physical contact band.
                steps.append({
                    "type": "force_spectral_denoise",
                    "params": {
                        "fft_size": 2048,
                        "hop_size": 512,
                        "reduction_strength": 0.7,
                        "force_coupling": 0.8,
                        "quiet_threshold": 0.1,
                    },
                })

            if getattr(self.config.synth_postprocess, "dynamic_denoise_enabled", False):
                steps.append({
                    "type": "force_dynamic_amplify",
                    "params": {
                        "target_rms": 0.04,
                        "max_gain_db": 16.0,
                        "dynamic_range_compression": 0.7,
                        "smooth_ms": 10.0,
                    },
                })

            steps.append({
                "type": "per_track_normalize",
                "params": {"target_peak": 0.9},
            })
            return self._wrap_chain(steps, config_obj, shared_params, track_name)

        # -------- unknown track: leave untouched --------
        return None

    # ------------------------------------------------------------------
    def _wrap_chain(self, steps: List[Dict[str, Any]], config_obj: Any, shared_params: Dict[str, Any], track_name: str) -> Optional[Dict[str, Any]]:
        """
        Wrap the processor list in the standard chain envelope, attaching
        the side-channel declarations that StorageEngine._get_props() will
        resolve at read time.

        The `props.custom` block has two responsibilities:
          1. Declare the `force` side-channel (a ForceDataSequence).
          2. Declare the `obj_config` side-channel (the ObjectConfig).
        """
        if not steps:
            return None

        # Only declare side-channels if at least one step requires them.
        # Cheap heuristic: force_* processors all need `force`; only a few
        # need `obj_config`. Declaring both unconditionally is harmless
        # because StorageEngine gracefully skips missing ones.
        custom: Dict[str, Any] = {
            "entity": "forces",
            "entity_idx": config_obj.idx,
            "obj_config": True,
            "objs_type": "objects",
        }

        return {
            "op": "sequential",
            "processors": steps,
            "props": {
                "custom": custom,
                "params": dict(shared_params),
            },
            "meta": {
                "track_name": track_name,
                "obj_name": getattr(config_obj, "name", None),
                "obj_idx": config_obj.idx,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_force_for_object(self, obj_idx: int) -> Optional[ForceDataSequence]:
        """
        Return the first ForceDataSequence in the entity manager whose
        `obj_idx` matches the requested object. Each object has exactly one
        aggregated ForceDataSequence that covers *all* its contacts.
        """
        forces = self.entity_manager.get("forces") or {}
        for _, seq in forces.items():
            if isinstance(seq, ForceDataSequence) and getattr(seq, "obj_idx", None) == obj_idx:
                return seq
        return None

    def _list_existing_tracks(self, obj_idx: int) -> List[str]:
        """
        Return the track names that actually exist for this object in the
        backend. We deliberately check the backend rather than the config,
        because `enable_noise_enhancement=False` means noise tracks may
        not exist even though they're in the schema.
        """
        try:
            obj_meta = self.storage_engine._backend.get_meta(path=f"/{self.collection}/{self.engine}/{obj_idx}")
            return list(obj_meta.child_names)
        except Exception as e:
            debug_print(f"could not list tracks for obj_idx={obj_idx}: {e}")
            return []

    # ------------------------------------------------------------------
    # Optional: cross-track global normalization helper
    # ------------------------------------------------------------------
    def compute_global_peak(self, obj_indices: Optional[Sequence[int]] = None, max_samples: int = 0, unprocessed: bool = True) -> float:
        """
        Compute a scene-wide peak across every audio-force track.

        This is the *replacement* for the old `GlobalNormalize.process()`
        file-scanning approach. Callers who want explicit global
        normalization append a `global_normalize` processor with
            params: {"peak_key": "global_peak"}
        and feed `props.custom["global_peak"] = peak` through the DAG
        evaluator at read time.

        Parameters
        ----------
        obj_indices : sequence of int, optional
            Objects to include. Defaults to all dynamic objects.
        max_samples : int
            If > 0, only read the first `max_samples` of each track (fast
            approximation for very long scenes).
        unprocessed : bool
            Read raw signals without evaluating any attached chain. That is
            the right thing when you intend to install a `global_normalize`
            node later (otherwise normalization cycles).

        Returns
        -------
        float
            The maximum absolute sample across all included tracks, or 1.0
            if nothing was found.
        """
        if obj_indices is None:
            obj_indices = [o.idx for o in self.config.objects if not o.static]

        peak = 0.0
        for obj_idx in obj_indices:
            for track_name in self._list_existing_tracks(obj_idx):
                if track_name in _CONTROL_TRACKS:
                    continue
                try:
                    sig = self.storage_engine.read(
                        collection=self.collection,
                        engine=self.engine,
                        obj_idx=obj_idx,
                        track_name=track_name,
                        unprocessed=unprocessed,
                    )
                except Exception as e:
                    debug_print(
                        f"cannot read {obj_idx}/{track_name} for global peak: {e}"
                    )
                    continue
                if sig is None or sig.size == 0:
                    continue
                if max_samples and sig.shape[-1] > max_samples:
                    sig = sig[..., :max_samples]
                local = float(np.max(np.abs(sig)))
                if local > peak:
                    peak = local

        return peak if peak > 0.0 else 1.0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def dump(self) -> Dict[int, Dict[str, Any]]:
        """
        Read back the currently attached chains for every dynamic object.

        Useful for debugging / regression tests.
        """
        out: Dict[int, Dict[str, Any]] = {}
        for config_obj in self.config.objects:
            if config_obj.static:
                continue
            obj_idx = config_obj.idx
            out[obj_idx] = {}
            for track_name in self._list_existing_tracks(obj_idx):
                try:
                    meta = self.storage_engine.get_track_meta(
                        collection=self.collection,
                        engine=self.engine,
                        obj_idx=obj_idx,
                        track_name=track_name,
                    )
                except StorageError:
                    continue
                out[obj_idx][track_name] = {
                    "chains": meta.chains,
                    "signal_names": list(meta.child_names),
                }
        return out

    def clear(self) -> int:
        """
        Remove every chain attached by a previous call to `process()`.

        Returns the number of track nodes that were reset.
        """
        cleared = 0
        for config_obj in self.config.objects:
            if config_obj.static:
                continue
            for track_name in self._list_existing_tracks(config_obj.idx):
                try:
                    cleared += self.storage_engine.clear_chains(
                        collection=self.collection,
                        engine=self.engine,
                        obj_idx=config_obj.idx,
                        track_name=track_name,
                    )
                except StorageError as e:
                    debug_print(f"clear_chains failed for {config_obj.idx}/{track_name}: {e}")
        return cleared
