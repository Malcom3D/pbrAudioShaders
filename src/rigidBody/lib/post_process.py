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
Non-destructive chain writer for rigidBodyEngine MODAL_DISPLACEMENT tracks.

This is the successor to the old pbrAudioPostProcess.postProcess module.
It writes ``chains`` metadata onto the AudioDataType nodes produced by
rigidBodyEngine in pbrAudioStorage. StorageEngine's DAG evaluator applies
them lazily at ``read()`` time.

Design notes
------------
The set of processors below is now split along two axes:

  * **MODAL_DISPLACEMENT tracks** (``rigidbody``, ``resonance``)
    use the ``force_modal_*`` family. These read the current object's
    ``ForceDataSequence`` via ``props.custom["force"]`` and treat
    ``obj_config`` as a physical reference for size/mass/contact
    calibration.

  * **Noise tracks** (``sliding``, ``scraping``, ``rolling``) stay
    on the AUDIO_FORCE / AUDIO_SIGNAL pipeline. They still use
    ``force_adaptive_smoothing``, ``force_phase_align``,
    ``force_dynamic_amplify`` — the modal-specific variants assume
    modal synthesis and would produce incorrect behaviour on noise.

Because a track is idempotent when re-applied, ``apply_for_object()``
can be called repeatedly without double-processing: ``set_chains()``
overwrites the previous chain dict.

The chain for MODAL_DISPLACEMENT tracks uses the following order,
which is the physically correct signal flow:

    1. force_modal_radiation_calibrator   (static + force_rms)
    2. force_modal_equalizer              (size/mass/contact-type shelf)
    3. force_modal_contact_tilt           (contact-type shelf + sharpen)
    4. force_modal_dc_blocker             (envelope-shaped DC)
    5. force_modal_smoothing              (fixed-length spike / dip)
    6. force_modal_compressor             (force-driven compression)
    7. force_modal_phase_align            (align to noise reference)
    8. force_modal_dynamic_amplify        (final level match)
    9. per_track_normalize                (peak ceiling)

The noise-track chain keeps the AUDIO_FORCE / AUDIO_SIGNAL pipeline:

    1. force_adaptive_smoothing
    2. force_phase_align
    3. force_dynamic_amplify
    4. per_track_normalize
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine

# ---------------------------------------------------------------------------
# Track classification
# ---------------------------------------------------------------------------
# Tracks whose Buffer is *modal displacement* (meters), processed by the
# force_modal_* family.
MODAL_TRACKS: List[str] = ["rigidbody", "resonance"]

# Tracks whose Buffer is an *audio-signal* representation of a contact
# (acoustic noise), processed by the force_adaptive_*/force_phase_align
# family.
NOISE_TRACKS: List[str] = ["sliding", "scraping", "rolling"]

# Everything the pipeline is allowed to touch, in a stable order.
DEFAULT_TRACKS_TO_POST: List[str] = MODAL_TRACKS + NOISE_TRACKS

# Per-track RMS targets. These are the value used at
# ``force_modal_dynamic_amplify`` (modal) or ``force_dynamic_amplify``
# (noise). They assume the calibrator has already placed the modal
# track on a physically plausible absolute level.
DEFAULT_TRACK_RMS: Dict[str, float] = {
    "rigidbody": 0.15,
    "resonance": 0.08,
    "sliding":   0.04,
    "scraping":  0.0036,
    "rolling":   0.032,
}

# Which FORCE track each sound is driven by. This map is informational
# only — the actual side-chain is bound by ``force_modal_*`` reading
# ``props.custom["force"]`` (see ``_sidechain_for``).
DEFAULT_FORCE_TRACK_MAP: Dict[str, str] = {
    "rigidbody":  "impact",
    "resonance":  "impact",
    "sliding":    "sliding",
    "scraping":   "scraping",
    "rolling":    "rolling",
}

# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------
def _build_modal_track_chain(track_name: str, cfg: Any) -> Dict[str, Any]:
    """
    Chain for MODAL_DISPLACEMENT tracks (rigidbody, resonance).

    Assumes ``force_modal_*`` processors will read the current object's
    ``ForceDataSequence`` from ``props.custom["force"]`` and the
    ``ObjectConfig`` from ``props.custom["obj_config"]``. The
    ``StorageEngine._get_props()`` helper injects those when it sees the
    matching ``entity`` / ``obj_config`` entries in the chain's
    ``props.custom``.
    """
    sp = cfg.synth_postprocess
    sr = cfg.system.sample_rate
    sfps = (cfg.system.fps / cfg.system.fps_base) * cfg.system.subframes

    steps: List[Dict[str, Any]] = []

    # --- 1. Radiation calibration (absolute level from size/mass) -----
    # This MUST be first: it establishes the physical reference so the
    # equalizer and compressor see a meaningful absolute scale.
    steps.append({
        "type": "force_modal_radiation_calibrator",
        "params": {
            # Use the same RMS that this track targets, so the calibrator
            # and the amplifer are consistent.
            "target_rms_active": DEFAULT_TRACK_RMS.get(track_name, sp.target_rms),
            "force_threshold": 0.05,
            "env_smooth_ms": sp.smoothing_window_ms or 5.0,
            "mode": "force_rms",
            "sample_rate": sr,
            "sfps": sfps,
        },
    })

    # --- 2. Size/mass/contact-type equalizer -------------------------
    steps.append({
        "type": "force_modal_equalizer",
        "params": {
            # Sensible defaults from the modern processor set.
            "size_pivot": 0.2,
            "mass_pivot": 1.0,
            "brightness_db_range": 6.0,
            "hardness_db_range": 3.0,
            "sample_rate": sr,
            "sfps": sfps,
        },
    })

    # --- 3. Contact-type tilt + IMPACT transient sharpen -------------
    steps.append({
        "type": "force_modal_contact_tilt",
        "params": {
            "sample_rate": sr,
            "sfps": sfps,
            "impulse_sharpen_depth": 0.20,
            "impulse_sharpen_ms": 3.0,
        },
    })

    # --- 4. Envelope-shaped DC removal ------------------------------
    steps.append({
        "type": "force_modal_dc_blocker",
        "params": {
            "sample_rate": sr,
            "sfps": sfps,
            "use_object_tau": True,
            "reference_tau_ms": 20.0,
        },
    })

    # --- 5. Spike / cancellation smoothing --------------------------
    if sp.smoothing_enabled:
        steps.append({
            "type": "force_modal_smoothing",
            "params": {
                "spike_ratio": 0.35,
                "blend_quiet": 0.9,
                "blend_active": 0.0 if sp.adaptive_smoothing else 0.15,
                "sample_rate": sr,
                "sfps": sfps,
            },
        })

    # --- 6. Force-driven compressor ---------------------------------
    steps.append({
        "type": "force_modal_compressor",
        "params": {
            "thr_min_db": -6.0,
            "thr_max_db": -24.0,
            "ratio_min": 2.0,
            "ratio_max": 4.0 if sp.adaptive_smoothing else 6.0,
            "attack_ms": 2.0,
            "release_ms": 80.0,
            "makeup_db": 0.0,
            "smooth_ms": sp.smoothing_window_ms or 5.0,
            "sample_rate": sr,
            "sfps": sfps,
        },
    })

    # --- 7. Align to the noise reference (sliding/scraping/rolling) -
    if sp.phase_align_enabled:
        steps.append({
            "type": "force_modal_phase_align",
            "params": {
                # ``modal_ref`` is resolved by StorageEngine._inject_custom
                # (see _sidechain_for below).
                "reference_key": "modal_ref",
                "max_delay_ms": 5.0,
                "window_ms": 20.0,
                "event_threshold": 0.2,
                "target_contact_types": (1, 2, 3, 5),   # IMPACT/SCRAPING/SLIDING/MIXED
                "sample_rate": sr,
                "sfps": sfps,
            },
        })

    # --- 8. Final level match ----------------------------------------
    steps.append({
        "type": "force_modal_dynamic_amplify",
        "params": {
            "target_rms": DEFAULT_TRACK_RMS.get(track_name, sp.target_rms),
            "target_rms_per_kg": 0.05,
            "max_gain_db": sp.max_gain_db,
            "smooth_ms": sp.smoothing_window_ms or 5.0,
            "sample_rate": sr,
            "sfps": sfps,
        },
    })

    # --- 9. Peak ceiling ---------------------------------------------
    if sp.normalize_output:
        steps.append({
            "type": "per_track_normalize",
            "params": {"target_peak": 0.95},
        })

    return {
        "op": "sequential",
        "processors": steps,
        # NOTE: entity/entity_idx are filled in per-object in
        # ``_chain_for`` and the modal_ref sibling is added there too.
    }


def _build_noise_track_chain(track_name: str, cfg: Any) -> Dict[str, Any]:
    """
    Chain for AUDIO_SIGNAL / AUDIO_FORCE tracks (sliding, scraping,
    rolling).

    These tracks are acoustic noise. They do NOT go through the modal
    calibrator / equalizer / compressor family because their absolute
    scale and spectral content are already governed by the noise
    synthesizer.
    """
    sp = cfg.synth_postprocess
    sr = cfg.system.sample_rate
    sfps = (cfg.system.fps / cfg.system.fps_base) * cfg.system.subframes

    steps: List[Dict[str, Any]] = []

    # --- 1. Force-driven adaptive smoothing --------------------------
    if sp.smoothing_enabled:
        steps.append({
            "type": "force_adaptive_smoothing",
            "params": {
                "alpha_min": 0.1,
                "alpha_max": 0.9 if sp.adaptive_smoothing else 0.5,
                "smooth_ms": sp.smoothing_window_ms or 2.0,
                "sample_rate": sr,
            },
        })

    # --- 2. Phase alignment to the impact track ---------------------
    # Noise events should not precede the rigid-body impact they
    # accompany. The AUDIO_FORCE variant of force_phase_align takes
    # the impact force from props.custom["force"] directly.
    if sp.phase_align_enabled:
        steps.append({
            "type": "force_phase_align",
            "params": {
                "crossfade_samples": sp.crossfade_samples,
                "sample_rate": sr,
            },
        })

    # --- 3. Force-driven amplification ------------------------------
    steps.append({
        "type": "force_dynamic_amplify",
        "params": {
            "target_rms": DEFAULT_TRACK_RMS.get(track_name, sp.target_rms),
            "max_gain_db": sp.max_gain_db,
            "dynamic_range_compression": sp.dynamic_range_compression,
            "sample_rate": sr,
        },
    })

    # --- 4. Peak ceiling --------------------------------------------
    if sp.normalize_output:
        steps.append({
            "type": "per_track_normalize",
            "params": {"target_peak": 0.95},
        })

    return {
        "op": "sequential",
        "processors": steps,
        # entity/entity_idx filled per object.
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
@dataclass
class postProcess:
    """
    Write non-destructive ``chains`` metadata for rigidBodyEngine tracks.

    Responsibilities
    ----------------
    * Decide *which* processors apply to *which* signal.
    * Write that decision into the AudioDataType of each track / signal
      via ``StorageEngine.set_chains()``.

    It does NOT load audio or materialise any sample. StorageEngine owns
    the DAG evaluation at read time.
    """

    entity_manager: EntityManager
    storage_engine: Optional[StorageEngine] = None

    # Per-object chain overrides. Keys are obj_idx, values are
    # dict[track_name, chains].
    overrides: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # Restrict which tracks are processed.
    tracks_to_process: List[str] = field(
        default_factory=lambda: list(DEFAULT_TRACKS_TO_POST)
    )

    # Optional per-object disabling. When a track name is absent
    # from this set for a given object, ``apply_for_object`` skips it.
    object_track_filter: Optional[Dict[int, List[str]]] = None

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
        Write chains metadata for every dynamic object's tracks.

        Returns ``{obj_name: patched_track_count}``.
        """
        results: Dict[str, int] = {}
        for obj in self.config.objects:
            name = getattr(obj, "name", str(obj.idx))
            results[name] = self.apply_for_object(obj_idx=obj.idx, obj_name=name)
        return results

    def apply_for_object(self, obj_idx: int, obj_name: Optional[str] = None) -> int:
        """
        Write chains for one object's tracks.

        Returns the number of tracks that were patched.
        """
        cfg_obj = self._find_config_obj(obj_idx)
        if cfg_obj is None:
            debug_print(f"postProcess: no ObjectConfig for obj_idx={obj_idx}")
            return 0

        allowed = None
        if self.object_track_filter is not None:
            allowed = self.object_track_filter.get(obj_idx, self.tracks_to_process)

        patched = 0
        for track_name in self.tracks_to_process:
            if allowed is not None and track_name not in allowed:
                continue

            if not self.storage_engine._signal_node_exists(
                self.collection, self.engine_name, obj_idx, track_name
            ):
                debug_print(
                    f"postProcess: {self.collection}/{self.engine_name}/"
                    f"{obj_idx}/{track_name} not materialized — skipped"
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
    def _chain_for(self, obj_idx: int, track_name: str) -> Optional[Dict[str, Any]]:
        # Caller-supplied override wins.
        if obj_idx in self.overrides and track_name in self.overrides[obj_idx]:
            chain = dict(self.overrides[obj_idx][track_name])
            self._attach_sidechain(chain, obj_idx)
            return chain

        if track_name in MODAL_TRACKS:
            chain = _build_modal_track_chain(track_name, self.config)
        elif track_name in NOISE_TRACKS:
            chain = _build_noise_track_chain(track_name, self.config)
        else:
            return None

        self._attach_sidechain(chain, obj_idx)
        return chain

    # ------------------------------------------------------------------
    def _attach_sidechain(self, chain: Dict[str, Any], obj_idx: int) -> None:
        """
        Fill in the per-object side-channel references on a chain dict.

        We set:

          * ``props.custom.entity`` / ``entity_idx``
                -> StorageEngine injects the ForceDataSequence here.

          * ``props.custom.obj_config`` / ``objs_type`` / ``entity_idx``
                -> StorageEngine injectss the ObjectConfig here (used by
                   RadiationCalibrator, Equalizer, DynamicAmplify, DCBlocker).

        For MODAL tracks we also set the sibling reference that the
        phase-align step uses to align to the noise track of the *same*
        object:

          * ``props.custom.sibling_track`` = "sliding" (preferred) or
            "scraping"/"rolling", whichever exists.
        """
        custom = chain.setdefault("props", {}).setdefault("custom", {})

        # Always inject the object's ForceDataSequence.
        custom.setdefault("entity", "forces")
        custom["entity_idx"] = obj_idx

        # Always inject the object's ObjectConfig (size/mass/shader).
        custom.setdefault("obj_config", "obj_config")
        custom.setdefault("objs_type", "objects")

        # For MODAL tracks, prefer aligning to a noise sibling if one
        # exists.
        if custom.get("sibling_track") is None:
            for candidate in ("sliding", "scraping", "rolling"):
                if self.storage_engine._signal_node_exists(self.collection, self.engine_name, obj_idx, candidate):
                    custom["sibling_track"] = candidate
                    custom["key"] = "modal_ref"
                    custom["engine"] = self.engine_name
                    custom["collection"] = self.collection
                    break

    # ------------------------------------------------------------------
    # Internal: storage mutation
    # ------------------------------------------------------------------
    def _find_config_obj(self, obj_idx: int):
        for obj in self.config.objects:
            if obj.idx == obj_idx:
                return obj
        return None

    def _write_chain_to_storage(self, obj_idx: int, track_name: str, chain: Dict[str, Any]) -> int:
        """
        Attach ``chain`` to every signal of the track via StorageEngine.
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
                f"{self.collection}/{self.engine_name}/{obj_idx}/"
                f"{track_name}: {e}"
            )
            return 0

        if updated <= 1:
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
