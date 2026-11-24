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
Non-destructive chain writer for ProxySynth (ellipsoidalProxy) tracks.

Successor to ``ellipsoidalProxy.lib.proxy_eq.ProxyEqualizer`` and the
block-FFT "dynamic_eq" block inside ``ProxySynth._convolve_with_irs``.

Just like :py:mod:`pbrAudioShaders.src.rigidBody.lib.post_process`, this
class only writes ``chains`` metadata onto AudioDataType nodes in
pbrAudioStorage. The actual DSP is applied lazily at ``read()`` time via
the StorageEngine DAG evaluator (see ``pbrAudioSignals``).

Signal flow for a proxy track
-----------------------------
The ProxySynth outputs a *modal displacement* stream. The processor
chain is deliberately distinct from rigidBodyEngine:

    rigidBody  = mesh2faust + continuous geometry
    proxy      = IR-table convolution (proxy_type + size_scale)

That means network-of-resonators-style force processors and modal
calibrators from ``force_modal_*`` cannot be used here without
double-counting the size / radiation correction already baked into
the IR table. We use the four ``proxy_modal_*`` processors that were
designed explicitly to reproduce the ProxyEqualizer behaviour:

    1. proxy_modal_size_tilt        (static shelf from size_scale)
    2. proxy_modal_contact_eq       (ProxyEqualizer 7 static curves)
    3. proxy_modal_force_dynamic_eq (block-FFT dynamic boost, per sample)
    4. proxy_modal_band_gain        (modal-band summation as biquad bank)
    5. per_track_normalize          (peak ceiling)

The first three replace ``ProxyEqualizer.apply_equalization``; the
fourth reconstructs what ``_convolve_with_irs`` performed on the
frequency axis. Order matters: the band-bank goes last so it operates
on already-EQ'd signal, matching ProxySynth's per-band summation
semantics.

Compatibility notes
-------------------
* ``ProxyEQ`` used the same 7 curves for all signals, with force
  modulation only in the "rolling_sound / sliding_sound /
  scraping_sound" branches. The processor chain above maps to exactly
  those curves.
* ``ProxySynth._convolve_with_irs`` did *not* have a "sound" branch;
  it always summed the raw IR rows. The extra "_sound" tracks in
  ``ProxyEngine._register_storage_schema`` are the ones that got the
  additional colouring from ``ProxyEqualizer``.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from pbrAudioStorage import StorageEngine


# ProxySynth emits exactly these tracks (see ProxyEngine_register_storage_schema).
PROXY_TRACKS: List[str] = [
    "impact",
    "resonance",
    "sliding",
    "scraping",
    "rolling",
]

# The three sound-coloured variants only exist when enable_noise_enhancement
# is set in SystemConfig. We carry them here so the post-processor stays
# aligned with whatever the engine actually materialized.
PROXY_SOUND_TRACKS: List[str] = [
    "sliding_sound",
    "scraping_sound",
    "rolling_sound",
]


# Curve contact-type segments (matches ProxyEqualizer._init_default_curves).
# These inform the "sound" tracks which ContactType drives the extra
# colouring. Non-"sound" tracks use the primary ContactType of the same
# event family.
#
#   ContactType values (from pbrAudioCommon.lib.force_data.ContactType):
#       0 = NO_CONTACT
#       1 = IMPACT
#       2 = SCRAPING
#       3 = SLIDING
#       4 = ROLLING
#       5 = MIXED
#       6 = STATIC
_TRACK_PROFILE: Dict[str, Dict[str, Any]] = {
    "impact": {
        "contact_types": (1,),              # IMPACT
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
    },
    "resonance": {
        # Resonance is the *tail* of impact excitation. Same contact type,
        # but the ProxySynth IR already carries the decay. We therefore
        # only apply the (subtle) impact curve here, no force boost.
        "contact_types": (1,),
        "apply_contact_eq": True,
        "apply_force_eq": False,
        "eq_boost": 0.5,
    },
    "sliding": {
        "contact_types": (3,),              # SLIDING
        "apply_contact_eq": True,
        "apply_force_eq": True,
               "eq_boost": 1.0,
    },
    "scraping": {
        "contact_types": (2,),              # SCRAPING
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
    },
    "rolling": {
        "contact_types": (4,),              # ROLLING
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
    },
    "sliding_sound": {
        "contact_types": (3,),
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
        # ProxyEqualizer reversed the curve family for *_sound tracks:
        # low-band emphasis instead of high-band.
        "eq_family": "sound",
    },
    "scraping_sound": {
        "contact_types": (2,),
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
        "eq_family": "sound",
    },
    "rolling_sound": {
        "contact_types": (4,),
        "apply_contact_eq": True,
        "apply_force_eq": True,
        "eq_boost": 1.0,
        "eq_family": "sound",
    },
}


# ----------------------------------------------------------------------
# Chain builders
# ----------------------------------------------------------------------
def _build_proxy_track_chain(track_name: str, cfg: Any) -> Optional[Dict[str, Any]]:
    """
    Build the MODAL_DISPLACEMENT chain for one ProxyEngine track.

    All steps read either:
      - the current object's ForceDataSequence  (``custom["force"]``), or
      - the ObjectConfig                        (``custom["obj_config"]``)

    ``StorageEngine._get_props()`` injects those from the chain's own
    ``props.custom`` dict, so we fill that in *per object* in
    :py:meth:`postProcess._attach_sidechain`.
    """
    profile = _TRACK_PROFILE.get(track_name)
    if profile is None:
        return None

    sp = cfg.synth_postprocess
    sr = cfg.system.sample_rate
    sfps = (cfg.system.fps / cfg.system.fps_base) * cfg.system.subframes

    steps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Static size-driven shelf.
    # ------------------------------------------------------------------
    # ProxyIRTable picks an IR slice based on ``size_scale``; that slice
    # is (implicitly) a size-dependent spectral window. When a
    # non-proxy pipeline reads a proxy track this tilt restores the
    # same "small=brighter / large=darker" behaviour.
    steps.append({
        "type": "proxy_modal_size_tilt",
        "params": {
            "size_min": 0.02,
            "size_max": 2.0,
            "tilt_db_range": 4.0,
            "neutral_size": 0.2,
            "low_shelf_hz": 250.0,
            "high_shelf_hz": 3000.0,
,
            "sample_rate": sr,
        },
    })

    # ------------------------------------------------------------------
    # 2. Contact-type static EQ (ProxyEqualizer._init_default_curves).
    # ------------------------------------------------------------------
    if profile["apply_contact_eq"]:
        eq_params: Dict[str, Any] = {
            "low_shelf_hz": 250.0,
            "high_shelf_hz": 3000.0,
            "smooth_ms": 12.0,
            # The static curves in ProxyEqualizer were applied to the
            # *whole* buffer once force was known. Blend-by-force keeps
            # quiet tails from being EQ'd, matching ProxySynth, where
            # the force force-driven dynamic EQ had a 0.1 threshold before
            # any high-frequency boost.
            "blend_with_force": True,
            "sample_rate": sr,
            "sfps": sfps,
        }
        # ProxyEqualizer has a reversed family for *_sound tracks:
        # ProxyModalContactEq exposes low/high shelves; we
        # *compress* the whole curve by treating the sound tracks as
        # an additional force-coupled low-shelf.
        if profile.get("eq_family") == "sound":
            # Sound tracks are coloured with high-shelf *cut* and
            # low-shelf *boost* (mirrors the "low-frequency emphasis"
            # of ProxyEqualizer's _sound curves).
            eq_params["low_shelf_hz"] = 150.0
            eq_params["high_shelf_hz"] = 5000.0

        steps.append({
            "type": "proxy_modal_contact_eq",
            "params": eq_params,
        })

    # ------------------------------------------------------------------
    # 3. Force-driven dynamic high-shelf (ProxyEqualizer._process_
    #    with_dynamic_eq, ported to per-sample).
    # ------------------------------------------------------------------
    if profile["apply_force_eq"]:
        # ProxyEqualizer boosted frequencies above 2 kHz by up to
        # 1 + 0.5 * block_force when block_force > 0.1. The proxy
        # processor expresses this as a shelf whose dB gain follows
        # the force envelope.
        steps.append({
            "type": "proxy_modal_force_dynamic_eq",
            "params": {
                "boost_per_unit_force": 0.5 * profile["eq_boost"],
                "max_boost_db": 6.0 * profile["eq_boost"],
                "air_per_unit_force": 1.5 * profile["eq_boost"],
                "low_shelf_hz": 250.0,
                # Match ProxyEqualizer's f > 2000 Hz cutoff.
                "high_shelf_hz": 2000.0,
                "air_shelf_hz": 10000.0,
                "smooth_ms": sp.smoothing_window_ms or 5.0,
                "sample_rate": sr,
                "sfps": sfps,
            },
        })

    # ------------------------------------------------------------------
    # 4. Modal-band summation as a peaking bank.
    # ------------------------------------------------------------------
    # ProxySynth convolved the excitation with *every* IR row and then
    # summed. That is a peaking bank centred on the IR's own modal
    # frequencies. The processor reproduces that behaviour with a
    # static peaking cascade that reads ``size`` from ObjectConfig.
    #
    # Only include this for the primary tracks: the *_sound variants
    # are the noise-coloured offshoots and were NOT fed through
    # _convolve_with_irs in ProxySynth (see ProxySynth.compute: the
    # sound tracks were loaded *as input*, not generated). Applying
    # the modal-band reconstruction to them would double-colour them.
    if not track_name.endswith("_sound"):
        steps.append({
            "type": "proxy_modal_band_gain",
            "params": {
                "num_modes": 5,
                "ref_size": 0.2,
                "band_q": 6.0,
                "max_band_gain_db": 3.0,
                "mode_gain_flatness": 0.7,
                "shape_factor": 1.0,
                "sample_rate": sr,
                "sfps": sfps,
            },
        })

    # ------------------------------------------------------------------
    # 5. Peak ceiling.
    # ------------------------------------------------------------------
    if sp.normalize_output:
        steps.append({
            "type": "per_track_normalize",
            "params": {"target_peak": 0.95},
        })

    return {
        "op": "sequential",
        "processors": steps,
        # NOTE: ``entity`` / ``entity_idx`` / ``obj_config`` are filled
        # in per-object inside :py:meth:`_attach_sidechain`.
    }


# ----------------------------------------------------------------------
# postProcess
# ----------------------------------------------------------------------
@dataclass
class postProcess:
    """
    Write non-destructive ``chains`` metadata for ProxyEngine tracks.

    Successor to :class:`ellipsoidalProxy.lib.proxy_eq.ProxyEqualizer`.

    Responsibilities
    ----------------
    * Decide which processors apply to which proxy track.
    * Write that decision into the AudioDataType of each signal via
      :py:meth:`StorageEngine.set_chains`.

    It does NOT load or touch any audio. StorageEngine evaluates the
    chain lazily at ``read()`` time, cached across reads, so the same
    proxy track is bit-for-bit identical on every playback.
    """

    entity_manager: EntityManager
    storage_engine: Optional[StorageEngine] = None

    # Per-object, per-track chain overrides. Keys are ``obj_idx``, values
    # are ``{track_name: chain_dict}``.
    overrides: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # Which tracks this instance will process. Defaults to ProxyEngine's
    # output set; pass ``PROXY_TRACKS + PROXY_SOUND_TRACKS`` to cover
    # the noise-enhanced tracks.
    tracks_to_process: List[str] = field(
        default_factory=lambda: list(PROXY_TRACKS) + list(PROXY_SOUND_TRACKS)
    )

    # Optional whitelist per object: {obj_idx: [track_name, ...]}.
    object_track_filter: Optional[Dict[int, List[str]]] = None

    # The engine name used when materializing ProxyEngine tracks
    # (ProxyEngine registers itself under this name; see
    # ``ProxyEngine.__class__.__name__``).
    engine_name: str = "ProxyEngine"

    # --------------------------------------------------------------
    def __post_init__(self):
        config = self.entity_manager.get("config")

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.config = config
        self.collection = config.system.collection

        if self.storage_engine is None:
            self.storage_engine = StorageEngine(self.entity_manager)

        # Only attempt to process the sound-tracks when the engine
        # actually materialized them.
        if not getattr(config.system, "enable_noise_enhancement", False):
            self.tracks_to_process = [
                t for t in self.tracks_to_process
                if t not in PROXY_SOUND_TRACKS
            ]

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def apply(self) -> Dict[str, int]:
        """
        Write chains metadata for every proxy object's tracks.

        Returns ``{obj_name: patched_track_count}``.
        """
        results: Dict[str, int] = {}
        for obj in self.config.objects:
            # Only objects that the ProxyEngine actually synthesized.
            if getattr(obj, "proxy_type", False) is False:
                continue
            if obj.proxy_type not in (0, 1, 2, 3, 4, 5):
                continue
            # ProxySynth only runs when enable_proxy_synth is set.
            if not getattr(self.config.system, "enable_proxy_synth", False):
                continue

            name = getattr(obj, "name", str(obj.idx))
            results[name] = self.apply_for_object(obj_idx=obj.idx, obj_name=name)
        return results

    def apply_for_object(self, obj_idx: int, obj_name: Optional[str] = None) -> int:
        """
        Write chains for one object's proxy tracks.

        Returns the number of *tracks* that were patched.
        """
        cfg_obj = self._find_config_obj(obj_idx)
        if cfg_obj is None:
            debug_print(f"proxy postProcess: no ObjectConfig for obj_idx={obj_idx}")
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
                    f"proxy postProcess: {self.collection}/{self.engine_name}/"
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
                f"proxy postProcess: patched {patched} tracks for "
                f"{obj_name or obj_idx}"
            )
        return patched

    def revert_for_object(self, obj_idx: int) -> int:
        """Remove chains from an object's ProxyEngine tracks."""
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
            if getattr(obj, "proxy_type", False) is False:
                continue
            self.revert_for_object(obj.idx)

    # --------------------------------------------------------------
    # Internal: chain resolution
    # --------------------------------------------------------------
    def _chain_for(self, obj_idx: int, track_name: str) -> Optional[Dict[str, Any]]:
        # Caller-supplied override wins outright.
        if obj_idx in self.overrides and track_name in self.overrides[obj_idx]:
            chain = dict(self.overrides[obj_idx][track_name])
            self._attach_sidechain(chain, obj_idx)
            return chain

        chain = _build_proxy_track_chain(track_name, self.config)
        if chain is None:
            return None

        self._attach_sidechain(chain, obj_idx)
        return chain

    # --------------------------------------------------------------
    def _attach_sidechain(self, chain: Dict[str, Any], obj_idx: int) -> None:
        """
        Fill the per-object side-channel references into a chain dict.

        Unlike the rigidBody counterpart, no sibling-track reference is
        created here: ProxySynth tracks are independent modal
        emissions and do not need cross-track phase alignment. Adding
        one would impose an unnecessary read of the sibling track on
        every ProxyEngine evaluation.
        """
        custom = chain.setdefault("props", {}).setdefault("custom", {})

        # -------- ForceDataSequence (for contact-type & force paths) -----
        # ForceModalSidechain looks up ``props.custom["force"]`` first,
        # then falls back to ``props.custom["forces"]`` (see _inject_custom
        # in StorageEngine). Providing both keys makes the processor
        # implementations work whichever alias they prefer.
        custom.setdefault("entity", "forces")
        custom["entity_idx"] = obj_idx

        # -------- ObjectConfig (for size tilt / band gain) --------
        custom.setdefault("obj_config", "obj_config")
        custom.setdefault("objs_type", "objects")

    # --------------------------------------------------------------
    # Internal: storage mutation
    # --------------------------------------------------------------
    def _find_config_obj(self, obj_idx: int):
        for obj in self.config.objects:
            if obj.idx == obj_idx:
                return obj
        return None

    def _write_chain_to_storage(
        self, obj_idx: int, track_name: str, chain: Dict[str, Any]
    ) -> int:
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
                f"proxy postProcess: set_chains failed for "
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
                    f"proxy postProcess: chain written on track node "
                    f"{track_name} (no signals present)"
                )
        else:
            debug_print(
                f"proxy postProcess: chain written on {updated} signal node(s) "
                f f"of {track_name}"
            )
        return updated
