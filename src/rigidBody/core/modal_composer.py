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
import blosc2
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import ScoreEvent, ScoreTrack
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix
from pbrAudioCommon import _adjust_for_fracture_shard

from pbrAudioStorage import StorageEngine # <-- Import StorageEngine

@dataclass
class ModalComposer:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        # Initialize the StorageEngine
        self.storage_engine = StorageEngine(self.entity_manager)

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')

        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = (fps / fps_base) * subframes

        score_track_final, score_track = (None for _ in range(2))
        score_tracks = self.entity_manager.get('score_tracks')
        for idx in score_tracks.keys():
            if score_tracks[idx].obj_idx == obj_idx and not score_tracks[idx].is_final:
                score_track = score_tracks[idx]
            elif score_tracks[idx].obj_idx == obj_idx and score_tracks[idx].is_final:
                score_track_final = score_tracks[idx]

        if score_track is None:
            # object do not collide (static?)
            return

        total_samples = score_track_final.total_samples

        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                # Use the new storage-based method
                force, coupling_strength = self._load_audioforce_tracks(obj_idx=config_obj.idx)

        # configure blosc2 compression
        cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, typesize=1, clevel=1, nthreads=8)
        dparams = blosc2.DParams(nthreads=16)

        for event_track in score_track.events:
            obj2_idx = event_track.coll_obj

            # init zeros array
            final_type = np.zeros_like(event_track.type)
            final_vertex_ids = np.full(event_track.vertex_ids.shape, np.bool_(False), dtype=np.bool_)
            final_contact_area = np.zeros_like(event_track.contact_area)
            final_force = np.zeros_like(coupling_strength)
            final_coupling_data = np.zeros_like(coupling_strength)

            mixed_mask = event_track.type == 5
            for event_type in range(1, 5):

                # score_track_final
                n_vertex_ids = np.count_nonzero(event_track.vertex_ids, axis=1)
                type_mask = event_track.type == event_type

                final_type[type_mask] = event_type
                final_coupling_data[type_mask] = coupling_strength[type_mask]
                final_contact_area[type_mask] = event_track.contact_area[type_mask]
                final_vertex_ids[type_mask.reshape(-1,)] = event_track.vertex_ids[type_mask.reshape(-1,)]
                final_force[type_mask] = np.divide(force[event_type][type_mask], n_vertex_ids[type_mask.reshape(-1,)], out=np.zeros_like(force[event_type][type_mask]), where=n_vertex_ids[type_mask.reshape(-1,)] != 0)

                # Rewrite final*[mixed_mask] with type == 2
                if event_type == 2:
                    final_type[mixed_mask] = event_type
                    final_coupling_data[mixed_mask] = coupling_strength[mixed_mask]
                    final_contact_area[mixed_mask] = event_track.contact_area[mixed_mask]
                    final_vertex_ids[mixed_mask.reshape(-1,)] = event_track.vertex_ids[mixed_mask.reshape(-1,)]
                    final_force[mixed_mask] = np.divide(force[event_type][mixed_mask], n_vertex_ids[mixed_mask.reshape(-1,)], out=np.zeros_like(force[event_type][mixed_mask]), where=n_vertex_ids[mixed_mask.reshape(-1,)] != 0)

            final_vertex_ids = blosc2.asarray(final_vertex_ids, cparams=cparams, dparams=dparams)
            score_track_final.add_event(ScoreEvent(coll_obj=obj2_idx, start_sample=event_track.start_sample, stop_sample=event_track.stop_sample, type=final_type, vertex_ids=final_vertex_ids, contact_area=final_contact_area, force=final_force, coupling_data=final_coupling_data))

            # Add event type 3 and 4 to complete mixed event
            # init zeros array
            final_type = np.zeros_like(event_track.type)
            final_vertex_ids = np.full(event_track.vertex_ids.shape, np.bool_(False), dtype=np.bool_)
            final_contact_area = np.zeros_like(event_track.contact_area)
            final_force = np.zeros_like(coupling_strength)
            final_coupling_data = np.zeros_like(coupling_strength)
            for event_type in [3, 4]:
                final_type[mixed_mask] = event_type
                final_coupling_data[mixed_mask] = coupling_strength[mixed_mask]
                final_contact_area[mixed_mask] = event_track.contact_area[mixed_mask]
                final_vertex_ids[mixed_mask.reshape(-1,)] = event_track.vertex_ids[mixed_mask.reshape(-1,)]
                final_force[mixed_mask] = np.divide(force[event_type][mixed_mask], n_vertex_ids[mixed_mask.reshape(-1,)], out=np.zeros_like(force[event_type][mixed_mask]), where=n_vertex_ids[mixed_mask.reshape(-1,)] != 0)

            final_vertex_ids = blosc2.asarray(final_vertex_ids, cparams=cparams, dparams=dparams)
            score_track_final.add_event(ScoreEvent(coll_obj=obj2_idx, start_sample=event_track.start_sample, stop_sample=event_track.stop_sample, type=final_type, vertex_ids=final_vertex_ids, contact_area=final_contact_area, force=final_force, coupling_data=final_coupling_data))

    def _load_audioforce_tracks(self, obj_idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load audio-force tracks for an object from the StorageEngine.
        """
        # Define track names to load, in an order that maps to contact types
        # Index: 0=non_collision, 1=impact, 2=scraping, 3=sliding, 4=rolling
        config = self.entity_manager.get('config')
        track_names = ['non_collision', 'impact', 'scraping', 'sliding', 'rolling', 'coupling_strength']
        
        # Initialize a dictionary to hold the loaded tracks
        loaded_tracks = {}

        debug_print(f"Loading audio-force tracks for obj_idx {obj_idx} from storage...")
        for track_name in track_names:
            try:
                # The read method returns a (1, N) array, so we flatten it.
                track_data = self.storage_engine.read(collection=config.system.collection,  engine='physicsSolver', obj_idx=obj_idx, track_name=track_name)
                if track_data is not None and np.asarray(track_data).size > 0:
                    loaded_tracks[track_name] = track_data.flatten().reshape(1,-1)
                else:
                    # If a track is not found, create a silent one.
                    # We need a length. We can get it from any other loaded track.
                    # If no tracks are loaded yet, we'll handle it later.
                    debug_print(f"Could not load track '{track_name}' for obj {obj_idx}: use a silent one")
                    loaded_tracks[track_name] = np.zeros((1,total_samples), dtype=np.float32)
            except Exception as e:
                debug_print(f"Could not load track '{track_name}' for obj {obj_idx}: {e}")
                loaded_tracks[track_name] = np.zeros((1,total_samples), dtype=np.float32)

        # Determine the total length from the first successfully loaded track
        total_samples = next((t.shape[0] for t in loaded_tracks.values() if t is not None), 0)
        
        # If no tracks were loaded, we can't proceed.
        if total_samples == 0:
            debug_print(f"No audio-force tracks found for obj {obj_idx}. Returning empty arrays.")
            # Return a list of empty arrays and an empty coupling array
            return [np.zeros(1) for _ in range(5)], np.zeros(1)

        # Fill in any missing tracks with zeros
        for track_name, track_data in loaded_tracks.items():
            if track_data is None:
                loaded_tracks[track_name] = np.zeros(total_samples, dtype=np.float32)

        # Assemble the final lists and arrays
        # The order must match the indexing used in the 'compute' method:
        # [non_collision, impact, scraping, sliding, rolling]
        force_tracks = [
            loaded_tracks.get('non_collision'),
            loaded_tracks.get('impact'),
            loaded_tracks.get('scraping'),
            loaded_tracks.get('sliding'),
            loaded_tracks.get('rolling'),
        ]
        
        coupling_strength_track = loaded_tracks.get('coupling_strength')

        # Reshape to (N, 1) to match the expected format in the compute method
        force_tracks = [track.reshape(-1, 1) for track in force_tracks]
        coupling_strength_track = coupling_strength_track.reshape(-1, 1)

        return force_tracks, coupling_strength_track
