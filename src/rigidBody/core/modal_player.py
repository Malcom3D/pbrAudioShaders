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
import soundfile as sf
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _parse_lib
from ..lib.filter import LinkwitzRileyFilter

@dataclass
class ModalPlayer:
    entity_manager: EntityManager
    obj_idx: int
    begin_idx: int = 0
    end_idx: int = 0
    player_id: int = None
    score: List = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_counter = self.entity_manager.get('sample_counter')
        self.connected_buffer = self.entity_manager.get('connected_buffer')
        self.connected_buffer.set_total_samples(self.sample_counter.total_samples)
        self.begin_idx = self.sample_counter.total_samples
        self.score_path = f"{config.system.cache_path}/score"
        self.output_dir = f"{config.system.cache_path}/modal_player"
        os.makedirs(self.output_dir, exist_ok=True)

        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
        spsf = sample_rate / sfps # samples per subframe

        # Generate unique player ID if not provided
        if self.player_id is None:
            self.player_id = id(self)
        
        # Register with sample counter
        self.sample_counter.register_player(self.player_id)

        self.synth_track = np.zeros(self.sample_counter.total_samples)
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj
                # Get modal model T60 times for both objects
                t60_obj = self._get_modal_t60(config_obj)
                # Calculate samples based on T60 (reverberation time)
                # We want to capture the full decay of the modal response
                self.t60_samples = int(t60_obj * spsf)

                sample_indices = []
                self.rigidbody_synth = self.entity_manager.get('rigidbody_synth')[self.obj_idx]
                score_tracks = self.entity_manager.get('score_tracks')
                for idx in score_tracks.keys():
                    if score_tracks[idx].obj_idx == self.obj_idx:
                        self.score.append(score_tracks[idx])
                        for event in score_tracks[idx].events:
                            sample_indices.append(event.sample_idx)
                self.begin_idx = min(sample_indices)
                end_idx = max(sample_indices) + self.t60_samples
                self.end_idx = end_idx if end_idx < self.sample_counter.total_samples else self.sample_counter.total_samples

    def compute(self) -> None:
        sample_idx = self.sample_counter.get_current()
        while not sample_idx == self.end_idx - 1:
            banks_output = 0
            if self.begin_idx <= sample_idx:
                events = []
                for score_idx in range(len(self.score)):
                    events += self.score[score_idx].get_events_at_sample(sample_idx)

                if len(events) > 1:
                    old_banks_state = self.rigidbody_synth.get_banks_state()
                    new_banks_state = []
                    for i in range(len(old_banks_state)):
                        new_state = 0
                        if not isinstance(old_banks_state[i], int):
                            new_state = []
                            for l in range(len(old_banks_state[i])):
                                new_state += [np.zeros_like(old_banks_state[i][l])]
                        new_banks_state += [new_state]
                    for idx in range(len(events)):
                        event = events[idx].to_dict()
                        banks_output += self.rigidbody_synth.process(event['sample_idx'], event['vertex_ids'], event['force'], event['coupling_data'])
                        print(sample_idx, self.obj_idx, 'MULTI: ', 'force: ', event['force'], 'banks_output: ', banks_output)
#                        banks_state = self.rigidbody_synth.get_banks_state()
#                        for banks_idx in range(len(banks_state)):
#                            if not isinstance(banks_state[banks_idx], int):
#                                if np.max(banks_state[banks_idx][0]) > 300000 or np.max(banks_state[banks_idx][1]) > 300000 or np.max(banks_state[banks_idx][2]) > 300000 or np.max(banks_state[banks_idx][3]) > 300000 or np.max(banks_state[banks_idx][4]) > 300000:
#                                    print('obj_idx: ', self.obj_idx, 'event: ', event)
#                                    print('obj_idx: ', self.obj_idx, 'banks_idx: ', banks_idx, 'banks_state: ', banks_state)
#                                    print('obj_idx: ', self.obj_idx, 'banks_idx: ', banks_idx, 'old_banks_state: ', old_banks_state)
#                                    return
#                                print(new_banks_state[banks_idx][0], banks_state[banks_idx][0])
#                                new_banks_state[banks_idx][0] = (new_banks_state[banks_idx][0] + banks_state[banks_idx][0]) / 2 # ToBe halfed?
#                                new_banks_state[banks_idx][1] = (new_banks_state[banks_idx][1] + banks_state[banks_idx][1]) / 2
#                                new_banks_state[banks_idx][2] = (new_banks_state[banks_idx][2] + banks_state[banks_idx][2])
#                                new_banks_state[banks_idx][3] = (new_banks_state[banks_idx][3] + banks_state[banks_idx][3])
#                                new_banks_state[banks_idx][4] = (new_banks_state[banks_idx][4] + banks_state[banks_idx][4])
#                        self.rigidbody_synth.set_banks_state(old_banks_state)
#                    self.rigidbody_synth.set_banks_state(new_banks_state)
#                    self.rigidbody_synth.set_banks_state(banks_state)
#                    print(sample_idx, self.obj_idx, 'MULTI: ', 'set_banks_state')
                elif len(events) == 1:
                    event = events[0].to_dict()
                    banks_output = self.rigidbody_synth.process(event['sample_idx'], event['vertex_ids'], event['force'], event['coupling_data'])
#                    print(sample_idx, 'SINGLE: ', 'force: ', event['force'], 'coupling_data: ', event['coupling_data'], 'banks_output: ', banks_output)

#            print('obj_idx: ', self.obj_idx, 'banks_output: ', type(banks_output), banks_output)
            self.synth_track[sample_idx] = banks_output if not np.isnan(banks_output) else 0
            
            # Get next sample (waits for all players to be ready)
            sample_idx = self.sample_counter.next(self.player_id)

#        # Normalize audio
#        self.synth_track = self.synth_track / np.max(self.synth_track)

        # Unregister when done
        self.sample_counter.unregister_player(self.player_id)
        print(f"Player {self.player_id} finished processing")

#        # Apply Linkwitz Riley BandPass Filter
#        config = self.entity_manager.get('config')
#        sample_rate = config.system.sample_rate
#        for conf_obj in config.objects:
#            if conf_obj.idx == self.obj_idx:
#                config_obj = conf_obj
#                low_freq = config_obj.acoustic_shader.low_frequency
#                high_freq = config_obj.acoustic_shader.high_frequency
#                self.synth_track, sample_rate = LinkwitzRileyFilter.linkwitz_riley_bandpass_filter(audio_data, sample_rate, low_freq, high_freq)

    def _get_modal_t60(self, config_obj: Any) -> float:
        """
        Get the modal model T60 (reverberation time) for an object.

        Parameters:
        -----------
        config_obj : ObjectConfig
            Object configuration

        Returns:
        --------
        float : Maximum T60 value from the modal model (in seconds)
        """
        # Get the path to the .lib file
        cache_path = self.entity_manager.get('config').system.cache_path
        lib_file = f"{cache_path}/dsp/{config_obj.name}.lib"

        # Parse the .lib file to get modal data
        modal_data = _parse_lib(lib_file)

        # Get T60 values array
        t60s = modal_data['t60s']

        # Return the maximum T60 (longest decay time)
        return float(np.max(t60s))

    def save_synth_track(self):
        """
        Save individual tracks as WAV files.
        Create a json multitrack project file (e.g., for Reaper, Ardour).
        """
        config = self.entity_manager.get('config')
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj

        sample_rate = config.system.sample_rate

        project_data = {
            'object_name': config_obj.name,
            'sample_rate': sample_rate,
            'duration': self.synth_track.shape[0] / sample_rate,
            'tracks': []
        }

        track_name = config_obj.name
        track_file = f"{track_name}.raw"
        wave_file = f"{self.output_dir}/{track_file}"
        sf.write(wave_file, self.synth_track, sample_rate, subtype='FLOAT')
        project_data['tracks'].append({
            'name': track_name,
            'file': track_file,
            'channels': 1,
            'position': 0.0,
            'volume': 1.0,
            'pan': 0.0
        })
        print(f"Saved {track_name} tracks to {self.output_dir}")

        # Save project file
        json_file = f"{self.output_dir}/{config_obj.name}.json"
        with open(json_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        print(f"Created synth track project: {json_file}")
