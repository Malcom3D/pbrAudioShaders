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

@dataclass
class ModalPlayer:
    entity_manager: EntityManager
    obj_idx: int
    end_idx: int = 0
    score: List = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_counter = self.entity_manager.get('sample_counter')
        self.connected_buffer = self.entity_manager.get('connected_buffer')
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
                t60_samples = int(t60_obj * spsf)

                self.rigidbody_synth = self.entity_manager.get('rigidbody_synth')[self.obj_idx]
                items = os.listdir(self.score_path)
                items = [x for x in items if x.startswith(f"{config_obj.name}") and x.endswith('.npz')]
                filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))

                for idx in range(len(filenames)):
                    filename = filenames[idx].strip('.npz').split('_')
                    start_idx = int(filename[-2])
                    stop_idx = int(filename[-1])
                    self.end_idx = stop_idx if self.end_idx < stop_idx else self.end_idx
                    score_data = np.load(f"{self.score_path}/{filenames[idx]}")
                    score_data.allow_pickle = True
                    score = score_data[score_data.files[0]]
                    self.score.append([start_idx, stop_idx, score])
                self.end_idx = self.end_idx + t60_samples

    def compute(self) -> None:

        sample_idx = self.sample_counter.get_current()
        while sample_idx != self.end_idx:
            active_score = []
            for score_idx in range(len(self.score)):
                if self.score[score_idx][0] <= sample_idx <= self.score[score_idx][1]:
                   active_score.append([self.score[score_idx][2][sample_idx]])
                elif self.score[score_idx][1] <= sample_idx <= self.score[score_idx][1] + t60_samples:
                   active_score.append([np.array([]), 0, self.score[score_idx][2][self.score[score_idx][1]][2]])

            banks_output = 0
            if len(active_score) > 1:
                old_banks_state = self.rigidbody_synth.get_banks_state()
                new_banks_state = [[np.array([])] for i in range(len(old_banks_state))]
                for idx in range(len(active_score)):
                    banks_output += self.rigidbody_synth.process(sample_idx, active_score[idx][0], active_score[idx][1], active_score[idx][2])
                    banks_state = self.rigidbody_synth.get_banks_state()
                    for banks_idx in range(len(banks_state)):
                        new_banks_state[banks_idx][0] = (new_banks_state[banks_idx][0] + banks_state[banks_idx][0]) # ToBe halfed?
                        new_banks_state[banks_idx][1] = (new_banks_state[banks_idx][1] + banks_state[banks_idx][1])
                        new_banks_state[banks_idx][2] = (new_banks_state[banks_idx][2] + banks_state[banks_idx][2])
                        new_banks_state[banks_idx][3] = (new_banks_state[banks_idx][3] + banks_state[banks_idx][3])
                    self.rigidbody_synth.set_banks_state(old_banks_state)
                self.rigidbody_synth.set_banks_state(new_banks_state)
            elif len(active_score) == 1:
                banks_output = self.rigidbody_synth.process(sample_idx, active_score[0], active_score[1], active_score[2])

            self.synth_track[sample_idx] += banks_output
            print(sample_idx, self.obj_idx, 'banks_output: ', banks_output)

            # Signal that we're done with this sample
            self.sample_counter.wait_for_all_players()
            
            # Get next sample (waits for all players to be ready)
            sample_idx = self.sample_counter.next()
        
        # Unregister when done
        self.sample_counter.unregister_player(self.player_id)
        print(f""Player {self.player_id} finished processing")

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
