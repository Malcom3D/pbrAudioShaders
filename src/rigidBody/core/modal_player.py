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
    begin_idx: int = 0
    end_idx: int = 0
    player_id: int = None
    score: List = field(default_factory=list)

    def __post_init__(self):
        print('ModalPlayer init: ', self.obj_idx)
        config = self.entity_manager.get('config')
        print('ModalPlayer sample_counter: ', self.obj_idx)
        self.sample_counter = self.entity_manager.get('sample_counter')
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
        print('Register with sample counter', self.obj_idx)
        self.sample_counter.register_player(self.player_id)

#        self.synth_track = np.zeros(self.sample_counter.total_samples)
        self.rigidbody_synth_track = np.zeros(self.sample_counter.total_samples)
        self.resonance_synth_track = np.zeros(self.sample_counter.total_samples)
        self.sliding_synth_track = np.zeros(self.sample_counter.total_samples)
        self.scraping_synth_track = np.zeros(self.sample_counter.total_samples)
        self.rolling_synth_track = np.zeros(self.sample_counter.total_samples)
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                config_obj = conf_obj
                # Get modal model T60 times for both objects
                print('Get modal model T60 times for both objects', self.obj_idx)
                t60_obj = self._get_modal_t60(config_obj)
                # Calculate samples based on T60 (reverberation time)
                # We want to capture the full decay of the modal response
                self.t60_samples = int(t60_obj * spsf)

                sample_indices = []
                print('ModalPlayer get rigidbody_synth: ', self.obj_idx)
                self.rigidbody_synth = self.entity_manager.get('rigidbody_synth')[self.obj_idx]
                print('ModalPlayer get resonance_synth: ', self.obj_idx)
                self.resonance_synth = self.entity_manager.get('resonance_synth')[self.obj_idx]
                print('ModalPlayer get score_tracks: ', self.obj_idx)
                score_tracks = self.entity_manager.get('score_tracks')
                for idx in score_tracks.keys():
                    if score_tracks[idx].obj_idx == self.obj_idx:
                        self.score.append(score_tracks[idx])
                        for event in score_tracks[idx].events:
                            sample_indices.append(event.sample_idx)
                self.begin_idx = min(sample_indices) if not len(sample_indices) == 0 else 0
                end_idx = self.t60_samples + (max(sample_indices) if not len(sample_indices) == 0 else self.sample_counter.total_samples)
                self.end_idx = end_idx if end_idx < self.sample_counter.total_samples else self.sample_counter.total_samples
                print('ModalPlayer init end: ', self.obj_idx)

    def compute(self) -> None:
        config = self.entity_manager.get('config')
        coupling_strength = []
        for conf_obj in config.objects:
            if conf_obj.idx == self.obj_idx:
                if isinstance(conf_obj.connected, np.ndarray):
                    coupling_strength = conf_obj.connected.tolist()
                sound_path = f"{config.system.cache_path}/audio_force"
                sliding_sound, scraping_sound, rolling_sound = self._load_sound_tracks(sound_path, conf_obj.name)
                
        print('ModalPlayer compute: ', self.obj_idx)
        sample_idx = self.sample_counter.get_current()
        while not sample_idx == self.end_idx - 1:
            rigidbody_output = 0
            resonance_output = 0
            sliding_output = 0
            scraping_output = 0
            rolling_output = 0
            if self.begin_idx <= sample_idx:
                events = []
                for score_idx in range(len(self.score)):
                    events += self.score[score_idx].get_events_at_sample(sample_idx)

                if len(events) == 1:
                    event = events[0].to_dict()
                    if int(event['type']) in [2,3]:
                        #print('ModalPlayer resonance_synth.process: ', self.obj_idx, event['type'], event['force'])
                        resonance_output = self.resonance_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                        sliding_output = sliding_sound[sample_idx] * event['contact_area']
                        scraping_output = scraping_sound[sample_idx] * event['contact_area']
                    elif int(event['type']) == 4:
#                        resonance_output = self.resonance_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                        rigidbody_output = self.rigidbody_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                        rolling_output = rolling_sound[sample_idx] * event['contact_area']
                    else:
                        #print('ModalPlayer rigidbody_synth.process: ', self.obj_idx, event['type'], event['force'])
                        rigidbody_output = self.rigidbody_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                elif len(events) > 1:
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
                            if int(event['type']) in [2,3,4]:
                                #print('ModalPlayer resonance_synth.process: ', self.obj_idx, event['type'], event['force'])
                                resonance_output += self.resonance_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                                sliding_output += sliding_sound[sample_idx] * event['contact_area']
                                scraping_output += scraping_sound[sample_idx] * event['contact_area']
                                banks_state = self.resonance_synth.get_banks_state()
#                                self.resonance_synth.set_banks_state(banks_state)
                            elif int(event['type']) == 4:
#                                resonance_output += self.resonance_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                                rigidbody_output += self.rigidbody_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                                rolling_output += rolling_sound[sample_idx] * event['contact_area']
                            else:
                                #print('ModalPlayer rigidbody_synth.process: ', self.obj_idx, event['type'], event['force'])
                                rigidbody_output += self.rigidbody_synth.process(event['type'], event['vertex_ids'], event['force'], event['contact_area'], event['coupling_data'])
                                banks_state = self.rigidbody_synth.get_banks_state()
#                                for banks_idx in range(len(banks_state)):
#                                    if not isinstance(banks_state[banks_idx], int):
#                                        new_banks_state[banks_idx][0] += banks_state[banks_idx][0] - old_banks_state[banks_idx][0] # ToBe halfed?
#                                        new_banks_state[banks_idx][1] += banks_state[banks_idx][1] - old_banks_state[banks_idx][1] # ToBe halfed?
#                                self.rigidbody_synth.set_banks_state(old_banks_state)
#                                self.rigidbody_synth.set_banks_state(banks_state)
#                        self.rigidbody_synth.set_banks_state(new_banks_state)
                elif len(events) == 0:
                    for synth_type in [2,3,4]:
                        resonance_output += self.resonance_synth.process(synth_type, [], 0.0, 0, coupling_strength)
                        #print('resonance_synth.process: ', self.obj_idx, 'static', resonance_output)
                    rigidbody_output = self.rigidbody_synth.process(1, [], 0.0, 0, coupling_strength)
                    #print('rigidbody_synth.process: ', self.obj_idx, 'static', rigidbody_output)

            self.rigidbody_synth_track[sample_idx] = rigidbody_output if not np.isnan(rigidbody_output) else 0
            self.resonance_synth_track[sample_idx] = resonance_output if not np.isnan(resonance_output) else 0
            self.sliding_synth_track[sample_idx] = sliding_output if not np.isnan(sliding_output) else 0
            self.scraping_synth_track[sample_idx] = scraping_output if not np.isnan(scraping_output) else 0
            self.rolling_synth_track[sample_idx] = rolling_output if not np.isnan(rolling_output) else 0
            
            # Get next sample (waits for all players to be ready)
            sample_idx = self.sample_counter.next(self.player_id)

        # Unregister when done
        self.sample_counter.unregister_player(self.player_id)
        print(f"Player {self.player_id} finished processing")

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

    def save_synth_tracks(self):
        self.save_synth_track(self.rigidbody_synth_track, 'rigidbody')
        self.save_synth_track(self.resonance_synth_track, 'resonance')
        self.save_synth_track(self.sliding_synth_track, 'sliding')
        self.save_synth_track(self.scraping_synth_track, 'scraping')
        self.save_synth_track(self.rolling_synth_track, 'rolling')

    def save_synth_track(self, track: np.ndarray, suffix: str):
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
            'duration': track.shape[0] / sample_rate,
            'tracks': []
        }

        track_name = config_obj.name
        track_file = f"{track_name}_{suffix}.raw"
        wave_file = f"{self.output_dir}/{track_file}"
        sf.write(wave_file, track, sample_rate, subtype='FLOAT')
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

        print(f"Created {suffix} synth track project: {json_file}")

    def _load_sound_tracks(self, sound_path: str, obj_name: str):
        sliding_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)
        scraping_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)
        rolling_sound = np.zeros(self.sample_counter.total_samples, dtype=np.float32)

        if os.path.exists(f"{sound_path}/{obj_name}_sliding_sound.raw"):
            sliding_sound = np.fromfile(f"{sound_path}/{obj_name}_sliding_sound.raw", dtype=np.float32)
        if os.path.exists(f"{sound_path}/{obj_name}_scraping_sound.raw"):
            scraping_sound = np.fromfile(f"{sound_path}/{obj_name}_scraping_sound.raw", dtype=np.float32)
        if os.path.exists(f"{sound_path}/{obj_name}_rolling_sound.raw"):
            rolling_sound = np.fromfile(f"{sound_path}/{obj_name}_rolling_sound.raw", dtype=np.float32)

        return sliding_sound, scraping_sound, rolling_sound
