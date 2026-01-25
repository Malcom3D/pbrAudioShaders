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
import soundfile as sf
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..core.collision_solver import CollisionSolver
from ..lib.collision_data import CollisionArea, CollisionData
from ..lib.modal_oscillator import ModalBank

from ..lib.functions import _parse_lib

@dataclass
class ModalSynth:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.collision_solver = CollisionSolver(self.entity_manager)
        self.dsp_path = f"{config.system.cache_path}/dsp"
        self.output_dir = f"{config.system.cache_path}/modal-synth"
        os.makedirs(self.output_dir, exist_ok=True)

        # Coupling buffers
        self.coupling_strength = 0.1
        self.coupling_buffer = 0.0  # From bank2 to bank1
        self.other_coupling_buffer = 0.0  # From bank1 to bank2

    def compute(self, obj_idx: int) -> None:
        config = self.entity_manager.get('config')
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds

        trajectories = self.entity_manager.get('trajectories')
        forces = self.entity_manager.get('forces')
        collision_data = self.entity_manager.get('collisions')
        collisions = []
        for conf_obj in config.objects:
            if conf_obj.idx == obj_idx:
                config_obj = conf_obj
                if not config_obj.static:
                    for t_idx in trajectories.keys():
                        if trajectories[t_idx].obj_idx == obj_idx:
                            trajectory = trajectories[t_idx]
                    for f_idx in forces.keys():
                        if forces[f_idx].obj_idx == obj_idx:
                            force = forces[f_idx]
                    for c_idx in collision_data.keys():
                        if collision_data[c_idx].obj1_idx == obj_idx or collision_data[c_idx].obj2_idx == obj_idx:
                            collisions.append(collision_data[c_idx])

        # Calculate total duration in samples
        frames = force.frames
        total_samples = int(trajectory.get_x()[-1])

        # Load audio-force tracks
        track_dir = f"{config.system.cache_path}/audio-force"
        obj_name = config_obj.name
        tracks = os.listdir(track_dir)
        for track in tracks:
            if f"{obj_name}_impact.raw" in track:
                impact_track = np.fromfile(f"{track_dir}/{track}", dtype=np.float64)
            if f"{obj_name}_non_collision.raw" in track:
                non_collision_track = np.fromfile(f"{track_dir}/{track}", dtype=np.float64)
            if f"{obj_name}_rolling.raw" in track:
                rolling_track = np.fromfile(f"{track_dir}/{track}", dtype=np.float64)
            if f"{obj_name}_sliding.raw" in track:
                sliding_track = np.fromfile(f"{track_dir}/{track}", dtype=np.float64)
            if f"{obj_name}_scraping.raw" in track:
                scraping_track = np.fromfile(f"{track_dir}/{track}", dtype=np.float64)

        audio_forces = {
            'impact': impact_track,
            'sliding': sliding_track,
            'scraping': scraping_track,
            'rolling': rolling_track,
            'non_collision': non_collision_track
        }

        synth_tracks = self._create_empty_tracks(total_samples)
        for sample_idx in frames:
#            # Synthesize non-collision forces (air resistance, etc etc.)
#            frame_samples = self._synthesize_non_collision(force, config_obj, sample_idx, total_samples, sample_rate)
            # Check if this frame contains a collision
            for collision in collisions:
                if collision.frame == sample_idx:
                    other_obj_idx = collision.obj2_idx if collision.obj1_idx == obj_idx else collision.obj1_idx
                    for conf_obj in config.objects:
                        if conf_obj.idx == other_obj_idx:
                            other_config_obj = conf_obj
                    for t_idx in trajectories.keys():
                        if trajectories[t_idx].obj_idx == other_obj_idx:
                            other_trajectory = trajectories[t_idx]

                    # Synthesize sound using audio-force tracks
                    synth_samples = self._synthesize_sound(trajectory, other_trajectory, audio_forces, collision, config_obj, other_config_obj, sample_idx, total_samples, sample_rate)
                    for key in synth_samples.keys():
                        synth_tracks[key] += synth_samples[key]

        self._save_tracks(config_obj, synth_tracks, total_samples, sample_rate)

    def _synthesize_sound(self, trajectory: Any, other_trajectory: Any, audio_forces: Dict[str, Any], collision: CollisionData, config_obj: Any, other_config_obj: Any, sample_idx: float, total_samples: int, sample_rate: int) -> Dict[str, Any]:
        """Synthesize sound using audio-force track to drive modal oscillators""" 

        start_samples = int(sample_idx - collision.impulse_range)
        stop_samples = int(sample_idx + collision.frame_range * 1.2)

        if collision.type.value == 'impact':
            stop_samples = int(sample_idx + collision.frame_range * 2)
        if not stop_samples <= total_samples:
            stop_samples = total_samples

        synth_tracks = self._create_empty_tracks(total_samples)

        collisions_area = []
        for idx in range(start_samples, stop_samples):
            if not audio_forces['impact'][idx] == 0 or not audio_forces['sliding'][idx] == 0 or not audio_forces['scraping'][idx] == 0 or not audio_forces['rolling'][idx] == 0 or not audio_forces['non_collision'][idx] == 0 or not self.other_coupling_buffer == 0 or not self.coupling_buffer == 0:

                collision_area = self.collision_solver.get_facing_face([config_obj.idx, other_config_obj.idx], idx)
                collisions_area.append(collision_area[0])
                print(collision_area)
                other_collision_area = collision_area[0][1][1].faces_idx
                collision_area = collision_area[0][1][0].faces_idx

                banks_output = 0
                if collision_area.shape[0] > 0:
                    mesh_faces = trajectory.get_faces()
                    collision_v_idx = np.unique(mesh_faces[collision_area])
                    modal_data = _parse_lib(f"{self.dsp_path}/{config_obj.name}.lib")

                    for v_idx in collision_v_idx:
                        banks = ModalBank(frequencies=modal_data['frequencies'], gains=modal_data['gains'][v_idx], t60s=modal_data['t60s'], sample_rate=sample_rate)
                        for track in synth_tracks.keys():
                            banks_input = (audio_forces[track][idx] / collision_v_idx.shape[0]) + self.coupling_buffer * self.coupling_strength
                            banks_output += banks.process(banks_input)
                            synth_tracks[track][idx] += banks_output

                other_banks_output = 0
                if other_collision_area.shape[0] > 0:
                    other_mesh_faces = other_trajectory.get_faces()
                    other_collision_v_idx = np.unique(other_mesh_faces[other_collision_area])
                    other_modal_data = _parse_lib(f"{self.dsp_path}/{other_config_obj.name}.lib")

                    for v_idx in other_collision_v_idx:
                        other_banks = ModalBank(frequencies=other_modal_data['frequencies'], gains=other_modal_data['gains'][v_idx], t60s=other_modal_data['t60s'], sample_rate=sample_rate)
                        for track in synth_tracks.keys():
                            other_banks_input = (audio_forces[track][idx] / other_collision_v_idx.shape[0]) + self.other_coupling_buffer * self.coupling_strength
                            other_banks_output += banks.process(other_banks_input)
                            synth_tracks[track][idx] += other_banks_output

                self.other_coupling_buffer = banks_output
                self.coupling_buffer = other_banks_output

        collision.add_area('collision_area', collisions_area)
        return synth_tracks 

    def _create_empty_tracks(self, total_samples: int) -> Dict:
        """Create empty tracks for silent sections."""
        result = {
            'impact': np.zeros(total_samples),
            'sliding': np.zeros(total_samples),
            'scraping': np.zeros(total_samples),
            'rolling': np.zeros(total_samples),
            'non_collision': np.zeros(total_samples)
        }

        return result

    def _save_tracks(self, config_obj: Any, tracks: Dict[str, np.ndarray], total_samples: int, sample_rate: int):
        """
        Save individual tracks as WAV files.
        Create a json multitrack project file (e.g., for Reaper, Ardour).
        """
        project_data = {
            'object_name': config_obj.name,
            'sample_rate': sample_rate,
            'duration': total_samples / sample_rate,
            'tracks': []
        }

        for track_name, track_data in tracks.items():
#            npz_file = f"{self.output_dir}/{config_obj.name}_{track_name}.npz"
#            np.savez_compressed(npz_file, track_data)
            track_file = f"{config_obj.name}_{track_name}.raw"
            wave_file = f"{self.output_dir}/{track_file}"
            sf.write(wave_file, track_data, sample_rate, subtype='DOUBLE')
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

        print(f"Created multitrack project: {json_file}")
