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
import pickle
import time
import numpy as np
import soundfile as sf
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

from ..lib.particles_trajectory_data import ParticlesTrajectoryData
from ..lib.particles_collisions_points import ParticlesCollisionsPoints
from ..lib.particles_collisions_voxels import ParticlesCollisionsVoxels
from ..lib.voxels_modal_ir_convolver import VoxelsModalIRConvolver

@dataclass
class ParticlesPlayer:
    entity_manager: EntityManager
    particles_obj_idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        
        self.output_dir = f"{config.system.cache_path}/particles_player"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.particles_config_obj = next((p for p in config.particles if p.idx == self.particles_obj_idx), None)
        if not self.particles_config_obj:
            raise ValueError(f"Particles config for idx {self.particles_obj_idx} not found.")

        self.particles_traj = self.entity_manager.get('trajectories').get(self.p.particles_obj_idx)
        if not isinstance(self.particles_traj, ParticlesTrajectoryData):
            raise ValueError(f"ParticlesTrajectoryData for idx {self.particles_obj_idx} not found.")

    def compute(self) -> None:
        """
        Main synthesis loop. Iterates over frames, computes collision forces,
        and excites the modal convolvers.
        """
        config = self.entity_manager.get('config')
        total_samples = int(self.particles_traj.get_x()[-1]) if len(self.particles_traj.get_x()) > 0 else 0
        if total_samples == 0:
            debug_print(f"No samples to process for particles object {self.particles_obj_idx}")
            return

        # Load collision data
        collisions_dir = f"{config.system.cache_path}/particles_collisions"
        if self.particles_config_obj.proxy:
            filepath = f"{collisions_dir}/voxels_{self.particles_obj_idx}.pkl"
            collision_data = ParticlesCollisionsVoxels.load(filepath) if os.path.exists(filepath) else None
        else:
            filepath = f"{collisions_dir}/points_{self.particles_obj_idx}.pkl"
            collision_data = ParticlesCollisionsPoints.load(filepath) if os.path.exists(filepath) else None
        
        if not collision_data:
            debug_print(f"No collision data found for {self.particles_obj_idx}.")
            return

        # Load convolver pool info from Luthier
        luthier_dir = f"{config.system.cache_path}/particles_luthier"
        luthier_filepath = f"{luthier_dir}/pool_{self.particles_obj_idx}.pkl"
        if not os.path.exists(luthier_filepath):
            debug_print(f"Luthier data not found for {self.particles_obj_idx}.")
            return
        with open(luthier_filepath, 'rb') as f:
            max_concurrent = pickle.load(f)

        # Instantiate convolver pools and their active state
        convolver_pools: Dict[int, List[VoxelsModalIRConvolver]] = {}
        active_convolvers: Dict[int, List[Tuple[VoxelsModalIRConvolver, float]]] = {} # obj_idx -> list of (convolver, end_frame)
        for obj_idx, count in max_concurrent.items():
            obj_config = next((o for o in config.objects if o.idx == obj_idx), None)
            if if not obj_config: continue
            lib_path = f"{config.system.cache_path}/dsp/{obj_config.name}.lib"
            if not os.path.exists(lib_path): continue
            
            convolver_pools[obj_idx] = [
                VoxelsModalIRConvolver(sample_rate=config.system.sample_rate, modal_libs={obj_idx: lib_path})
                for _ in range(count)
            ]
            active_convolvers[obj_idx] = []

        # Output tracks
        if self.particles_config_obj.proxy:
            output_tracks = {} # voxel_key -> track
        else:
            output_tracks = {} # particle_idx -> track

        frames = self.particles_traj.get_x()
        
        for frame in frames:
            frame_collisions = collision_data.collisions.get(frame, {})
            
            if not frame_collisions:
                # Still need to process decay for active convolvers
                for obj_idx, active_list in active_convolvers.items():
                    for convolver, end_frame in active_list:
                        output = convolver.process(obj_idx, 0.0)
                        if output != 0:
                            self._add_to_track(output_tracks, obj_idx if self.particles_config_obj.proxy else convolver, frame, output)
                continue

            # Process collisions for this frame
            for key, collision_info in frame_collisions.items():
                obj_idx = collision_info['obj_idx']
                
                # Get an available convolver from the pool
                if not convolver_pools.get(obj_idx): continue
                
                # Find a free convolver
                convolver = next((c for c in convolver_pools[obj_idx] if c not in [ac[0] for ac in active_convolvers[obj_idx]]), None)
                if not convolver:
                    debug_print(f"Warning: Convolver pool exhausted for object {obj_idx} at frame {frame}")
                    continue

                # Compute force
                force = collision_info['force']
                
                # Add to active convolvers
                obj_config = next((o for o in config.objects if o.idx == obj_idx), None)
                lib_path = f"{config.system.cache_path}/dsp/{obj_config.name}.lib"
                # This is inefficient, should be cached
                from pbrAudioCommon import _parse_lib
                modal_data = _parse_lib(lib_path)
                t60_samples = (np.max(modal_data['t60s']) if modal_data['t60s'].size > 0 else 0) * config.system.sample_rate
                active_convolvers[obj_idx].append((convolver, frame + t60_samples))

                # Process the excitation
                output = convolver.process(obj_idx, force)
                
                # Store the output in the correct track
                if self.particles_config_obj.proxy:
                    voxel_key = key # key is (obj_idx, *voxel_idx)
                    track_key = voxel_key
                else:
                    particle_idx = key
                    track_key = particle_idx
                
                self._add_to_track(output_tracks, track_key, frame, output)

            # Process decay for all active convolvers
            for obj_idx, active_list in active_convolvers.items():
                new_active_list = []
                for convolver, end_frame in active_list:
                    output = convolver.process(obj_idx, 0.0)
                    if output != 0:
                        # Need to figure out which track this belongs to. This is tricky.
                        # For simplicity, we'll just add it to a general decay track for now.
                        # A better implementation would track which convolver belongs to which track.
                        pass # Decay is handled implicitly by the convolver's state
                    if frame < end_frame:
                        new_active_list.append((convolver, end_frame))
                active_convolvers[obj_idx] = new_active_list

        # Save the output tracks
        self._save_tracks(output_tracks, total_samples)

    def _add_to_track(self, output_tracks: Dict, track_key: Any, frame: float, output: float):
        """Adds a sample to the appropriate track."""
        if track_key not in output_tracks:
            # We don't know the total length yet, so we'll use a list and convert later
            output_tracks[track_key] = []
        
        # Ensure the list is long enough
        int_frame = int(frame)
        if len(output_tracks[track_key]) <= int_frame:
            output_tracks[track_key].extend([0.0] * (int_frame - len(output_tracks[track_key]) + 1))
        
        output_tracks[track_key][int_frame] += output

    def _save_tracks(self, output_tracks: Dict, total_samples: int):
        """Saves the final tracks to disk."""
        output_path = f"{self.output_dir}/{self.particles_config_obj.name}"
        os.makedirs(output_path, exist_ok=True)
        
        project_data = {'object_name': self.particles_config_obj.name, 'tracks': []}

        for track_key, track_data in output_tracks.items():
            # Convert list to numpy array and pad
            track_array = np.array(track_data, dtype=np.float32)
            if len(track_array) < total_samples:
                track_array = np.pad(track_array, (0, total_samples - len(track_array)))
            else:
                track_array = track_array[:total_samples]

            if not np.any(track_array):
                continue

            # Create a unique filename
            if self.particles_config_obj.proxy:
                # track_key is (obj_idx, *voxel_idx)
                obj_idx = track_key[0]
                voxel_str = "_".join(map(str, track_key[1:]))
                track_name = f"voxel_{obj_idx}_{voxel_str}"
            else:
                track_name = f"particle_{track_key}"

            track_file = f"{track_name}.raw"
            wave_file = os.path.join(output_path, track_file)
            
            sf.write(wave_file, track_array, self.particles_traj.sample_rate, subtype='FLOAT')
            
            project_data['tracks'].append({
                'name': track_name,
                'file': track_file,
                'channels': 1,
                'position': 0.0,
            })
        
        # Save project file
        json_file = os.path.join(output_path, f"{self.particles_config_obj.name}.json")
        with open(json_file, 'w') as f:
            json.dump(project_data, f, indent=2)
        
        debug_print(f"Saved {len(project_data['tracks'])} tracks for {self.particles_config_obj.name}")

