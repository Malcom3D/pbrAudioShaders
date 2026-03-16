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
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, field
from dask import delayed, compute
import trimesh

from physicsSolver import EntityManager, TrajectoryData, CollisionData, ForceDataSequence
from rigidBody import ModalPlayer, SampleCounter, ConnectedBuffer

from ..lib.fracture_data import FractureEvent, FractureType, FragmentData
from ..lib.fracture_modal import FractureModalModel
from ..lib.fracture_synth import FractureSynth

@dataclass
class fractureEngine:
    """Main engine for fracture sound synthesis."""
    
    entity_manager: EntityManager
    fracture_events: List[FractureEvent] = field(default_factory=list)
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.fracture_dir = f"{config.system.cache_path}/fracture"
        self.fracture_modal_dir = f"{config.system.cache_path}/fracture_modal"
        self.fracture_audio_dir = f"{config.system.cache_path}/fracture_audio"
        
        os.makedirs(self.fracture_dir, exist_ok=True)
        os.makedirs(self.fracture_modal_dir, exist_ok=True)
        os.makedirs(self.fracture_audio_dir, exist_ok=True)
        
        # Load existing fracture events if any
        self._load_fracture_events()
    
    def _load_fracture_events(self):
        """Load previously computed fracture events."""
        if os.path.exists(self.fracture_dir):
            for filename in os.listdir(self.fracture_dir):
                if filename.endswith('.pkl'):
                    event = FractureEvent.load(f"{self.fracture_dir}/{filename}")
                    self.fracture_events.append(event)
    
    def bake(self):
        config = self.entity_manager.get('config')
        for conf_obj in config.objects:
            if conf_obj.fractured:
                if isinstance(conf_obj.shard, np.ndarray):
                    original_obj = conf_obj.idx
                    fragment_indices = obj.shard.tolist()
                    _ = self.detect_fracture_events(original_obj, fragment_indices)
                    self.process_all_fractures()
        
    def detect_fracture_events(self, obj_idx: int, fragment_indices: List[int]) -> List[FractureEvent]:
        """
        Detect fracture events by analyzing the transition from original object to fragments.
        
        Parameters:
        -----------
        obj_idx : int
            Index of the original object before fracture
        fragment_indices : List[int]
            Indices of the fragments after fracture
            
        Returns:
        --------
        List[FractureEvent]
            Detected fracture events
        """
        config = self.entity_manager.get('config')
        
        # Get original object config
        original_obj = None
        for obj in config.objects:
            if obj.idx == obj_idx:
                original_obj = obj
                break
        
        if not original_obj:
            raise ValueError(f"Object {obj_idx} not found")
        
        # Get fragment objects
        fragments = []
        for frag_idx in fragment_indices:
            for obj in config.objects:
                if obj.idx == frag_idx:
                    fragments.append(obj)
                    break
        
        # Get trajectories
        trajectories = self.entity_manager.get('trajectories')
        original_trajectory = None
        fragment_trajectories = []
        
        for traj in trajectories.values():
            if isinstance(traj, TrajectoryData):
                if traj.obj_idx == obj_idx:
                    original_trajectory = traj
                elif traj.obj_idx in fragment_indices:
                    fragment_trajectories.append(traj)
        
        if not original_trajectory:
            raise ValueError(f"Trajectory for object {obj_idx} not found")
        
        # Find fracture frame where object splits into fragments
        fracture_frame = self._find_fracture_frame(original_trajectory, fragment_trajectories)
        
        if fracture_frame is None:
            # No fracture detected
            return []
        
        # Get collision data at fracture time
        collisions = self.entity_manager.get('collisions')
        fracture_collisions = []
        
        for coll in collisions.values():
            if isinstance(coll, CollisionData):
                if coll.obj1_idx == obj_idx or coll.obj2_idx == obj_idx:
                    if abs(coll.frame - fracture_frame) < 10:  # Within 10 frames of fracture
                        fracture_collisions.append(coll)
        
        # Get force data
        forces = self.entity_manager.get('forces')
        fracture_forces = []
        
        for force in forces.values():
            if isinstance(force, ForceDataSequence):
                if force.obj_idx == obj_idx or force.other_obj_idx == obj_idx:
                    if hasattr(force, 'frames') and len(force.frames) > 0:
                        if abs(force.frames[0] - fracture_frame) < 10:
                            fracture_forces.append(force)
        
        # Create fracture events for each fragment pair
        events = []
        
        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i+1:], i+1):
                # Check if these fragments separate from each other
                if self._fragments_separate(frag1.idx, frag2.idx, fracture_frame):
                    event = FractureEvent(
                        fracture_type=FractureType.SHATTER,
                        frame=fracture_frame,
                        original_obj_idx=obj_idx,
                        fragment1_idx=frag1.idx,
                        fragment2_idx=frag2.idx,
                        collisions=fracture_collisions,
                        forces=fracture_forces
                    )
                    events.append(event)
        
        # Save events
        for i, event in enumerate(events):
            event.save(f"{self.fracture_dir}/event_{i:05d}.pkl")
            self.fracture_events.append(event)
        
        return events
    
    def _find_fracture_frame(self, original_traj: TrajectoryData, fragment_trajs: List[TrajectoryData]) -> Optional[float]:
        """
        Find the frame where the original object fractures into fragments.
        
        Uses velocity change and distance between fragments to detect fracture.
        """
        # Get time range
        orig_frames = original_traj.get_x()
        
        if len(orig_frames) == 0:
            return None
        
        # Find frame where fragment trajectories begin
        fragment_start_frames = []
        for frag_traj in fragment_trajs:
            frames = frag_traj.get_x()
            if len(frames) > 0:
                fragment_start_frames.append(frames[0])
        
        if not fragment_start_frames:
            return None
        
        # Fracture frame is when fragments first appear
        fracture_frame = min(fragment_start_frames)
        
        # Verify that original object exists up to that frame
        last_orig_frame = orig_frames[-1]
        
        if last_orig_frame < fracture_frame:
            # Original disappears before fragments appear - fracture occurred
            return last_orig_frame
        
        return fracture_frame
    
    def _fragments_separate(self, frag1_idx: int, frag2_idx: int, fracture_frame: float) -> bool:
        """
        Check if two fragments separate from each other after fracture.
        """
        trajectories = self.entity_manager.get('trajectories')
        
        frag1_traj = None
        frag2_traj = None
        
        for traj in trajectories.values():
            if isinstance(traj, TrajectoryData):
                if traj.obj_idx == frag1_idx:
                    frag1_traj = traj
                elif traj.obj_idx == frag2_idx:
                    frag2_traj = traj
        
        if not frag1_traj or not frag2_traj:
            return False
        
        # Get positions just after fracture
        post_frame = fracture_frame + 1
        
        try:
            pos1 = frag1_traj.get_position(post_frame)
            pos2 = frag2_traj.get_position(post_frame)
            
            # Check distance
            distance = np.linalg.norm(pos1 - pos2)
            
            # If distance is significant, fragments separate
            return distance > 0.01  # 1 cm threshold
        except:
            return False
    
    def prebake_fracture_modal(self, event: FractureEvent, fragment_idx: int):
        """
        Pre-bake modal models for fracture fragments.
        
        This creates modified modal models based on the original object's
        modal properties and the fracture pattern.
        """
        fmm = FractureModalModel(self.entity_manager)
        fmm.compute_fragment_modal(event, fragment_idx)
    
    def bake_fracture_sound(self, event: FractureEvent):
        """
        Bake fracture sound for a fracture event.
        
        This synthesizes the sound of the fracture itself.
        """
        fs = FractureSynth(self.entity_manager)
        fs.compute(event)
    
    def process_all_fractures(self):
        """
        Process all fracture events in parallel using Dask.
        """
        # Pre-bake modal models for all fragments
        modal_tasks = []
        for event in self.fracture_events:
            # Get fragment indices
            fragments = [event.fragment1_idx, event.fragment2_idx]
            for frag_idx in fragments:
                modal_tasks.append(
                    self._delayed_prebake_fracture_modal(event, frag_idx)
                )
        
        if modal_tasks:
            compute(*modal_tasks)
        
        # Bake fracture sounds
        sound_tasks = [
            self._delayed_bake_fracture_sound(event)
            for event in self.fracture_events
        ]
        
        if sound_tasks:
            compute(*sound_tasks)
    
    @delayed
    def _delayed_prebake_fracture_modal(self, event: FractureEvent, fragment_idx: int):
        """Delayed wrapper for prebake_fracture_modal."""
        self.prebake_fracture_modal(event, fragment_idx)
    
    @delayed
    def _delayed_bake_fracture_sound(self, event: FractureEvent):
        """Delayed wrapper for bake_fracture_sound."""
        self.bake_fracture_sound(event)
