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

import threading
from dataclasses import dataclass, field
from typing import Optional, List

class FunctionLocker:
    def __init__(self):
        self.condition = threading.Condition()
        self.ready = False
    
    def wait_for_condition(self):
        """Wait until condition is satisfied"""
        with self.condition:
            while not self.ready:
                self.condition.wait()
    
    def signal_ready(self):
        """Signal that condition is satisfied"""
        with self.condition:
            self.ready = True
            self.condition.notify_all()
    
    def lock_function(self, func):
        """Decorator to lock function until condition is satisfied"""
        def wrapper(*args, **kwargs):
            self.wait_for_condition()
            return func(*args, **kwargs)
        return wrapper

locker = FunctionLocker()

@dataclass
class SampleCounter:
    total_samples: int = None
    current_sample: int = 0
    num_players: int = 0
    players_ready: List[int] = field(default_factory=list)
    players_registered: List[int] = field(default_factory=list)
    
    def register_player(self, player_id: int) -> None:
        """Register a ModalPlayer instance."""
        if player_id not in self.players_registered:
            self.players_registered.append(player_id)
            self.num_players += 1
            print(f"Player {player_id} registered. Total players: {self.num_players}")
    
    def unregister_player(self, player_id: int) -> None:
        """Unregister a ModalPlayer instance."""
        if player_id in self.players_registered:
            self.players_registered.remove(player_id)
            self.num_players -= 1
            print(f"Player {player_id} unregistered. Total players: {self.num_players}")
    
    def get_current(self) -> int:
        """Get the current sample index."""
        return self.current_sample
    
    def next(self, player_id: int) -> int:
        """
        Increment to the next sample when all players are ready.
        Returns the new current sample index.
        """
        if player_id in self.players_registered:
            # Mark this player as ready to advance
            self.players_ready.append(player_id)
            
            if len(self.players_ready) == self.num_players and self.num_players > 0:
            # Increment the sample counter
                if self.current_sample < self.total_samples - 1:
                    self.current_sample += 1
                    print('SampleCounter: ', self.current_sample, self.total_samples)
                # Reset ready counter
                self.players_ready = []
                locker.signal_ready() 
            else:
                self._locked_next()

        return self.current_sample

    @locker.lock_function
    def _locked_next(self):
        pass
    
    def set_total_samples(self, total_samples: int) -> None:
        """Set the total number of samples."""
        self.total_samples = total_samples
        print(f"Total samples set to: {total_samples}")
    
    def get_progress(self) -> float:
        """Get progress as a percentage."""
        if self.total_samples and self.total_samples > 0:
            return (self.current_sample / self.total_samples) * 100.0
        return 0.0
