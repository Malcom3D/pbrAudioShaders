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
import time
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SampleCounter:
    total_samples: int = None
    current_sample: int = 0
    num_players: int = 0
    players_ready: int = 0
    players_done: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(default_factory=threading.Condition)
    players_registered: List[int] = field(default_factory=list)
    
    def register_player(self, player_id: int) -> None:
        """Register a ModalPlayer instance."""
        with self.lock:
            if player_id not in self.players_registered:
                self.players_registered.append(player_id)
                self.num_players += 1
                print(f"Player {player_id} registered. Total players: {self.num_players}")
    
    def unregister_player(self, player_id: int) -> None:
        """Unregister a ModalPlayer instance."""
        with self.lock:
            if player_id in self.players_registered:
                self.players_registered.remove(player_id)
                self.num_players -= 1
                print(f"Player {player_id} unregistered. Total players: {self.num_players}")
    
    def get_current(self) -> int:
        """Get the current sample index."""
        with self.lock:
            return self.current_sample
    
    def next(self) -> int:
        """
        Increment to the next sample when all players are ready.
        Returns the new current sample index.
        """
        with self.condition:
            # Mark this player as ready to advance
            self.players_ready += 1
            
            # Wait until all players are ready
            while self.players_ready < self.num_players and self.num_players > 0:
                self.condition.wait()
            
            # If we're the last player to become ready
            if self.players_ready == self.num_players:
                # Increment the sample counter
                if self.current_sample < self.total_samples - 1:
                    self.current_sample += 1
                
                # Reset ready counter
                self.players_ready = 0
                
                # Notify all waiting players
                self.condition.notify_all()
            
            return self.current_sample
    
    def wait_for_all_players(self) -> None:
        """Wait until all players have processed the current sample."""
        with self.condition:
            self.players_done += 1
            
            # Wait until all players are done
            while self.players_done < self.num_players and self.num_players > 0:
                self.condition.wait()
            
            # If we're the last player to finish
            if self.players_done == self.num_players:
                # Reset done counter
                self.players_done = 0
                # Notify all waiting players
                self.condition.notify_all()
    
    def reset(self(self) -> None:
        """Reset the sample counter."""
        with self.lock:
            self.current_sample = 0
            self.players_ready = 0
            self.players_done = 0
    
    def set_total_samples(self, total_samples: int) -> None:
        """Set the total number of samples."""
        with self.lock:
            self.total_samples = total_samples
            print(f"Total samples set to: {total_samples}")
    
    def is_finished(self) -> bool:
        """Check if all samples have been processed."""
        with self.lock:
            return self.current_sample >= self.total_samples - 1 if self.total_samples else False
    
    def get_progress(self) -> float:
        """Get progress as a percentage."""
        with self.lock:
            if self.total_samples and self.total_samples > 0:
                return (self.current_sample / self.total_samples) * 100.0
            return 0.0
