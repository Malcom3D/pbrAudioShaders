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
import threading
from dataclasses import dataclass, field
from typing import Optional, List

from ..lib.functions import _update_status

@dataclass
class SampleCounter:
    status_file: str = None
    total_samples: int = None
    current_sample: int = 0
    num_players: int = 0
    condiction: threading.Condition = None
    players_ready: List[int] = field(default_factory=list)
    players_registered: List[int] = field(default_factory=list)
    soft_players_registered: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.condiction = threading.Condition()

    def register_player(self, player_id: int) -> None:
        """Register a ModalPlayer instance."""
        if player_id not in self.players_registered:
            self.players_registered.append(player_id)
            self.num_players += 1
            print(f"Player {player_id} registered. Total players: {self.num_players}")
            return self.condiction
    
    def unregister_player(self, player_id: int) -> None:
        """Unregister a ModalPlayer instance."""
        if player_id in self.players_registered:
            self.players_registered.remove(player_id)
            self.num_players -= 1
            print(f"Player {player_id} unregistered. Total players: {self.num_players}")
    
    def get_current(self) -> int:
        """Get the current sample index."""
        return self.current_sample
    
    def get_next(self, player_id: int) -> int:
        """ Returns the next sample index. """
        if player_id in self.players_registered and not player_id in self.players_ready and self.num_players > 0:
            return self.current_sample + 1

    def ready(self, player_id):
        """
        wait until all players are ready.
        """
        if player_id in self.players_registered and not player_id in self.players_ready:
            self.players_ready.append(player_id)
            if len(self.players_ready) == len(self.players_registered):
                if self.current_sample < self.total_samples:
                    print('SampleCounter: ', self.current_sample, self.total_samples, self.get_progress())
                    self.current_sample += 1
                    if self.current_sample % int(self.total_samples/100) == 0:
                       _update_status(self.status_file, int(self.get_progress()))
                self.players_ready = []
                self.condiction.notify_all()
            else:
                self.condiction.wait()

    def set_total_samples(self, total_samples: int) -> None:
        """Set the total number of samples."""
        self.total_samples = total_samples
        print(f"Total samples set to: {total_samples}")
    
    def get_progress(self) -> float:
        """Get progress as a percentage."""
        if self.total_samples and self.total_samples > 0:
#            return (self.current_sample * 80) / (self.total_samples - 1)
            return 10 + ((self.current_sample * 80) / (self.total_samples)) # 10 + for rigidboy bake with 80 as 100 in range 10-90
        return 0.0
