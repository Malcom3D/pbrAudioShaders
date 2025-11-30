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
from typing import List, Tuple, Any
from ..utils.config import Config

class ImpactManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: str):
        with self._lock:
            if not self._initialized:
                self._impacts = {}
                self._initialized = True

                self._config = Config(config)

    def register(self, impact_data: Any, idx: int = None) -> None:
        if 'ImpactData' in str(type(impact_data)):
            self._impacts[idx] = impact_data

    def get(self, type: str = None, idx: int = None) -> Any:
        if type == None:
            return self._config, self._impacts
        if 'config' in type:
            return self._config
        elif 'impact' in type:
            return self._impacts[idx] if not idx == None else self._impacts

    def unregister(self) -> None:
        pass

