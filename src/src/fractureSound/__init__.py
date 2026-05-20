# ./fractureSound/__init__.py

__version__ = "0.1.0"
__author__ = "Malcom3D"
__description__ = "Physically plausible fracture sound synthesis"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.fracture_engine import fractureEngine
from .lib.fracture_data import FractureEvent, FractureType, FragmentData
from .lib.fracture_modal import FractureModalModel
from .lib.fracture_synth import FractureSynth

__all__ = [
    'fractureEngine',
    'FractureEvent',
    'FractureType',
    'FragmentData',
    'FractureModalModel',
    'FractureSynth'
]
