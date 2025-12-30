################################################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../../src')
from rigidBody import rigidBody
config_file = 'config.json'
rbs = rigidBody(config_file)
rbs.prebake()
rbs.bake()
################################################################################

#################################################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src')
from ImpactShader import ImpactShader
config_file = 'config.json'
imp = ImpactShader(config_file)
################################################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../../src')
from ImpactShader import ImpactShader
config_file = 'config.json'
imp = ImpactShader(config_file)
################################################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/ImpactShader')
from lib.impacts_data import Collision, ObjCollision, ImpactData
from core.impact_manager import ImpactManager
from core.impact_engine import ImpactEngine
from core.impact_analyzer import ImpactAnalyzer
from core.mesh2modal import Mesh2Modal
from core.optimize_obj import OptimizeObj
from core.synth_writer import SynthWriter

config_file = 'config.json'
im = ImpactManager(config_file)
sw = SynthWriter(im)
optimize_obj = OptimizeObj(im)

config = im.get('config')
for obj in config.objects:
    optimize_obj.compute(obj.idx)

impact = ImpactData(idx=0, time=0, coord=[1.0,3.6,8.2])
collision1 = Collision(54, [3.4,2.0,5])
collision2 = Collision(34, [4.0,1.0,4])
obj_collision1 = ObjCollision(obj_idx=0, collision=collision1)
obj_collision2 = ObjCollision(obj_idx=1, collision=collision2)
impact.add_collision(collision=obj_collision1)
impact.add_collision(collision=obj_collision2)
im.register(impact)

sw.compute()

import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src')
from ImpactShader import ImpactShader
config_file = 'config.json'
imp = ImpactShader(config_file)

import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src')
config_file = 'config.json'
from ImpactShader import ImpactShader
from ImpactShader.core.impact_manager import ImpactManager
from ImpactShader.core.impact_engine import ImpactEngine
im = ImpactManager(config_file)
ie = ImpactEngine(im)
ie.compute()











import sys, os
import numpy as np
sys.path.append('/home/malcom/Sources/pbrAudio/pbrAudioShaders/pbrAudioShaders/src/ImpactShader')

from tools.faust_render import FaustRender

dsp_file = 'test.dsp'
output_file = 'test_pcm_float32_192000.raw'
duration = 3.0
FaustRender(dsp_file=dsp_file, output_file=output_file, duration=duration)
audio_data = np.fromfile('test_pcm_float32_192000.raw', dtype=np.float32)


import sys, os
import numpy as np
sys.path.append('/home/malcom/Sources/pbrAudio/pbrAudioShaders/pbrAudioShaders/src/ImpactShader')

from tools.pym2f import Pym2f

obj_file = 'icosphere/icosphere0001.obj'
young_modulus = 2E11
poisson_ration = 0.29
density = 7850
alpha_rayleigh = 5
beta_rayleigh = 3E-8
expos = [0,10]
output_name = 'icosphere'
icos = Pym2f()
icos.convert(obj_file=obj_file, young_modulus=young_modulus, poisson_ration=poisson_ration, density=density, alpha_rayleigh=alpha_rayleigh, beta_rayleigh=beta_rayleigh, expos=expos, output_name=output_name)
