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
