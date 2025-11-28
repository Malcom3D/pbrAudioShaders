import os, sys
import ctypes
import numpy as np
import soundfile as sf
from typing import Optional, List, Tuple

lib = ctypes.CDLL('./lib/faust_python.so')

lib.createDSPFactoryFromFileWrapper.restype = ctypes.c_void_p
lib.createDSPFactoryFromFileWrapper.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
lib.deleteDSPFactoryWrapper.restype = ctypes.c_bool
lib.deleteDSPFactoryWrapper.argtypes = [ctypes.c_void_p]
lib.createDSPInstanceWrapper.restype = ctypes.c_void_p
lib.createDSPInstanceWrapper.argtypes = [ctypes.c_void_p]
lib.deleteDSPInstanceWrapper.argtypes = [ctypes.c_void_p]
lib.dsp_compute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
lib.dsp_getNumInputs.restype = ctypes.c_int
lib.dsp_getNumInputs.argtypes = [ctypes.c_void_p]
lib.dsp_getNumOutputs.restype = ctypes.c_int
lib.dsp_getNumOutputs.argtypes = [ctypes.c_void_p]

lib.create_dummyaudio.restype = ctypes.c_void_p
lib.create_dummyaudio.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
lib.delete_dummyaudio.restype = ctypes.c_void_p
lib.delete_dummyaudio.argtypes = [ctypes.c_void_p]
lib.dummyaudio_init.restype = ctypes.c_bool
lib.dummyaudio_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
lib.dummyaudio_start.restype = ctypes.c_bool
lib.dummyaudio_start.argtypes = [ctypes.c_void_p]
lib.dummyaudio_stop.restype = ctypes.c_void_p
lib.dummyaudio_stop.argtypes = [ctypes.c_void_p]
lib.dummyaudio_render.restype = ctypes.c_void_p
lib.dummyaudio_render.argtypes = [ctypes.c_void_p]
lib.dummyaudio_get_buffer_size.restype = ctypes.c_int
lib.dummyaudio_get_buffer_size.argtypes = [ctypes.c_void_p]
lib.dummyaudio_get_num_outputs.restype = ctypes.c_int
lib.dummyaudio_get_num_outputs.argtypes = [ctypes.c_void_p]

# NEW: Add the new wrapper functions
lib.dummyaudio_get_num_buffers_rendered.restype = ctypes.c_int
lib.dummyaudio_get_num_buffers_rendered.argtypes = [ctypes.c_void_p]
lib.dummyaudio_get_total_samples.restype = ctypes.c_int
lib.dummyaudio_get_total_samples.argtypes = [ctypes.c_void_p]
lib.dummyaudio_get_all_samples.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.dummyaudio_get_channel_samples.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

sample_rate = 48000
buffer_size = 512

#dsp_file = 'dsp_files/test.dsp'
if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]) and os.path.isfile(sys.argv[1]):
        dsp_file = sys.argv[1]
    else:
        print(f"Usage: python3 sys.argv[0] file.dsp")

error_msg = ctypes.create_string_buffer(1024)

factory = lib.createDSPFactoryFromFileWrapper(dsp_file.encode('utf-8'), 0, None, b"", error_msg, -1)

dsp_instance = lib.createDSPInstanceWrapper(factory)
num_inputs = lib.dsp_getNumInputs(dsp_instance)
num_outputs = lib.dsp_getNumOutputs(dsp_instance)

dummy_name = 'dummy'
dummy_audio = lib.create_dummyaudio(sample_rate, buffer_size, 1024, 24, True, False)
lib.dummyaudio_init(dummy_audio, dummy_name.encode('utf-8'), dsp_instance)

# Render the audio
lib.dummyaudio_start(dummy_audio)
# Note: dummyaudio_render is called automatically in the start() method for finite buffers

# Get the number of buffers rendered
num_buffers_rendered = lib.dummyaudio_get_num_buffers_rendered(dummy_audio)
print(f"Rendered {num_buffers_rendered} buffers")

# Get total number of samples
total_samples = lib.dummyaudio_get_total_samples(dummy_audio)
print(f"Total samples: {total_samples}")

all_samples_array = np.zeros(total_samples, dtype=np.float32)
lib.dummyaudio_get_all_samples(dummy_audio, all_samples_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

data = all_samples_array
sf.write('new_file.wav', data, 48000)
