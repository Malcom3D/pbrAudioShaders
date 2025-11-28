//#ifndef FAUST_PYTHON_WRAPPER_H
//#define FAUST_PYTHON_WRAPPER_H

extern "C" {
// Factory functions
llvm_dsp_factory* createDSPFactoryFromFileWrapper(const char* filename, int argc, const char** argv, 
                                                 const char* target, char* error_msg, int opt_level);
bool deleteDSPFactoryWrapper(llvm_dsp_factory* factory);

// DSP instance functions  
llvm_dsp* createDSPInstanceWrapper(llvm_dsp_factory* factory);
void deleteDSPInstanceWrapper(llvm_dsp* dsp);
void dsp_compute(llvm_dsp* dsp, int count, float** inputs, float** outputs);
int dsp_getNumInputs(llvm_dsp* dsp);
int dsp_getNumOutputs(llvm_dsp* dsp);

// Dummy audio functions
dummyaudio* create_dummyaudio(int sr, int bs, int count, int sample, bool manager, bool exit);
void delete_dummyaudio(dummyaudio* audio);
bool dummyaudio_init(dummyaudio* audio, const char* name, llvm_dsp* dsp);
bool dummyaudio_start(dummyaudio* audio);
void dummyaudio_stop(dummyaudio* audio);
void dummyaudio_render(dummyaudio* audio);

// NEW: Functions to access stored samples
int dummyaudio_get_num_buffers_rendered(dummyaudio* audio);
int dummyaudio_get_total_samples(dummyaudio* audio);
void dummyaudio_get_all_samples(dummyaudio* audio, float* output_array);
void dummyaudio_get_channel_samples(dummyaudio* audio, int channel, float* output_array);
int dummyaudio_get_buffer_size(dummyaudio* audio);
int dummyaudio_get_num_outputs(dummyaudio* audio);
}

//#endif

