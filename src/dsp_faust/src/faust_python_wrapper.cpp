#include "faust/dsp/llvm-dsp.h"
#include "faust/gui/GUI.h"
#include "faust/gui/PathBuilder.h"
#include "faust/gui/MetaDataUI.h"
#include "faust/audio/audio.h"
#include "dsp-faust/dummy-audio.h"
#include "dsp-faust/DspFaust.h"

using namespace std;

extern "C" {

// Factory functions
llvm_dsp_factory* createDSPFactoryFromFileWrapper(const char* filename, int argc, const char** argv, 
                                                 const char* target, char* error_msg, int opt_level) {
    string error_str;
    llvm_dsp_factory* factory = createDSPFactoryFromFile(filename, argc, argv, target, error_str, opt_level);
    strncpy(error_msg, error_str.c_str(), 1023); // Prevent buffer overflow
    error_msg[1023] = '\0';
    return factory;
}

bool deleteDSPFactoryWrapper(llvm_dsp_factory* factory) {
    return deleteDSPFactory(factory);
}

// DSP instance functions  
llvm_dsp* createDSPInstanceWrapper(llvm_dsp_factory* factory) {
    return factory->createDSPInstance();
}

void deleteDSPInstanceWrapper(llvm_dsp* dsp) {
    delete dsp;
}

void dsp_compute(llvm_dsp* dsp, int count, float** inputs, float** outputs) {
    dsp->compute(count, inputs, outputs);
}

int dsp_getNumInputs(llvm_dsp* dsp) {
    return dsp->getNumInputs();
}

int dsp_getNumOutputs(llvm_dsp* dsp) {
    return dsp->getNumOutputs();
}

// Dummy audio functions
dummyaudio* create_dummyaudio(int sr, int bs, int count, int sample, bool manager, bool exit) {
    return new dummyaudio(sr, bs, count, sample, manager, exit);
}

void delete_dummyaudio(dummyaudio* audio) {
    delete audio;
}

bool dummyaudio_init(dummyaudio* audio, const char* name, llvm_dsp* dsp) {
    return audio->init(name, dsp);
}

bool dummyaudio_start(dummyaudio* audio) {
    return audio->start();
}

void dummyaudio_stop(dummyaudio* audio) {
    audio->stop();
}

void dummyaudio_render(dummyaudio* audio) {
    audio->render();
}

// NEW: Functions to access stored samples
int dummyaudio_get_num_buffers_rendered(dummyaudio* audio) {
    return audio->getNumBuffersRendered();
}

int dummyaudio_get_total_samples(dummyaudio* audio) {
    auto buffers = audio->getOutputBuffers();
    int total = 0;
    for (const auto& buffer : buffers) {
        total += buffer.size();
    }
    return total;
}

void dummyaudio_get_all_samples(dummyaudio* audio, float* output_array) {
    auto all_samples = audio->getAllSamples();
    std::copy(all_samples.begin(), all_samples.end(), output_array);
}

void dummyaudio_get_channel_samples(dummyaudio* audio, int channel, float* output_array) {
    auto channel_samples = audio->getChannelSamples(channel);
    std::copy(channel_samples.begin(), channel_samples.end(), output_array);
}

int dummyaudio_get_buffer_size(dummyaudio* audio) {
    return audio->getBufferSize();
}

int dummyaudio_get_num_outputs(dummyaudio* audio) {
    return audio->getNumOutputs();
}
}
