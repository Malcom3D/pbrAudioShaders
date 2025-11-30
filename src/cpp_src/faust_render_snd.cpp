/*
 * Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
 *
 * This file is part of pbrAudio.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "faust/dsp/llvm-dsp.h"
#include "dsp-faust/DspFaust.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <cstdlib> // for atoi, atof

void renderWithLibsndfile(const std::string& dspFile, const std::string& outputWav, 
                         double durationSeconds, int sampleRate = 192000) {
    
    std::string error_msg;
    llvm_dsp_factory* factory = createDSPFactoryFromFile(dspFile, 0, nullptr, "", error_msg, -1);
    
    if (!factory) {
        std::cerr << "Error creating factory: " << error_msg << std::endl;
        return;
    }
    
    llvm_dsp* dsp = factory->createDSPInstance();
    if (!dsp) {
        std::cerr << "Error creating DSP instance" << std::endl;
        delete factory;
        return;
    }
    
    dsp->init(sampleRate);
    int numOutputs = dsp->getNumOutputs();
    
    // Setup libsndfile
    SF_INFO sfinfo;
    sfinfo.samplerate = sampleRate;
    sfinfo.channels = numOutputs;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT; // Changed to WAV format
    
    SNDFILE* outfile = sf_open(outputWav.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Error opening output file: " << sf_strerror(nullptr) << std::endl;
        delete dsp;
        delete factory;
        return;
    }
    
    // Render parameters
    int totalFrames = static_cast<int>(durationSeconds * sampleRate);
    int bufferSize = 512;
    
    // Buffers
    int numInputs = dsp->getNumInputs();
    float** inputs = nullptr;
    float** outputs = new float*[numOutputs];
    
    // Only allocate input buffers if there are inputs
    if (numInputs > 0) {
        inputs = new float*[numInputs];
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = new float[bufferSize]();
        }
    } else {
        inputs = nullptr;
    }
    
    for (int i = 0; i < numOutputs; i++) {
        outputs[i] = new float[bufferSize];
    }
    
    // Interleaved buffer for libsndfile
    std::vector<float> interleaved(bufferSize * numOutputs);
    
    // Render
    int framesRendered = 0;
    while (framesRendered < totalFrames) {
        int framesToRender = std::min(bufferSize, totalFrames - framesRendered);
        
        dsp->compute(framesToRender, inputs, outputs);
        
        // Interleave
        for (int frame = 0; frame < framesToRender; frame++) {
            for (int chan = 0; chan < numOutputs; chan++) {
                interleaved[frame * numOutputs + chan] = outputs[chan][frame];
            }
        }
        
        sf_writef_float(outfile, interleaved.data(), framesToRender);
        framesRendered += framesToRender;
    }
    
    // Cleanup - in correct order
    sf_close(outfile);
    
    // Cleanup output buffers
    for (int i = 0; i < numOutputs; i++) {
        delete[] outputs[i];
    }
    delete[] outputs;
    
    // Cleanup input buffers if they were allocated
    if (numInputs > 0) {
        for (int i = 0; i < numInputs; i++) {
            delete[] inputs[i];
        }
        delete[] inputs;
    }
    
    // Delete Faust objects in correct order
    delete dsp;
    delete factory;
    
    std::cout << "Rendering complete: " << outputWav << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <dsp_file> <output_file> <duration_seconds>" << std::endl;
    std::cout << "Example: " << programName << " dsp_files/test.dsp output.wav 10.0" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check for correct number of arguments
    if (argc != 4) {
        std::cerr << "Error: Incorrect number of arguments." << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    std::string dspFile = argv[1];
    std::string outputFile = argv[2];
    double duration = std::atof(argv[3]);
    
    // Validate duration
    if (duration <= 0) {
        std::cerr << "Error: Duration must be a positive number." << std::endl;
        return 1;
    }
    
    std::cout << "Rendering " << dspFile << " to " << outputFile 
              << " for " << duration << " seconds" << std::endl;
    
    renderWithLibsndfile(dspFile, outputFile, duration);
    
    return 0;
}
