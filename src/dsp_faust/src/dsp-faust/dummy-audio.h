/************************** BEGIN dummy-audio.h *************************
 FAUST Architecture File
 Copyright (C) 2003-2022 GRAME, Centre National de Creation Musicale
 ---------------------------------------------------------------------
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 
 EXCEPTION : As a special exception, you may create a larger work
 that contains this FAUST architecture section and distribute
 that work under terms of your choice, so long as this FAUST
 architecture section is not modified.
 ************************************************************************/
#ifndef __dummy_audio__
#define __dummy_audio__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <iomanip>
#include <vector>

#ifdef USE_PTHREAD
#include <pthread.h>
#else
#include <thread>
#endif

#define BUFFER_TO_RENDER 10

struct dummyaudio_base : public audio {
    virtual void render() = 0;
    
    // NEW: Add virtual methods for accessing samples
    virtual const std::vector<std::vector<FAUSTFLOAT>>& getOutputBuffers() const = 0;
    virtual int getNumBuffersRendered() const = 0;
    virtual std::vector<FAUSTFLOAT> getAllSamples() const = 0;
    virtual std::vector<FAUSTFLOAT> getChannelSamples(int channel) const = 0;
};

template <typename REAL>
class dummyaudio_real : public dummyaudio_base {

    private:
        dsp* fDSP;
        int fSampleRate;
        int fBufferSize;
        REAL** fInChannel;
        REAL** fOutChannel;
        int fNumInputs;
        int fNumOutputs;
        bool fRunning;
        int fRender;
        int fCount;
        int fSample;
        bool fManager;
        bool fExit;

        // NEW: Storage for output samples
        std::vector<std::vector<REAL>> fOutputBuffers;
        int fCurrentBufferIndex;

        void runAux()
        {
            try {
                process();
            } catch (...) {
                if (fExit) exit(EXIT_FAILURE);
            }
        }

    #ifdef USE_PTHREAD
        pthread_t fAudioThread;
        static void* run(void* ptr)
        {
            static_cast<dummyaudio_real*>(ptr)->runAux();
        }
    #else
        std::thread* fAudioThread = nullptr;
        static void run(dummyaudio_real* audio)
        {
            audio->runAux();
        }
    #endif

        void process()
        {
            while (fRunning && (fRender-- > 0)) {
//                if (fSample > 0) { std::cout << "Render one buffer"; }
                if (fSample > 0)
                render();
            }
            fRunning = false;
        }

    public:
        dummyaudio_real(int sr, int bs,
                        int count = BUFFER_TO_RENDER,
                        int sample = -1,
                        bool manager = false,
                        bool exit = false)
        :fSampleRate(sr), fBufferSize(bs),
        fInChannel(nullptr), fOutChannel(nullptr),
        fNumInputs(-1), fNumOutputs(-1),
        fRender(0), fCount(count),
        fSample(sample), fManager(manager),
        fExit(exit), fCurrentBufferIndex(0)
        {
            // NEW: Pre-allocate storage for output buffers
            if (fCount > 0) {
                fOutputBuffers.resize(fCount);
            }
        }

        dummyaudio_real(int count = BUFFER_TO_RENDER)
        :fSampleRate(48000), fBufferSize(512),
        fInChannel(nullptr), fOutChannel(nullptr),
        fNumInputs(-1), fNumOutputs(-1),
        fRender(0), fCount(count),
        fSample(512), fManager(false),
        fExit(false), fCurrentBufferIndex(0)
        {
            // NEW: Pre-allocate storage for output buffers
            if (fCount > 0) {
                fOutputBuffers.resize(fCount);
            }
        }

        virtual ~dummyaudio_real()
        {
            for (int i = 0; i < fNumInputs; i++) {
                delete[] fInChannel[i];
            }
            for (int i = 0; i < fNumOutputs; i++) {
                delete[] fOutChannel[i];
            }
            delete [] fInChannel;
            delete [] fOutChannel;
        }

        virtual bool init(const char* name, dsp* dsp)
        {
            fDSP = dsp;
            fNumInputs = fDSP->getNumInputs();
            fNumOutputs = fDSP->getNumOutputs();

            fInChannel = new REAL*[fNumInputs];
            fOutChannel = new REAL*[fNumOutputs];

            for (int i = 0; i < fNumInputs; i++) {
                fInChannel[i] = new REAL[fBufferSize];
                memset(fInChannel[i], 0, sizeof(REAL) * fBufferSize);
            }
            for (int i = 0; i < fNumOutputs; i++) {
                fOutChannel[i] = new REAL[fBufferSize];
                memset(fOutChannel[i], 0, sizeof(REAL) * fBufferSize);
            }

            if (fManager) {
                fDSP->instanceInit(fSampleRate);
            } else {
                fDSP->init(fSampleRate);
            }

            return true;
        }

        virtual bool start()
        {
            fRender = fCount;
            fRunning = true;
            if (fCount == INT_MAX) {
            #ifdef USE_PTHREAD
                if (pthread_create(&fAudioThread, 0, run, this) != 0) {
                    fRunning = false;
                }
            #else
                fAudioThread = new std::thread(dummyaudio_real::run, this);
            #endif
                return fRunning;
            } else {
                process();
                return true;
            }
        }

        virtual void stop()
        {
            if (fRunning) {
                fRunning = false;
            #ifdef USE_PTHREAD
                pthread_join(fAudioThread, 0);
            #else
                fAudioThread->join();
                delete fAudioThread;
                fAudioThread = 0;
            #endif
            }
        }

        void render()
        {
            AVOIDDENORMALS;

            fDSP->compute(fBufferSize, reinterpret_cast<FAUSTFLOAT**>(fInChannel), reinterpret_cast<FAUSTFLOAT**>(fOutChannel));
            
            // NEW: Store output samples
            if (fCurrentBufferIndex < fOutputBuffers.size()) {
                fOutputBuffers[fCurrentBufferIndex].resize(fBufferSize * fNumOutputs);
                for (int frame = 0; frame < fBufferSize; frame++) {
                    for (int chan = 0; chan < fNumOutputs; chan++) {
                        fOutputBuffers[fCurrentBufferIndex][frame * fNumOutputs + + chan] = fOutChannel[chan][frame];
                    }
                }
                fCurrentBufferIndex++;
            }
            
//            if (fNumInputs > 0) {
//                for (int frame = 0; frame < fSample; frame++) {
//                    std::cout << std::fixed << std::setprecision(6) << "sample in " << fInChannel[0][frame] << std::endl;
//                }
//            }
//            if (fNumOutputs > 0) {
//                for (int frame = 0; frame < fSample; frame++) {
//                    std::cout << std::fixed << std::setprecision(16) << "sample out " << fOutChannel[0][frame] << std::endl;
//                }
//            }
        }

        // NEW: Methods to access stored samples - must be public
        const std::vector<std::vector<REAL>>& getOutputBuffers() const override {
            return fOutputBuffers;
        }
        
        int getNumBuffersRendered() const override {
            return fCurrentBufferIndex;
        }
        
        std::vector<REAL> getAllSamples() const override {
            std::vector<REAL> allSamples;
            for (int i = 0; i < fCurrentBufferIndex; i++) {
                allSamples.insert(allSamples.end(), fOutputBuffers[i].begin(), fOutputBuffers[i].end());
            }
            return allSamples;
        }
        
        std::vector<REAL> getChannelSamples(int channel) const override {
            std::vector<REAL> channelSamples;
            for (int i = 0; i < fCurrentBufferIndex; i++) {
                for (int frame = 0; frame < fBufferSize; frame++) {
                    if (channel < fNumOutputs) {
                        channelSamples.push_back(fOutputBuffers[i][frame * fNumOutputs + channel]);
                    }
                }
            }
            return channelSamples;
        }

        virtual int getBufferSize() { return fBufferSize; }
        virtual int getSampleRate() { return fSampleRate; }
        virtual int getNumInputs() { return fNumInputs; }
        virtual int getNumOutputs() { return fNumOutputs; }
};

struct dummyaudio : public dummyaudio_real<FAUSTFLOAT> {
    dummyaudio(int sr, int bs,
               int count = BUFFER_TO_RENDER,
               int sample = -1,
               bool manager = false,
               bool exit = false)
    : dummyaudio_real<FAUSTFLOAT>(sr, bs, count, sample, manager, exit)
    {}

    dummyaudio(int count = BUFFER_TO_RENDER) 
    : dummyaudio_real<FAUSTFLOAT>(count)
    {}

    // NEW: Explicitly inherit the methods
    using dummyaudio_real<FAUSTFLOAT>::getOutputBuffers;
    using dummyaudio_real<FAUSTFLOAT>::getNumBuffersRendered;
    using dummyaudio_real<FAUSTFLOAT>::getAllSamples;
    using dummyaudio_real<FAUSTFLOAT>::getChannelSamples;
};

#endif
/**************************  END  dummy-audio.h **************************/
