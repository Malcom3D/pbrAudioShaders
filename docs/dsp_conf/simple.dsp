// SimpleSynth.dsp
// A basic monophonic synthesizer with ADSR envelope

import("stdfaust.lib");

// Parameters
freq = hslider("freq", 440, 20, 2000, 1);
gain = hslider("gain", 0.5, 0, 1, 0.01);
cutoff = hslider("cutoff", 1000, 100, 5000, 10);
gate = button("gate");

// ADSR Envelope
attack = 0.1;
decay = 0.2;
sustain = 0.7;
release = 0.5;

envelope = en.adsr(attack, decay, sustain, release, gate);

// Oscillator
oscillator = os.osc(freq);

// Low-pass filter
filtered = fi.lowpass(6, cutoff, oscillator);

// Apply envelope and gain
output = filtered * envelope * gain;

// Output (mono)
process = output;

