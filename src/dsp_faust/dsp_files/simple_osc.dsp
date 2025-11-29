import("stdfaust.lib");

freq = nentry("freq", 440, 20, 2000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.1);

process = os.osc(freq) * gain;
