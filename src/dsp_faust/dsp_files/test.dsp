import("dsp_files/faustlib/stdfaust.lib");

// Import the two modal models
import("dsp_files/modalModel_A.lib");
import("dsp_files/modalModel_B.lib");

// Excitation positions for the two objects
exPos_A = nentry("exPos_A", 0, 0, 6, 1);
exPos_B = nentry("exPos_B", 0, 0, 6, 1);

t60Scale = 1;
gain = 0.6;

// === Hammer Excitation ===
// We excite the hammer model to get its characteristic impact sound
hammerExcitation = 1.0 : ba.impulsify;
// Continuous excitation instead of single impulse
//hammerExcitation = 1.0 : ba.impulsify : + ~ _;
// Or use a periodic impulse
//hammerExcitation = 1.0 : pm.djembe(1, 0.3, 0.4, 0.2) * 0.1;

// Process the hammer excitation through modalModel_B
// This gives us the hammer's vibration signature
hammerVibration = hammerExcitation : modalModel_B(exPos_B, t60Scale);

excitation_A = hammerVibration * gain;
excitation_B = hammerVibration * (1 - gain);

// === Individual Object Processing ===
object_A = excitation_A : modalModel_A(exPos_A, t60Scale) * gain;
object_B = excitation_B;
process = (object_A + object_B) * 0.75;

