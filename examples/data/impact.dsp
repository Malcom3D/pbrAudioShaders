import("stdfaust.lib");

// Import the two modal models
import("modalModel_A.lib");
import("modalModel_B.lib");

// === Impact Trigger and Global Controls ===
impactForce = hslider("impactForce", 0.5, 0, 1, 0.01) : ba.sAndH(gate);
gate = button("gate"); // Trigger for the impact

// Excitation positions for the two objects
exPos_A = nentry("exPos_A", 0, 0, 6, 1) : ba.sAndH(gate);
exPos_B = nentry("exPos_B", 0, 0, 6, 1) : ba.sAndH(gate);

// Energy distribution between objects
energyToA = hslider("energyToA", 0.4, 0, 1, 0.01) : ba.sAndH(gate);
//energyToB = hslider("energyToB", 0.4, 0, 1, 0.01) : ba.sAndH(gate);

// Global resonance controls
t60Scale = hslider("t60Scale", 1, 0, 100, 0.01) : ba.sAndH(gate);

// === Hammer Excitation ===
// We excite the hammer model to get its characteristic impact sound
hammerExcitation = 1.0 : ba.impulsify(gate) * impactForce;

// Process the hammer excitation through modalModel_B
// This gives us the hammer's vibration signature
hammerVibration = hammerExcitation : modalModel_B(exPos_B, t60Scale);

// === Impact Coupling ===
// The hammer's vibration becomes the excitation for objects A and B
// We also preserve some of the hammer's own sound
excitation_A = hammerVibration * energyToA;
//excitation_B = hammerVibration * energyToB;
excitation_B = hammerVibration * (1 - energyToA);  // Hammer's residual vibration


// === Individual Object Processing ===
// Each object responds to its excitation with its own modal characteristics
object_A = excitation_A : modalModel_A(exPos_A, t60Scale) * gain_A;
object_B = excitation_B : modalModel_B(exPos_B, t60Scale) * gain_B;
//object_B = excitation_B; // Hammer sound is already processed

// === Individual Gains and Mix ===
gain_A = hslider("gain_A", 0.3, 0, 1, 0.01);
gain_B = hslider("gain_HammerGain", 0.1, 0, 1, 0.01); // Hammer is usually quieter
//gain_B = hslider("gain_B", 0.3, 0, 1, 0.01);

// Final mix with optional high-pass to remove excessive low energy
//process = (object_A * gain_A) + (object_B * gain_B) : fi.highpass(6, 50);
//process = (object_B * gain_B) : (object_A * gain_A) : fi.highpass(6, 50);
process = modalModel_A(exPos_A, t60Scale) : *(gain_A);
