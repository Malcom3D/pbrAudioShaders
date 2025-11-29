import("dsp_files/faustlib/stdfaust.lib");
process = 1.0 : ba.impulsify : pm.clarinet_modal(440, 0.5, 0.5, 0.5) * 0.1;
