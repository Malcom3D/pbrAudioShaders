import dsp_faust
import numpy as np
import matplotlib.pyplot as plt

# Create Faust instance
faust = dsp_faust.DspFaust(44100, 512)

# Set some parameters
faust.setParamValue("impactForce", 0.8)
faust.setParamValue("exPos_A", 2)
faust.setParamValue("t60Scale", 1.5)

# Trigger the gate to start the impact sound
faust.trigger_gate()

# Render 5 seconds of audio at 44.1kHz
sample_rate = 44100
duration_seconds = 5
num_frames = sample_rate * duration_seconds

# Render as numpy array (16-bit)
audio_data = faust.render_offline_mono(num_frames, bit_depth=16)

print(f"Rendered {len(audio_data)} samples")
print(f"Audio range: {np.min(audio_data):.3f} to {np.max(audio_data):.3f}")

# Plot the audio
plt.figure(figsize=(12, 4))
plt.plot(audio_data)
plt.title("Rendered Impact Sound")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# You can also trigger the gate multiple times
for i in range(3):
    faust.trigger_gate()
    # Render short segment after each trigger
    segment = faust.render_offline_mono(22050, bit_depth=16)  # 0.5 seconds

# Or hold the gate open
faust.set_gate(True)  # Gate on
sustained = faust.render_offline_mono(44100, bit_depth=16)  # 1 second
faust.set_gate(False)  # Gate off

# Clean up
del faust

