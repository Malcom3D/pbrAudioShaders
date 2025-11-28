import numpy as np
import dsp_faust
import soundfile as sf

class ImpactRenderer:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Load the Faust DSP from file
#        with open('impact.dsp', 'r') as f:
#            dsp_content = f.read()
        
        # Create Faust instance
        self.faust = dsp_faust.DspFaust('impact.dsp', sample_rate, buffer_size, False)
        self.faust.start()
        
    def trigger_impact(self, force=0.5, ex_pos_A=0, ex_pos_B=0, 
                     energy_to_A=0.4, energy_to_B=0.4, t60_scale=1.0,
                     gain_A=0.3, gain_B=0.3, duration=2.0):
        """
        Trigger an impact and render the resulting audio
        
        Args:
            force: Impact force (0-1)
            ex_pos_A: Excitation position for object A (0-6)
            ex_pos_B: Excitation position for object B (0-6)
            energy_to_A: Energy distribution to object A (0-1)
            energy_to_B: Energy distribution to object B (0-1)
            t60_scale: Global resonance control
            gain_A: Gain for object A (0-1)
            gain_B: Gain for object B (0-1)
            duration: Duration of rendered audio in seconds
        """
        # Set parameters
        self.faust.setParamValue("impactForce", force)
        self.faust.setParamValue("exPos_A", ex_pos_A)
        self.faust.setParamValue("exPos_B", ex_pos_B)
        self.faust.setParamValue("energyToA", energy_to_A)
        self.faust.setParamValue("energyToB", energy_to_B)
        self.faust.setParamValue("t60Scale", t60_scale)
        self.faust.setParamValue("gain_A", gain_A)
        self.faust.setParamValue("gain_B", gain_B)
        
        # Trigger the impact
        self.faust.propagateMidi(0, 0, 0x90, 0, 60, 127)  # Note on
        self.faust.propagateMidi(1000, 0, 0x80, 0, 60, 0)  # Note off after 1ms
        
        # Render audio
        return self.render_audio(duration)
    
    def render_audio(self, duration):
        """Render audio for specified duration"""
        num_samples = int(duration * self.sample_rate)
        num_blocks = int(np.ceil(num_samples / self.buffer_size))
        
        audio = np.zeros((num_samples, 2))  # Stereo output
        
        # Render audio block by block
        for i in range(num_blocks):
            # Compute audio block
            block = self.faust.compute()
            
            # Convert to numpy array (assuming compute() returns interleaved stereo)
            block_np = np.array(block).reshape(-1, 2)
            
            # Determine samples to copy copy
            start_idx = i * self.buffer_size
            end_idx = min(start_idx + self.buffer_size, num_samples)
            samples_to_copy = end_idx - start_idx
            
            if samples_to_copy > 0:
                audio[start_idx:end_idx] = block_np[:samples_to_copy]
        
        return audio
    
    def save_audio(self, audio, filename, bit_depth=16):
        """Save audio to file with specified bit depth"""
        # Normalize audio to appropriate range for bit depth
        if bit_depth == 16:
            audio_normalized = np.int16(audio * 32767)
            subtype = 'PCM_16'
        elif bit_depth == 24:
            audio_normalized = np.int32(audio * 8388607)
            subtype = 'PCM_24'
        elif bit_depth == 32:
            audio_normalized = audio.astype(np.float32)
            subtype = 'FLOAT'
        else:
            raise ValueError("Unsupported bit depth. Use 16, 24, or 32.")
        
        sf.write(filename, audio_normalized, self.sample_rate, subtype=subtype)
    
    def close(self):
        """Clean up"""
        self.faust.stop()

# Usage example
if __name__ == "__main__":
    # Create renderer
    renderer = ImpactRenderer(sample_rate=44100)
    
    try:
        # Trigger a gentle impact
        print("Rendering gentle impact...")
        audio_gentle = renderer.trigger_impact(
            force=0.3,
            ex_pos_A=2,
            ex_pos_B=3,
            energy_to_A=0.6,
            energy_to_B=0.3,
            t60_scale=1.5,
            duration=3.0
        )
        
        # Save with different bit depths
        renderer.save_audio(audio_gentle, "gentle_impact_16bit.wav", 16)
        renderer.save_audio(audio_gentle, "gentle_impact_24bit.wav", 24)
        renderer.save_audio(audio_gentle, "gentle_impact_32bit.wav", 32)
        
        # Trigger a strong impact
        print("Rendering strong impact...")
        audio_strong = renderer.trigger_impact(
            force=0.9,
            ex_pos_A=0,
            ex_pos_B=6,
            energy_to_A=0.2,
            energy_to_B=0.7,
            t60_scale=0.8,
            duration=4.0
        )
        
        renderer.save_audio(audio_strong, "strong_impact_24bit.wav", 24)
        
        print("All renders completed!")
        
    finally:
        renderer.close()
