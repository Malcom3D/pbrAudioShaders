import ctypes
import numpy as np
import os
from typing import Optional, List, Tuple

class DspFaust:
    def __init__(self, dsp_file: str, sample_rate: int = 44100, buffer_size: int = 512):
        """
        Python wrapper for DspFaust dynamic DSP.
        
        Args:
            dsp_file: Path to the Faust .dsp file
            sample_rate: Audio sample rate
            buffer_size: Audio buffer size
        """
        self.lib = ctypes.CDLL('./faust_python.so')
        
        # Define function signatures
        self._setup_function_signatures()
        
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.dsp_file = dsp_file
        
        # Initialize Faust DSP
        self.factory = None
        self.dsp_instance = None
        
        self._initialize_faust()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the Faust library."""
        
        # createDSPFactoryFromFileWrapper
        self.lib.createDSPFactoryFromFileWrapper.restype = ctypes.c_void_p
        self.lib.createDSPFactoryFromFileWrapper.argtypes = [
            ctypes.c_char_p,  # filename
            ctypes.c_int,     # argc
            ctypes.POINTER(ctypes.c_char_p),  # argv
            ctypes.c_char_p,  # target
            ctypes.c_char_p,  # error_msg
            ctypes.c_int      # opt_level
        ]
        
        # deleteDSPFactoryWrapper
        self.lib.deleteDSPFactoryWrapper.restype = ctypes.c_bool
        self.lib.deleteDSPFactoryWrapper.argtypes = [ctypes.c_void_p]
        
        # createDSPInstanceWrapper
        self.lib.createDSPInstanceWrapper.restype = ctypes.c_void_p
        self.lib.createDSPInstanceWrapper.argtypes = [ctypes.c_void_p]
        
        # deleteDSPInstanceWrapper
        self.lib.deleteDSPInstanceWrapper.argtypes = [ctypes.c_void_p]
        
        # dsp_compute
        self.lib.dsp_compute.argtypes = [
            ctypes.c_void_p,  # dsp instance
            ctypes.c_int,     # count
            ctypes.c_void_p,  # inputs
            ctypes.c_void_p   # outputs
        ]
        
        # dsp_getNumInputs
        self.lib.dsp_getNumInputs.restype = ctypes.c_int
        self.lib.dsp_getNumInputs.argtypes = [ctypes.c_void_p]
        
        # dsp_getNumOutputs
        self.lib.dsp_getNumOutputs.restype = ctypes.c_int
        self.lib.dsp_getNumOutputs.argtypes = [ctypes.c_void_p]
    
    def _initialize_faust(self):
        """Initialize the Faust DSP from file."""
        # Create DSP factory from file
        error_msg = ctypes.create_string_buffer(1024)
        
        self.factory = self.lib.createDSPFactoryFromFileWrapper(
            self.dsp_file.encode('utf-8'),
            0,
            None,
            b"",  # empty target for native compilation
            error_msg,
            -1
        )
        
        if not self.factory:
            raise RuntimeError(f"Failed to create DSP factory: {error_msg.value.decode()}")
        
        # Create DSP instance
        self.dsp_instance = self.lib.createDSPInstanceWrapper(self.factory)
        if not self.dsp_instance:
            raise RuntimeError("Failed to create DSP instance")
        
        # Get number of inputs and outputs
        self.num_inputs = self.lib.dsp_getNumInputs(self.dsp_instance)
        self.num_outputs = self.lib.dsp_getNumOutputs(self.dsp_instance)
        
        print(f"DSP loaded: {self.num_inputs} inputs, {self.num_outputs} outputs")
    
    def compute(self, input_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute one buffer of audio.
        
        Args:
            input_data: Input audio data as numpy array (num_channels x buffer_size)
                      If None, generates silence
            
        Returns:
            Output audio data as numpy array (num_outputs x buffer_size)
        """
        # Prepare input buffers
        if input_data is None:
            # Create silent input
            input_arrays = []
            input_ptrs = (ctypes.POINTER(ctypes.c_float) * self.num_inputs)()
            for i in range(self.num_inputs):
                arr = np.zeros(self.buffer_size, dtype=np.float32)
                input_arrays.append(arr)
                input_ptrs[i] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            # Convert numpy array to C pointers
            if input_data.shape[0] != self.num_inputs:
                raise ValueError(f"Input data has {input_data.shape[0]} channels, expected {self.num_inputs}")
            if input_data.shape[1] != self.buffer_size:
                raise ValueError(f"Input data has {input_data.shape[1]} samples, expected {self.buffer_size}")
            
            input_ptrs = (ctypes.POINTER(ctypes.c_float) * self.num_inputs)()
            for i in range(self.num_inputs):
                input_ptrs[i] = input_data[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Prepare output buffers
        output_arrays = []
        output_ptrs = (ctypes.POINTER(ctypes.c_float) * self.num_outputs)()
        for i in range(self.num_outputs):
            arr = np.zeros(self.buffer_size, dtype=np.float32)
            output_arrays.append(arr)
            output_ptrs[i] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call DSP compute
        self.lib.dsp_compute(
            self.dsp_instance,
            self.buffer_size,
            ctypes.cast(input_ptrs, ctypes.c_void_p),
            ctypes.cast(output_ptrs, ctypes.c_void_p)
        )
        
        # Convert output to numpy array
        output_data = np.stack(output_arrays)
        return output_data
    
    def render_audio(self, num_buffers: int = 10) -> np.ndarray:
        """
        Render multiple buffers of audio.
        
        Args:
            num_buffers: Number of buffers to render
            
        Returns:
            Concatenated audio data as numpy array (num_outputs x total_samples)
        """
        all_outputs = []
        
        for _ in range(num_buffers):
            buffer_output = self.compute()
            all_outputs.append(buffer_output)
        
        # Concatenate all buffers
        full_output = np.concatenate(all_outputs, axis=1)
        return full_output
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'dsp_instance') and self.dsp_instance:
            self.lib.deleteDSPInstanceWrapper(self.dsp_instance)
        if hasattr(self, 'factory') and self.factory:
            self.lib.deleteDSPFactoryWrapper(self.factory)


# Simple test function
def test_faust_dsp(dsp_file: str):
    """Test function to verify Faust DSP loading and computation."""
    try:
        faust = DspFaust(dsp_file, sample_rate=44100, buffer_size=512)
        
        # Test single buffer computation
        print("Testing single buffer computation...")
        single_buffer = faust.compute()
        print(f"Single buffer shape: {single_buffer.shape}")
        print(f"Output range: [{single_buffer.min():.6f}, {single_buffer.max():.6f}]")
        
        # Test multiple buffers
        print("\nTesting multiple buffer rendering...")
        multi_buffer = faust.render_audio(num_buffers=5)
        print(f"Multi buffer shape: {multi_buffer.shape}")
        print(f"Output range: [{multi_buffer.min():.6f}, {multi_buffer.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python faust_binding.py <dsp_file>")
        sys.exit(1)
    
    dsp_file = sys.argv[1]
    success = test_faust_dsp(dsp_file)
    sys.exit(0 if success else 1)
