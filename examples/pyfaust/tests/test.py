import os, sys

sys.path.append(os.getcwd())

# Try using a dummy ALSA device
os.environ['ALSAAUDIO_DRIVER'] = 'dummy'
os.environ['AUDIODRIVER'] = 'dummy'

import faust_module

def test_basic_usage():
    # Create Faust instance
    faust = faust_module.DspFaust(48000, 512, False)
    
    # Start audio processing
    result = faust.start()
    print(f"Start result: {result}")
    
    # Get parameter information
    param_count = faust.getParamsCount()
    print(f"Number of parameters: {param_count}")
    
    # List all parameters
    for i in range(param_count):
        address = faust.getParamAddress(i)
        min_val = faust.getParamMin(i)
        max_val = faust.getParamMax(i)
        init_val = faust.getParamInit(i)
        current_val = faust.getParamValue(i)
        print(f"Param {i}: {address} (min: {min_val}, max: {max_val}, init: {init_val}, current: {current_val})")
    
    # Test MIDI
    voice = faust.keyOn(60, 100)  # Middle C, velocity 100
    print(f"Voice created: {voice}")
    
    faust.keyOff(60)
    
    # Test parameter setting
    if param_count > 0:
        first_param = faust.getParamAddress(0)
        original_value = faust.getParamValue(0)
        faust.setParamValue(0, original_value + 0.1)
        new_value = faust.getParamValue(0)
        print(f"Parameter 0: {original_value} -> {new_value}")
    
    # Get JSON descriptions
    json_ui = faust.getJSONUI()
    json_meta = faust.getJSONMeta()
    print(f"UI JSON length: {len(json_ui) if json_ui else 0}")
    print(f"Meta JSON length: {len(json_meta) if json_meta else 0}")
    
    # Get CPU load
    cpu_load = faust.getCPULoad()
    print(f"CPU Load: {cpu_load}")
    
    # Check if running
    is_running = faust.isRunning()
    print(f"Is running: {is_running}")
    
    # Stop processing
    faust.stop()

if __name__ == "__main__":
    test_basic_usage()

