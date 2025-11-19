#include <pybind11/pybind11.h>
#include "dsp-faust/DspFaust.h"

namespace py = pybind11;

PYBIND11_MODULE(dsp_faust, m) {
    m.doc() = "Python bindings for DspFaust audio processing";
    
    py::class_<DspFaust>(m, "DspFaust")
        // Static DSP constructors
        .def(py::init<bool>(), py::arg("auto_connect") = true)
        .def(py::init<int, int, bool>(), py::arg("SR"), py::arg("BS"), py::arg("auto_connect") = true)
        
        // Dynamic DSP constructor
#ifdef DYNAMIC_DSP
        .def(py::init<const std::string&, int, int, bool>(), 
             py::arg("dsp_content"), py::arg("SR"), py::arg("BS"), py::arg("auto_connect") = true)
#endif
        
        // Audio processing methods
        .def("start", &DspFaust::start, "Start the audio processing")
        .def("stop", &DspFaust::stop, "Stop the audio processing")
        .def("isRunning", &DspFaust::isRunning, "Check if audio is running")
        
        // MIDI methods
        .def("keyOn", &DspFaust::keyOn, 
             "Send MIDI note on", py::arg("pitch"), py::arg("velocity"))
        .def("keyOff", &DspFaust::keyOff, 
             "Send MIDI note off", py::arg("pitch"))
        .def("newVoice", &DspFaust::newVoice, "Create a new polyphonic voice")
        .def("deleteVoice", &DspFaust::deleteVoice, "Delete a polyphonic voice", py::arg("voice"))
        .def("allNotesOff", &DspFaust::allNotesOff, "Terminate all active voices", py::arg("hard") = false)
        .def("propagateMidi", &DspFaust::propagateMidi, 
             "Propagate raw MIDI message", 
             py::arg("count"), py::arg("time"), py::arg("type"), 
             py::arg("channel"), py::arg("data1"), py::arg("data2"))
        
        // UI control methods
        .def("setParamValue", py::overload_cast<const char*, float>(&DspFaust::setParamValue), 
             "Set parameter value by address", py::arg("address"), py::arg("value"))
        .def("setParamValue", py::overload_cast<int, float>(&DspFaust::setParamValue), 
             "Set parameter value by id", py::arg("id"), py::arg("value"))
        .def("getParamValue", py::overload_cast<const char*>(&DspFaust::getParamValue), 
             "Get parameter value by address", py::arg("address"))
        .def("getParamValue", py::overload_cast<int>(&DspFaust::getParamValue), 
             "Get parameter value by id", py::arg("id"))
        
        // Voice parameter methods
        .def("setVoiceParamValue", py::overload_cast<const char*, uintptr_t, float>(&DspFaust::setVoiceParamValue), 
             "Set voice parameter value by address", py::arg("address"), py::arg("voice"), py::arg("value"))
        .def("setVoiceParamValue", py::overload_cast<int, uintptr_t, float>(&DspFaust::setVoiceParamValue),
             "Set voice parameter value by id", py::arg("id"), py::arg("voice"), py::arg("value"))
        .def("getVoiceParamValue", py::overload_cast<const char*, uintptr_t>(&DspFaust::getVoiceParamValue), 
             "Get voice parameter value by address", py::arg("address"), py::arg("voice"))
        .def("getVoiceParamValue", py::overload_cast<int, uintptr_t>(&DspFaust::getVoiceParamValue), 
             "Get voice parameter value by id", py::arg("id"), py::arg("voice"))
        
        // Metadata methods
        .def("getParamsCount", &DspFaust::getParamsCount, "Get number of parameters")
        .def("getParamAddress", &DspFaust::getParamAddress, 
             "Get parameter address by index", py::arg("id"))
        .def("getVoiceParamAddress", &DspFaust::getVoiceParamAddress, 
             "Get voice parameter address by index", py::arg("id"), py::arg("voice"))
        .def("getParamMin", py::overload_cast<const char*>(&DspFaust::getParamMin), 
             "Get parameter minimum value by address", py::arg("address"))
        .def("getParamMin", py::overload_cast<int>(&DspFaust::getParamMin), 
             "Get parameter minimum value by id", py::arg("id"))
        .def("getParamMax", py::overload_cast<const char*>(&DspFaust::getParamMax), 
             "Get parameter maximum value by address", py::arg("address"))
        .def("getParamMax", py::overload_cast<int>(&DspFaust::getParamMax), 
             "Get parameter maximum value by id", py::arg("id"))
        .def("getParamInit", py::overload_cast<const char*>(&DspFaust::getParamInit), 
             "Get parameter default value by address", py::arg("address"))
        .def("getParamInit", py::overload_cast<int>(&DspFaust::getParamInit), 
             "Get parameter default value by id", py::arg("id"))
        
        // JSON methods
        .def("getJSONUI", &DspFaust::getJSONUI, "Get JSON UI description")
        .def("getJSONMeta", &DspFaust::getJSONMeta, "Get JSON metadata description")
        
        // Sensor methods
        .def("propagateAcc", &DspFaust::propagateAcc, 
             "Propagate accelerometer", py::arg("acc"), py::arg("v"))
        .def("setAccConverter", &DspFaust::setAccConverter, 
             "Set accelerometer converter", 
             py::arg("id"), py::arg("acc"), py::arg("curve"), 
             py::arg("amin"), py::arg("amid"), py::arg("amax"))
        .def("propagateGyr", &DspFaust::propagateGyr, 
             "Propagate gyroscope", py::arg("gyr"), py::arg("v"))
        .def("setGyrConverter", &DspFaust::setGyrConverter, 
             "Set gyroscope converter", 
             py::arg("id"), py::arg("gyr"), py::arg("curve"), 
             py::arg("amin"), py::arg("amid"), py::arg("amax"))
        
        // OSC methods
        .def("configureOSC", &DspFaust::configureOSC, 
             "Configure OSC", 
             py::arg("xmit"), py::arg("inport"), py::arg("outport"), 
             py::arg("errport"), py::arg("address"))
        .def("isOSCOn", &DspFaust::isOSCOn, "Check if OSC is enabled")
        
        // Utility methods
        .def("getCPULoad", &DspFaust::getCPULoad, "Get CPU load")
        .def("getScreenColor", &DspFaust::getScreenColor, "Get screen color")
        
        // Metadata access
        .def("getMetadata", py::overload_cast<const char*, const char*>(&DspFaust::getMetadata), 
             "Get metadata by address", py::arg("address"), py::arg("key"))
        .def("getMetadata", py::overload_cast<int, const char*>(&DspFaust::getMetadata), 
             "Get metadata by id", py::arg("id"), py::arg("key"));
}
