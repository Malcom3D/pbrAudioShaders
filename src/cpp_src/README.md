# Enter ImpactShader/bin directory
cd ../ImpactShader/bin

# create DspFaust.cpp and DspFaust.h with:
/usr/bin/faust2api -dummy -dynamic -nozip -target ../../cpp_src/dsp-faust ../../cpp_src/dsp_files/modal_dummy.dsp

# build the libfaust_dynamic.so
c++ -std=c++11 -Ofast ../../cpp_src/dsp-faust/DspFaust.cpp /usr/local/lib/libfaust.a -fPIC -shared -static-libgcc -static-libstdc++ -lz `llvm-config --ldflags --libs all --system-libs`  `pkg-config --cflags --static --libs alsa sndfile` -o libfaust_dynamic.so

# build the faust_render_snd
c++ -std=c++11 ../../cpp_src/faust_render_snd.cpp libfaust_dynamic.so -I/usr/include/faust -static-libgcc -static-libstdc++ -lz `llvm-config --ldflags --libs all --system-libs`  `pkg-config --cflags --static --libs alsa sndfile` -o render_faust_snd
