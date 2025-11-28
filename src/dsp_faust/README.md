# create DspFaust.cpp and DspFaust.h with:
sudo /usr/bin/faust2api -dummy -dynamic -nozip -target ./dsp-faust dsp_files/modal_dummy.dsp

# build the libfaust_dynamic.so
c++ -std=c++11 -Ofast src/dsp-faust/DspFaust.cpp /usr/local/lib/libfaust.a -fPIC -shared -lz `llvm-config --ldflags --libs all --system-libs`  `pkg-config --cflags --static --libs alsa sndfile` -o lib/libfaust_dynamic.so

# build the faust_python.so
c++ -std=c++11 -fPIC -shared src/faust_python_wrapper.cpp -L. ./lib/libfaust_dynamic.so -I/usr/include/faust -o lib/faust_python.so

