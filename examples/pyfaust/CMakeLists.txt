cmake_minimum_required(VERSION 3.12)
project(faust_module)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link statically
set(CMAKE_C_FLAGS "-static")
set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")

# Find pybind11
list(APPEND CMAKE_PREFIX_PATH "/home/malcom/Sources/pbrAudio.venv/lib/python3.11/site-packages/pybind11")
find_package(pybind11 REQUIRED)

# Find ALSA
find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

# Find ncurses and related libraries
find_library(NCURSES_LIBRARY ncurses)
find_library(TINFO_LIBRARY tinfo)
find_library(FORM_LIBRARY form)  # Sometimes needed
find_library(MENU_LIBRARY menu)  # Sometimes needed

# Find Faust libraries - try common locations
find_library(FAUST_LIBRARY 
    NAMES faust libfaust
    PATHS /home/malcom/Sources/faust-2.81.2/faust/build/lib
)

find_library(LLVM_FAUST_LIBRARY 
    Names faustwithllvm libfaustwithllvm
    PATHS /home/malcom/Sources/faust-2.81.2/faust/build/lib
)

# Add your source files
add_library(faust_module MODULE
    src/faust_module.cpp
    src/DspFaust.cpp
)

# Link with pybind11
target_link_libraries(faust_module PRIVATE
    pybind11::module
    ${ALSA_LIBRARIES}
)

# Include directories
target_include_directories(faust_module PRIVATE src)
target_include_directories(faust_module SYSTEM PRIVATE ${ALSA_INCLUDE_DIRS})

# Add Faust include directories if available
target_include_directories(faust_module SYSTEM PRIVATE 
    /home/malcom/Sources/faust-2.81.2/faust/architecture/faust
)

# Add Faust libraries if found
if(FAUST_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${FAUST_LIBRARY})
    message(STATUS "Found Faust library: ${FAUST_LIBRARY}")
else()
    message(WARNING "Faust library not found, trying to link without it")
endif()

if(LLVM_FAUST_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${LLVM_FAUST_LIBRARY})
    message(STATUS "Found Faust LLVM library: ${LLVM_FAUST_LIBRARY}")
else()
    message(WARNING "Faust LLVM library not found")
endif()

# Add ncurses and related libraries if found
if(NCURSES_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${NCURSES_LIBRARY})
    message(STATUS "Found ncurses: ${NCURSES_LIBRARY}")
endif()

if(TINFO_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${TINFO_LIBRARY})
    message(STATUS "Found tinfo: ${TINFO_LIBRARY}")
endif()

if(FORM_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${FORM_LIBRARY})
endif()

if(MENU_LIBRARY)
    target_link_libraries(faust_module PRIVATE ${MENU_LIBRARY})
endif()


# Set output properties
set_target_properties(faust_module PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Platform-specific settings
if(MSVC)
    target_compile_options(faust_module PRIVATE /EHsc)
else()
    target_compile_options(faust_module PRIVATE -Wno-attributes)
endif()
