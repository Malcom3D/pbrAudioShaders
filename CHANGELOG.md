
# pbrAudioRender
## [v0.3.0] - 2026-11-04

### Added
- New `pbrAudioDAW` application complete with a Text User Interface (TUI).
- New 3D viewport widgets GUI and Viewport Panel integration into `pbrAudioBlender`.
- New processors in `pbrAudioSignals` including mono modal displacement to acoustic pressure processors, modal displacement, and audio forces processors.
- New methods in the `StorageEngine` class to edit metadata on the backend and support for writing audio data processing chains via the `SynthPostProcess` class.
- bz2 compression and decompression support for classes saved by `KleptoEntityStore`.
- New `pbrAudioFDTD` and `pbrAudioRay` engines with advanced open boundary conditions, ray tracing, wave propagation, and multiple output options (Ambisonic, Surround, Stereo, Stereo w/ HRTF).

### Changed
- Complete code rewrite and architectural shift of `pbrAudioStorage` to use parallel append and mmapped modes for the `Blosc2Backend` class.
- Reworked `physicsSolver` and `rigidBody` pipelines, replacing threading and dask with multiprocessing for compatibility and performance.
- Splitting of post-processing classes into the dedicated `pbrAudioPostProcess` module and aligned rigidBody/physicsSolver post-processing.

### Fixed
- Fixed data misalignments between signals and `ScoreEvent` objects originating from `physicsSolver` and `rigidBody`.
- Prevented Out-Of-Memory (OOM) issues in `ModalComposer`.
- Resolved various typos, syntax errors, missing backend registrations, and module import errors across storage, signals, and Blender integration components.

# pbrAudioRender
## [v0.3.0] - 2026-11-05

### Added
- **Acoustic Rendering Engines & Outputs:**
  - Added new `pbrAudioFDTD` solver and engine with advanced open boundary conditions, damping, wave propagation, resonance support (Helmholtz, parallel wall, and tube), and layer management.
  - Added `pbrAudioRay` ray tracer engine supporting transmission, reflection, scattering, and absorption interfaces along with ambisonic convolution and interpolation.
  - Added new output options for acoustic rendering, including Ambisonic, Surround, Stereo, and Stereo with HRTF support, alongside multi-channel spatial arrangements (tetrahedral, cube, dodecahedron, sphere).
  - Added mono and ambisonic renderers (`MonoRender`, `AmbisonicRender`) and planar/spherical acoustic sources.

- **Blender Integration & GUI:**
  - Integrated Viewport Panels and 3D viewport widgets into `pbrAudioBlender`.
  - Added new acoustic node tree components, frequency response files nodes, curves nodes, frequency points graphs, spatial frequency response nodes, and AcousticMaterialPreview nodes.
  - Added support for collision exports, proxy options, and custom cache paths with collision collections.

- **Audio Processors & Synthesizers:**
  - Added mono modal displacement to acoustic pressure processors, modal displacement processors, and audio forces processors.
  - Added `ForceSynth` to render audio-forces for stochastic physically-based models of contact forces, incorporating overlap-add synthesis and denoising/smoothing postprocessing.
  - Added fallback mechanisms for `mesh2faust` to generate approximate modal models.

- **Storage & Compression:**
  - Added bz2 compression and decompression support for classes saved by `KleptoEntityStore`.
  - Added metadata editing methods to the `StorageEngine` class and support for parallel append and mmapped modes in `Blosc2Backend`.

# pbrAudioRender
## [v0.3.0] - 2026-11-05

### Added
- Added `pbrAudioDAW` with a new TUI.
- Added `Viewport Panel` integration into `pbrAudioBlender`.
- Added new 3D viewport widgets GUI to `pbrAudioBlender`.
- Added new mono modal displacement to acoustic pressure processors in `pbrAudioSignals`.
- Added new modal displacement and audio forces processors in `pbrAudioSignals`.
- Added new methods to `StorageEngine` class to edit metadata on the backend.
- Added `track_group` handling to `pbrAudioSignals` to subdivide processors based on DAG node/edge goals.
- Added new `postProcess` class to describe `rigidBodySynth` post-processing in `pbrAudioStorage` metadata.
- Added `SynthPostProcess` class to write audio data processing chains to `pbrAudioStorage` metadata.
- Added `bz2` compression/decompression for classes saved by `KleptoEntityStore`.
- Added `StorageEngine._get_props()` to load `AudioDataType.chains` custom configurations.

### Changed
- Refactored `pbrAudioStorage` to switch to parallel append and mmapped mode for the `Blosc2Backend` class.
- Refactored `physicsSolver` and `rigidBody` post-process classes for alignment with new storage and signal features.
- Completed code rewrite for `pbrAudioStorage`.
- Refactored `postprocess` variable naming conventions.

### Fixed
- Fixed signal and `ScoreEvent` misalignment from `physicsSolver` and `rigidBody`.
- Fixed typos in `StorageEngine`.
- Fixed out-of-memory issues in `ModalComposer`.
- Fixed `KleptoEntityStore` typos and errors.
- Fixed `storage root_path` in `CollisionExporter` and `RenderExporter`.
- Fixed `enable_proxy_synth` indentation in `pbrAudio/properties/scenePG.py`.
- Fixed typos and errors in `Blosc2Backend` (removed `_rstore`).
- Fixed various `NameError` and `ModuleNotFoundError` issues in `pbrAudioStorage` and `physicsSolver` modules.
- Fixed various `SyntaxError` and indentation errors in `pbrAudioStorage`, `pbrAudioSignals`, and `pbrAudioBlender`.
