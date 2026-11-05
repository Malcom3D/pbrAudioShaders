
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
