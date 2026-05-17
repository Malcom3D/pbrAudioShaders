rigidBody
=========

The `rigidBody` module a sound synthesis engine of the `pbrAudioShaders` framework, responsible for generating physically plausible collision sounds for rigid body simulations. It synthesizes impact, rolling, scraping, and sliding sounds by combining Finite Element Analysis (FEA) modal models with physics-based force data from the `physicsSolver` module.

## Core Concept

When objects collide in a 3D animation, the resulting sound depends on the shape, material properties, contact geometry, and impact forces. The `rigidBody` module captures this complexity through a multi-stage pipeline:

1.  **Modal Analysis:** It converts 3D object meshes into modal synthesizer instruments using the [Faust Physical Modeling Toolkit](https://faust.grame.fr/), computing the object's natural frequencies, damping characteristics, and mode shapes from its physical material properties (Young's modulus, density, Poisson's's ratio).
2.  **Force-Driven Synthesis:** It uses the audio-rate force signals and collision scores computed by the `physicsSolver` module as excitation sources, driving the modal instruments to produce realistic collision sounds.
3.  **Resonance Modeling:** It accounts for both internal object resonances and resonances transmitted through contact between connected objects, enabling realistic coupling effects.
4.  **Multi-Contact Synthesis:** It handles simultaneous contacts at multiple vertices, combining forces from different collision types (impact, sliding, scraping and rolling) into a coherent audio output.

## Workflow

The `rigidBody` module operates in two main phases:

### Prebake Phase

1.  **Mesh-to-Modal Conversion (`Mesh2Modal`):** For each object (both dynamic and static), the module converts the 3D mesh into a modal model using `mesh2faust`, a command-line tool from the Faust Physical Modeling Toolkit. This generates `.lib` files containing the modal parameters (frequencies, gains, T60 decay times) for each vertex.
2.  **Score Composition (`ModalComposer`):** For each collision event, the module combines the audio-force tracks (impact, sliding, scraping and rolling) with the score events, assigning forces to specific vertices and computing the coupling data to handle the excitation forces of dynamic or static objects in contact that generate mutual resonances.

### Bake Phase

1.  **Instrument Building (`ModalLuthier`):** The module creates the actual synthesis instruments (`RigidBodySynth` and `ResonanceSynth`) from the modal `.lib` files. These instruments use Numba-accelerated modal banks to efficiently process audio in real-time.
2.  **Audio Rendering (`ModalPlayer`):** For each object, a `ModalPlayer` processes the score events sample-by-sample, driving the modal banks with the appropriate forces and coupling data. The output is rendered as multi-track audio files (rigidbody, sliding, scraping, rolling and resonance).
3.  **Sample Synchronization (`SampleCounter`):** Multiple players are synchronized ensuring that all objects are processed at the same sample index for consistent coupling and phase alignment.

## Key Components

### `core/`

This directory contains the main processing classes.

*   `rigidBodyEngine`: The main orchestrator that runs the entire synthesis pipeline using Dask for parallel processing. It manages the prebake and bake phases, coordinates the various sub-modules, and handles caching of intermediate results.
*   `Mesh2Modal`: Converts 3D object meshes into modal models by calling the `mesh2faust` tool. It handles both the primary modal model and an optional resonance model for each object.
*   `ModalComposer`: Processes collision events and combines force data with score events. It handles the different collision types and computes coupling data for connected objects.
*   `ModalLuthier`: Builds the synthesis instruments (`RigidBodySynth` and `ResonanceSynth`) from the modal `.lib` files. It sets up the modal banks with the appropriate vertex lists and sample rates.
*   `ModalPlayer`: The core synthesis engine that processes audio sample-by-sample. It reads score events, drives the modal banks, handles connected buffer coupling, and manages T60 decay tails for natural-sounding resonances.

### `lib/`

This directory contains utility classes and data structures.

*   `rigidbody_synth`: Implements the main modal synthesis instrument using Numba-accelerated modal banks. Each vertex has its own bank of modal oscillators, and forces are distributed across the contact vertices.
*   `resonance_synth`: Implements the resonance synthesis instrument, which models the internal resonances of an object when excited by contact forces. It supports contact area scaling and type-specific gain adjustments.
*   `modal_bank`: A Numba-accelerated implementation of a bank of modal oscillators. Each oscillator is a second-order resonant filter with frequency, gain, and T60 decay parameters.
*   `connected_buffer`: Manages the coupling between connected objects. When forces are transmitted through contact, they are written to a shared buffer that other objects can read and process.
*   `sample_counter`: A synchronization mechanism that ensures all `ModalPlayer` instances process the same sample index simultaneously, enabling proper coupling and phase alignment.

### `tools/`

This directory contains utility classes for binary tools.

*   `pym2f`: Python wrapper for the `mesh2faust` command-line tool. It handles file conversion, Rayleigh damping coefficient computation, and the generation of modal `.lib` files.
*   `faust_render`: Python wrapper for the pbrAudio `render_faust_snd` command-line tool, used for rendering Faust DSP files for audio preview.

## Input Format

The `rigidBody` module expects the following inputs:

*   **Configuration (JSON):** A project configuration file defining system parameters and object properties. Each object must specify:
    *   `idx`: A unique integer ID.
    *   `name`: The object's name (used for file naming).
    *   `obj_path`: Path to the object's mesh data.
    *   `static`: Boolean flag for static objects.
    *   `acoustic_shader`: Material properties including `young_modulus`, `poisson_ratio`, `density`, `damping`, `low_frequency`, and `high_frequency`.
    *   `resonance`: Optional boolean to enable resonance model generation.
    *   `connected`: Optional array of connected object indices for coupling.

*   **Mesh Data:** NPZ files containing vertices, normals, and faces. Can be converted from vertex indexed (landmarks) Wavefront OBJ files.

*   **Force Data:** Audio-rate force signals (RAW PCM FLOAT32 files) and score events (PKL files) from the `physicsSolver` module.

## Outputs

The module generates the following outputs in the cache directory:

*   `dsp/`: Contains the modal `.lib` files generated by `mesh2faust` for each object.
*   `modal_player/`: Contains the rendered audio tracks for each object:
    *   `{object_name}_rigidbody.raw`: Direct modal synthesis output.
    *   `{object_name}_resonance.raw`: Resonance synthesis output.
    *   `{object_name}_sliding.raw`: Sliding sound track.
    *   `{object_name}_scraping.raw`: Scraping sound track.
    *   `{object_name}_rolling.raw`: Rolling sound track.
    *   `{object_name}_{track}.json`: Project metadata for each track.
*   `status/`: Progress tracking files for the prebake and bake phases.

## Dependencies

*   `numpy`
*   `numba`
*   `soundfile`
*   `dask`
*   `scipy`
*   `trimesh`
*   [Faust](https://faust.grame.fr/) `mesh2faust` and pbrAudio `render_faust_snd` tools sources and multiplatform binary (Linux, Windows and macOS) are shipped with pbrAudioShader.

## Example Usage

### Configuration (JSON)

```json
{
  "system": {
    "sample_rate":": 48000,
    "fps": 24,
    "fps_base": 1,
    "subframes": 1,
    "cache_path": "./pbrAudioCache",
    "modal_modes": 100,
    "bit_depth": 32,
    "file_format": "wav"
  },
  "objects": [
    {
      "idx": 0,
      "name": "bowl",
      "obj_path": "./objects/bowl/",
      "static": false,
      "resonance": true,
      "resonance_modes": 50,
      "acoustic_shader": {
        "density": 2700,
        "young_modulus": 70000000000,
        "poisson_ratio": 0.33,
        "damping": 0.01,
        "low_frequency": 100,
        "high_frequency": 5000
      }
    }
  ]
}
```

### Python Usage

```python
import pbrAudioShaders.rigidBody as rb
from physicsSolver import EntityManager

# 1. Create an EntityManager with the path to your config file
entity_manager = EntityManager("path/to/your/config.json")

# 2. Instantiate and run the rigid body engine
engine = rb.rigidBodyEngine(entity_manager)
engine.prebake()  # Generate modal models and compose scores
engine.bake()     # Render audio

# 3. Audio tracks are now available in the cache directory
# (e.g., ./pbrAudioCache/modal_player/bowl_rigidbody.wav)
```

The output audio tracks can be mixed together and integrated into the final audio output, with different tracks representing different physical sound generation mechanisms.
