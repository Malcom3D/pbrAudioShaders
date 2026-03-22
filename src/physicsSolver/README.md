physicsSolver
=============

The `physicsSolver` module is the hidden engine powering the `pbrAudioShaders` for the `pbrAudioRender` renderer. Its primary function is to reverse-engineer the physical dynamics of a 3D animation sequence. By analyzing a sequence of vertex-indexed Wavefront OBJ like files of triangulated mesh, it reconstructs the hidden forces, collisions, and contact events that are lost due to low frame rates. The module synthesizes this data into audio-force tracks and score files, which serve as the essential input for the physically-based sound synthesis engines.

## Core Concept

Traditional 3D animation is sampled at a discrete frame rate (e.g., 24 or 30 fps). This rate is far too low to capture the high-frequency details of collisions, impacts, and continuous contact (like sliding or scraping) needed for realistic audio. The `physicsSolver` bridges this gap by:

1.  **Analyzing Motion:** It treats the animated mesh sequence as a time-discrete signal.
2.  **Detecting Interactions:** It solves for the exact moments of collision, the contact geometry, and the resulting forces between objects.
3.  **Reconstructing Continuous Forces:** It interpolates and solves for the forces that occurred *between* the given frames.
4.  **Synthesizing Audio-Force Data:** It converts these forces and collisions into audio-rate signals 32bit float RAW audio files and structured data that is used by the pbrAudioShaders to render realistic sounds.

## Workflow

The `physicsSolver` operates in a multi-stage pipeline, processing each object and interaction:

1.  **Configuration:** The process is driven by a JSON configuration file (`Config`) that defines the system parameters, object paths (OBJ sequences), and material properties (`AcousticShader`).
2.  **Solving for Missing Frames:** For dynamic objects, it solves for the position, rotation, vertex positions, and vertex normals at the exact moments of collisions (using `PositionSolver`, `RotationSolver`, `VertexSolver`, `NormalSolver`).
3.  **Creating Continuous Trajectories:** It builds smooth, high-resolution trajectories from the sparse original and solved frames (`FlightPath`).
4.  **Calculating Distances:** For every pair of objects, it calculates the minimum distance over time to detect potential interactions (`DistanceSolver`).
5.  **Analyzing Collisions:** It identifies distinct collision events (impacts and continuous contacts) and classifies them based on the distance profile (`CollisionData`).
6.  **Computing Forces:** For each collision, it calculates the forces involved using a combination of rigid-body physics and Hertzian contact theory (`ForceSolver`).
7.  **Synthesizing Audio-Force:** It generates audio-rate force signals (impact envelopes, rolling and sliding/scraping noise) that is used as excitation sources for modal synthesis (`ForceSynth`).
8.  **Generating Score Data:** It creates a detailed "score" for each object, listing the time, type, and location of every contact event (`ScoreTrack`).

## Key Components

The module is organized into several submodules and classes:

### `core/`

This directory contains the main processing classes.

*   `physicsEngine`: The main orchestrator that runs the entire pipeline using Dask for parallel processing.
*   `EntityManager`: A singleton that acts as the central data hub, storing and managing all data (configs, trajectories, forces, collisions, etc.) for the duration of the solving process.
*   **Solvers (`PositionSolver`, `RotationSolver`, `VertexSolver`, `NormalSolver`)**: These classes compute the exact state (position, rotation, vertices, normals) of an object at the moment of a collision by interpolating between keyframes applying physics-based estimation.
*   `FlightPath`: Takes the solved frames and creates continuous, high-sample-rate `TrajectoryData` objects using spline interpolation (`CubicSpline`, `RotationSpline`). This is the primary data structure for querying an object's state at any arbitrary time.
*   `DistanceSolver`: Calculates the minimum distance between all object pairs and performs statistical analysis to classify types of contacts.
*   `ForceSolver`: Calculates the forces between colliding objects. It uses a `HertzianContact` model for detailed impact parameters.
*   `CollisionSolver`: Analyzes the geometry of a collision event, identifies the specific faces and vertices involved, and generates a `ScoreTrack` of events.
*   `ForceSynth`: Converts the computed forces into audio-rate signals, generating different tracks for impacts, sliding, scraping, and rolling, taking in account fractures events.

### `lib/`

This directory contains utility classes and data structures.

*   `trajectory_data`: Defines `TrajectoryData` (for final, interpolated motion) and `tmpTrajectoryData` (for temporary intermediate, solved frames).
*   `force_data`: Defines `ForceData` (for a single frame) and `ForceDataSequence` (for a continuous contact event).
*   `collision_data`: Defines `CollisionData` and the `CollisionType` enum.
*   `score_data`: Defines `ScoreEvent` and `ScoreTrack`, which prepare every contact event of an object for the pbrAudioShaders engines.
*   `acoustic_shader`: Defines the `AcousticShader` class, which holds material properties (density, Young's modulus, friction, roughness) and frequency-dependent acoustic coefficients (absorption, reflection, etc.).
*   `hertzian_contact`: Implements the `HertzianContact` model for detailed impact mechanics (contact radius, penetration depth, duration, impact impulse, etc.).
*   `contact_geometry`: Provides utilities for finding approximated contact points and normals between meshes.
*   `interpolator`: Provides specialized interpolators for frequency-dependent data.
*   `functions`: Contains helper functions for loading mesh and pose data.

### `utils/`

*   `config`: Defines the `Config`, `SystemConfig`, and `ObjectConfig` dataclasses for loading and managing the project's configuration file.

## Input Format

The `physicsSolver` expects a specific directory structure and file formats:

*   **Configuration (JSON):** A single JSON file defining the project. It must contain a `system` object and an `objects` list. Each object entry must specify:
    *   `idx`: A unique integer ID.
    *   `name`: The object's name.
    *   `obj_path`: Path to a directory containing a sequence of `.npz` files for the object's meshes (each file contains `vertices`, `normals`, `faces`).
    *   `pose_path`: Path to a `.npz` file containing the object's animation (`positions`, `rotations`).
    *   `static`: Boolean flag for static objects.
    *   `acoustic_shader`: A dictionary defining the object's material properties.

*   **Mesh Data (NPZ):** Each `.npz` file for a mesh should contain three arrays: `vertices` (Nx3), `normals` (Nx3), and `faces` (Mx3). These are standard for a triangulated mesh.

*   **Pose Data (NPZ):** The `.npz` file for the animation should contain two arrays: `positions` (Frames x 3) and `rotations` (Frames x 3) in XYZ Euler angles.

## Outputs

The solver generates the following outputs in the `pbrAudioCache/` directory:

*   `trajectories/`: Contains `TrajectoryData` objects (as `.pkl` files) for each object, representing their continuous motion.
*   `distances/`: Contains cached minimum distance data (as `.npz`) for all object pairs.
*   `forces_data/`: Contains `ForceDataSequence` objects (as `.pkl`) for all collision events.
*   `collisions/`: Contains `CollisionData` objects (as `.pkl`) for all detected collisions.
*   `modalvertices/`: Contains `ModalVertices` data (as `.json`) listing vertices involved in collisions.
*   `scoretracks/`: Contains `ScoreTrack` objects (as `.pkl`) with a timeline of all contact events.
*   `audio_force/`: Contains the final synthesized audio-force signals (as `.wav` and a multitrack `.json` project file) for each dynamic object.

## Dependencies

*   `numpy`
*   `scipy`
*   `trimesh`
*   `soundfile`
*   `resampy`
*   `numba`
*   `dask`

## Example Usage

### Configuration (JSON)

```json
{
  "system": {
    "sample_rate": 48000,
    "fps": 24,
    "collision_margin": 0.05
  },
  "objects": [
    {
      "idx": 0,
      "name": "sphere",
      "obj_path": "./objects/sphere/",
      "pose_path": "./poses/sphere.npz",
      "static": false,
      "acoustic_shader": {
        "density": 2700,
        "young_modulus": 70000000000,
        "poisson_ratio": 0.33,
        "friction": 0.5,
        "roughness": 0.1,
        "damping": 0.01
      }
    }
  ]
}
```



```python
import physicsSolver as ps

# 1. Create an EntityManager with the path to your config file
entity_manager = ps.EntityManager("path/to/your/config.json")

# 2. Instantiate and run the physics engine
engine = ps.physicsEngine(entity_manager)
engine.bake()

# 3. Results are now available in the cache directory
# (e.g., ./pbrAudioCache/audio_force/object_name.json and its .raw tracks)
```

The output audio-force tracks is loaded by pbrAudioShaders engines to synthesize the final audio.
