# Example usage of the fractureSound module

from physicsSolver import EntityManager, physicsEngine
from rigidBody import rigidBodyEngine
from fractureSound import fractureEngine

# Initialize entity manager with config
entity_manager = EntityManager("config.json")

# Run physics engine to compute trajectories, forces, etc.
physics_engine = physicsEngine(entity_manager)
physics_engine.bake()

# Run rigid body engine to compute modal models
rigidbody_engine = rigidBodyEngine(entity_manager)
rigidbody_engine.prebake()  # Compute modal models
rigidbody_engine.bake()      # Bake sounds

# Initialize fracture engine
fracture_engine = fractureEngine(entity_manager)

# Detect fracture events
# Original object index 0 fractures into fragments 1, 2, 3
fracture_events = fracture_engine.detect_fracture_events(
    obj_idx=0, 
    fragment_indices=[1, 2, 3]
)

# Process all fracture events (parallel)
fracture_engine.process_all_fractures()

# Fracture sounds are saved to cache_path/fracture_audio/
