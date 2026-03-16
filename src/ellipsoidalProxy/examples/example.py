"""
Example usage of ellipsoidalProxy module.
"""

import os
from physicsSolver import EntityManager, physicsEngine
from ellipsoidalProxy import EllipsoidalProxy

# Initialize entity manager with config
entity_manager = EntityManager("config.json")

# Create ellipsoidal proxy generator
# Objects smaller than 0.1m will be replaced with ellipsoidal proxies
proxy_gen = EllipsoidalProxy(entity_manager, size_threshold=0.1)

# Process all objects and replace small ones
proxy_objects = proxy_gen.process_all_objects()
print(f"Replaced {len(proxy_objects)} objects with ellipsoidal proxies")

# Integrate with physics solver (updates config to use proxy meshes)
proxy_gen.integrate_with_physics_solver()

# Now run the physics solver with the proxy meshes
physics_engine = physicsEngine(entity_manager)
physics_engine.bake()

print("Physics solver completed with ellipsoidal proxies")
