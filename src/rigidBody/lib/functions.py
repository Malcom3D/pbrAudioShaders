import os
import random
import string
import trimesh
import numpy as np
from typing import Any, Tuple, Optional

def _soxel_grid_shape(grid_geometry, voxel_size):
    pass

def _mesh_to_obj(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, obj_file: str):
    """
    Convert an npz mesh file to Wavefront OBJ format.
    
    Args:
        vertices: numpy array with indexed vertices
        normals: numpy array with vertex normals
        faces: numpy array with faces indices
        obj_file: Path where the output .obj file will be saved
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=faces)
    mesh.export(obj_file, file_type='obj')
    return

def _load_mesh(config_obj: Any, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all pose sequence for an object."""
    if not 'ObjectConfig' in str(type(config_obj)):
        raise ValueError(f"{config_obj} is not of ObjectConfig type.")

    if config_obj.static:
        for filename in os.listdir(config_obj.obj_path):
            if filename.endswith('.npz'):
                filename = f"{config_obj.obj_path}/{filename}"
    elif not config_obj.static:
        items = os.listdir(config_obj.obj_path)
        items = [x for x in items if x.endswith('.npz')]
        filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
        filename = os.path.join(config_obj.obj_path, filenames[frame_idx])
    if not os.path.exists(filename):
        raise FileNotFoundError(f"NPZ file not found for {config_obj.name}: {filename}")
    data = np.load(filename)
    data.allow_pickle = False
    vertices = data[data.files[0]]
    normals = data[data.files[1]]
    faces = data[data.files[2]]
    return vertices, normals, faces

def _load_pose(config_obj: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Load all pose sequence for an object."""
    if not 'ObjectConfig' in str(type(config_obj)):
        raise ValueError(f"{config_obj} is not of ObjectConfig type.")

    pose_path = config_obj.pose_path
    obj_name = config_obj.name

    npz_file = os.path.join(pose_path, f"{obj_name}.npz")

    if not npz_file:
        raise ValueError(f"No pose files found for {obj_name} in {pose_path}")

    pose = np.load(npz_file)
    positions = pose[pose.files[0]]
    rotations = pose[pose.files[1]]

    return positions, rotations

def _euler_to_rotation_matrix(q: np.ndarray, degrees=False):
    """
    Convert Euler angles to rotation matrix (ZYX convention).
   
    Parameters:
    -----------
    q : np.ndarray
        yaw, pitch, roll
        yaw : float
            Rotation around Z axis (yaw)
        pitch : float
            Rotation around Y axis (pitch)
        roll : float
            Rotation around X axis (roll)
    degrees : bool, optional
        If True, input angles are in degrees. Default is False (radians).

    Returns:
    --------
    R : np.ndarray
        3x3 rotation matrix
    """
    roll, pitch, yaw = q

    if degrees:
        # Convert degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

    # Precompute trigonometric functions
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # ZYX rotation matrix (R = Rz(yaw) * Ry(pitch) * Rx(roll))
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    return R
