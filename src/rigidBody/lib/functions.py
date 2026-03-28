import os
import re
import random
import string
import trimesh
import numpy as np
import numba as nb
from typing import Any, Tuple, Optional, List, Union, Dict

def _mesh_to_obj(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, obj_file: str, resonance: bool = False):
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

    # Create simplified convex hull for resonance model
    if resonance:
        simplified = mesh.simplify_quadric_decimation(percent=0.6, aggression=0)
        simplified.export(f"{obj_file.removesuffix('.obj')}_resonance.obj", file_type='obj')

    # Export as obj
    mesh.export(obj_file, file_type='obj')
    return

def _load_mesh(obj_config, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh for an object at a given frame index."""
    if obj_config.static:
        # For static, load once
        filename = obj_config.obj_path
        # Assume single npz file or directory with one file
        if os.path.isdir(obj_config.obj_path):
            files = [f for f in os.listdir(obj_config.obj_path) if f.endswith('.npz')]
            filename = os.path.join(obj_config.obj_path, files[0])
        else:
            filename = obj_config.obj_path
    else:
        # For dynamic, load sequence
        items = os.listdir(obj_config.obj_path)
        items = [x for x in items if x.endswith('.npz')]
        filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
        filename = os.path.join(obj_config.obj_path, filenames[frame_idx])

    data = np.load(filename, allow_pickle=False)
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

def _generate_band_frequencies(lowest_frequency: float, higher_frequency: float, steps_per_octave: int):
    """
    Generate frequencies from lowest_frequency to higher_frequency with specified steps per octave
    """
    frequencies = []
    current_freq = lowest_frequency
    
    # Calculate the frequency ratio for one step
    step_ratio = 2 ** (1 / steps_per_octave)

    while current_freq <= higher_frequency:
        frequencies.append(current_freq)
        current_freq *= step_ratio

    return frequencies

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


def _parse_lib(lib_content: str):
    """
    Parse mesh2faust .lib file content

    Args:
        lib_content: String content of the .lib file

    Returns:
        Dictionary with modal parameters
    """
    freq_pattern = r'modeFreqsUnscaled.*?=.*?ba\.take.*?$$(.*?)$$'
    t60_pattern = r'modesT60s.*?=.*?t60Scale.*?ba\.take.*?$$(.*?)$$'
    gain_pattern = r'modesGains.*?=.*?waveform\{(.*?)\}'
    parentesis_match = r'\(([^()]+)\)'
    tuple_match = r'\d+\.?\d+'

    with open(lib_content, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Extract frequencies from modeFreqsUnscaled
            freq_match = re.search(freq_pattern, line, re.DOTALL)
            if not freq_match == None:
                freq_tuple_match = re.findall(tuple_match, freq_match.group())
                frequencies = [float(f) for f in freq_tuple_match]
            # Extract T60 values
            t60_match = re.search(t60_pattern, line, re.DOTALL)
            if not t60_match == None:
                t60_par_match = re.findall(parentesis_match, t60_match.group())
                t60_tuple_match = re.findall(tuple_match, t60_par_match[1])
                t60s = [float(f) for f in t60_tuple_match]
            # Extract gains - this is complex due to the large waveform
            gain_match = re.search(gain_pattern, line, re.DOTALL)
            if not gain_match == None:
                gain_tuple_match = re.findall(gain_pattern, gain_match.group())
                gain_tuple_match = re.sub("'", "", gain_tuple_match[0])
                gains = [float(f) for f in gain_tuple_match.split(",")]

    return {
        'frequencies': np.array(frequencies),
        't60s': np.array(t60s),
        'gains': np.hsplit(np.array(gains), len(gains)/len(frequencies)),
        'nModes': len(frequencies)
    }

def _update_status(file_path: str, progress: Optional[int] = None):
    with open(file_path, 'w') as file:
        if not progress == None:
            file.write(f"{progress}")

def _cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        (azimuth, elevation, radius) in degrees and units
    """
    radius = np.sqrt(x*x + y*y + z*z)

    if radius == 0:
        return 0.0, 0.0, 0.0

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / radius)

    return azimuth, elevation, radius

@nb.jit(nopython=True)
def _trilinear_interpolate(field: np.ndarray, position: Tuple[float, float, float]) -> float:
    """Fast trilinear interpolation using numba"""
    i, j, k = position
   
    i0, j0, k0 = int(np.floor(i)), int(np.floor(j)), int(np.floor(k))
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
   
    # Check bounds
    if (i0 < 0 or i1 >= field.shape[0] or
        j0 < 0 or j1 >= field.shape[1] or
        k0 < 0 or k1 >= field.shape[2]):
        return 0.0
   
    # Calculate interpolation weights
    di, dj, dk = i - i0, j - j0, k - k0
    di1, dj1, dk1 = 1.0 - di, 1.0 - dj, 1.0 - dk

    # Get the 8 corner values
    v000 = field[i0, j0, k0]
    v001 = field[i0, j0, k1]
    v010 = field[i0, j1, k0]
    v011 = field[i0, j1, k1]
    v100 = field[i1, j0, k0]
    v101 = field[i1, j0, k1]
    v110 = field[i1, j1, k0]
    v111 = field[i1, j1, k1]

    # Trilinear interpolation
    c00 = v000 * di1 + v100 * di
    c01 = v001 * di1 + v101 * di
    c10 = v010 * di1 + v110 * di
    c11 = v011 * di1 + v111 * di

    c0 = c00 * dj1 + c10 * dj
    c1 = c01 * dj1 + c11 * dj

    value = c0 * dk1 + c1 * dk

    return value

def _degrees_to_radians(phase_coeffs, input_unit='auto'):
    """
    Verify if phase coefficients are in normalized radians and convert if needed.

    Parameters:
    -----------
    phase_coeffs : np.array
        Array of phase coefficients
    input_unit : str, optional
        Input unit type: 'radians', 'degrees', 'gradians', or 'auto' (default)
        If 'auto', the function will attempt to detect the unit

    Returns:
    --------
    normalized_phase : np.array
        normalized_phase : phase coefficients in normalized radians [-π, π]
    """
    # Make a copy to avoid modifying the original array
    phase = phase_coeffs.copy()
    was_normalized = False

    if input_unit == 'auto':
        # Auto-detection logic
        max_abs = np.max(np.abs(phase))

        if max_abs <= np.pi:
            # Likely already in radians
            original_unit = 'radians'
        elif max_abs <= 180:
            # Likely in degrees
            original_unit = 'degrees'
            was_normalized = True
        elif max_abs > 180:
            # Likely in gradians
            original_unit = 'gradians'
            was_normalized = True
        else:
            # Default to radians if uncertain
            original_unit = 'radians'
            print("Error: Could not auto-detect unit with certainty.")
            return None
    else:
        original_unit = input_unit
        was_normalized = (input_unit != 'radians')

    # Perform conversion if needed
    if original_unit == 'degrees':
        # Convert degrees to radians
        phase = np.deg2rad(phase)
        was_normalized = True
    elif original_unit == 'gradians':
        # Convert gradians to radians (200 gradians = 180 degrees = π radians)
        phase = phase * (np.pi / 200)
        was_normalized = True

    # Normalize to [-π, π] range
    if was_normalized or input_unit == 'radians':
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

    return phase
