import numpy as np
import h5py
import tyro
from pathlib import Path
import joblib
from scipy.spatial.transform import Rotation
from typing import Tuple


from loguru import logger

ASAP_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

ASAP_JOINT_AXIS = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]


def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.

    Args:
        h5file: An open h5py.File object.
        path: The current path in the HDF5 file.

    Returns:
        A nested dictionary with the data.
    """
    result = {}

    # Get the current group
    if path == "/":
        current_group = h5file
    else:
        current_group = h5file[path]

    # Load datasets and groups
    for key in current_group.keys():
        if path == "/":
            key_path = key
        else:
            key_path = f"{path}/{key}"

        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path)
        else:
            result[key] = h5file[key_path][:]

    # Load attributes of the current group
    for attr_key, attr_value in current_group.attrs.items():
        result[attr_key] = attr_value

    return result


def main(
    input_path: Path,
    output_path: Path,
    name: str,
    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.8),
):
    with h5py.File(input_path, "r") as f:
        joints = f["joints"][:]  # (N, J)
        root_pos = f["root_pos"][:]  # (N, 3)
        root_quat = f["root_quat"][:]  # (N, 4) in xyzw convention

        # link_pos and link_quat are not needed
        # link_pos = f["link_pos"][:]
        # link_quat = f["link_quat"][:]

        # link_names is not needed
        # link_names = f.attrs["link_names"]

        joint_names = f.attrs["joint_names"]
        fps = f.attrs["fps"]

    # --- 1. Process root translation ---
    initial_pos_np = np.array(initial_pos)
    root_pos_offset = initial_pos_np - root_pos[0]
    root_trans_offset = root_pos + root_pos_offset

    # --- 2. Process root and joint rotations to create pose_aa ---
    # Convert root quaternion (xyzw) to axis-angle
    root_rot = Rotation.from_quat(root_quat)  # (N,)
    first_inv = root_rot[0].inv()
    root_rot_aligned = first_inv * root_rot  # (N,)
    root_aa = root_rot_aligned.as_rotvec()  # (N, 3)
    root_aa = root_aa[:, np.newaxis, :]  # Reshape to (N, 1, 3)

    # Apply the same rotation to root_trans_offset
    root_trans_offset = first_inv.apply(root_trans_offset)

    # Reorder joints to match ASAP_JOINT_ORDER
    source_joint_names = joint_names  # No decoding needed as per feedback
    name_to_idx = {name: i for i, name in enumerate(source_joint_names)}
    reorder_indices = [name_to_idx[name] for name in ASAP_JOINT_ORDER]
    reordered_joints = joints[:, reorder_indices]  # (N, 23)

    # Convert joint angles to axis-angle representation
    asap_axis_np = np.array(ASAP_JOINT_AXIS)  # (23, 3)
    # Use broadcasting to multiply each angle by its corresponding axis vector
    joints_aa = reordered_joints[..., np.newaxis] * asap_axis_np  # (N, 23, 3)

    # Concatenate root and joint axis-angle arrays
    pose_aa = np.concatenate([root_aa, joints_aa], axis=1)  # (N, 24, 3)

    # --- 3. Assemble and save the final data structure ---
    asap_format_data = {
        "root_trans_offset": root_trans_offset.tolist(),
        "pose_aa": pose_aa.tolist(),
        "fps": float(fps),
    }

    final_data = {name: asap_format_data}

    joblib.dump(final_data, output_path)
    logger.info(f"Converted data saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
