import numpy as np
import h5py
import tyro
from pathlib import Path
import joblib
from scipy.spatial.transform import Rotation
from loguru import logger
import mujoco
import os.path as osp
from typing import List, Tuple

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


def get_ordered_joint_indices(
    model: mujoco.MjModel, policy_joint_order: List[str]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Finds and returns the indices for qpos, qvel, and actuators in the MuJoCo
    model that correspond to the given list of joint names.
    """
    mj_qpos_indices: List[int] = []
    mj_qvel_indices: List[int] = []
    mj_actuator_indices: List[int] = []

    for joint_name in policy_joint_order:
        try:
            joint_id = model.joint(joint_name).id
        except KeyError:
            available_joints = [mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
            raise ValueError(
                f"Configuration Error: Joint '{joint_name}' "
                f"not found in the MuJoCo model. Available joints: {available_joints}"
            )

        mj_qpos_indices.append(model.jnt_qposadr[joint_id])
        mj_qvel_indices.append(model.jnt_dofadr[joint_id])

        actuator_found_for_joint = False
        for act_id in range(model.nu):
            if (
                model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[act_id, 0] == joint_id
            ):
                mj_actuator_indices.append(act_id)
                actuator_found_for_joint = True
                break

        if not actuator_found_for_joint:
            raise ValueError(
                f"Configuration Error: No actuator found for joint '{joint_name}'."
            )

    return mj_qpos_indices, mj_qvel_indices, mj_actuator_indices


def compute_feet_positions_with_mujoco(
    joints: np.ndarray,
    root_pos: np.ndarray, 
    root_quat: np.ndarray,
    joint_names: List[str]
) -> np.ndarray:
    """
    Use MuJoCo forward kinematics to compute feet positions
    
    Args:
        joints: Joint angles (N, J)
        root_pos: Root positions (N, 3)
        root_quat: Root quaternions (N, 4) in xyzw convention
        joint_names: List of joint names
    
    Returns:
        feet_positions: Computed feet positions using MuJoCo FK (N, 2, 3)
    """
    # Load MuJoCo model
    xml_path = osp.join(osp.dirname(__file__), "assets/robot_asset/g1/g1_29dof_anneal_23dof_disable_collision.xml")
    if not osp.exists(xml_path):
        logger.warning(f"MuJoCo XML not found at {xml_path}, skipping MuJoCo verification")
        return None
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Get joint indices
    mj_qpos_indices, _, _ = get_ordered_joint_indices(model, joint_names)
    
    feet_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_positions = []
    
    N = joints.shape[0]
    logger.info(f"Computing MuJoCo FK for {N} frames...")
    
    for t in range(N):
        # Set root pose (position + quaternion in wxyz format for MuJoCo)
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = np.concatenate([root_quat[t][3:4], root_quat[t][:3]])  # xyzw -> wxyz
        
        # Set joint angles  
        for i, angle in enumerate(joints[t]):
            data.qpos[mj_qpos_indices[i]] = angle
            
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Get feet positions
        frame_feet_pos = []
        for foot_name in feet_link_names:
            try:
                body_id = model.body(foot_name).id
                foot_pos = data.xpos[body_id].copy()
                frame_feet_pos.append(foot_pos)
            except:
                logger.warning(f"Body {foot_name} not found in MuJoCo model")
                frame_feet_pos.append(np.array([0, 0, 0]))
        
        feet_positions.append(frame_feet_pos)
    
    feet_positions = np.array(feet_positions)  # (N, 2, 3)
    logger.info(f"Successfully computed feet positions for {N} frames")
    
    return feet_positions


def main(
    input_path: Path,
    output_path: Path,
    name: str,
):
    with h5py.File(input_path, "r") as f:
        joints = f["joints"][:]  # (N, J)
        root_pos = f["root_pos"][:]  # (N, 3)
        root_quat = f["root_quat"][:]  # (N, 4) in xyzw convention

        joint_names = f.attrs["joint_names"]
        fps = f.attrs["fps"]

    # === Compute Feet Positions with MuJoCo Forward Kinematics ===
    logger.info("Computing feet positions using MuJoCo forward kinematics...")
    feet_positions = compute_feet_positions_with_mujoco(
        joints, root_pos, root_quat, joint_names
    )
    
    if feet_positions is not None:
        feet_link_z = feet_positions[:, :, 2]  # (N, 2) - z coordinates only
        print(f"MuJoCo computed feet z positions: {feet_link_z}")
    else:
        logger.error("Failed to compute feet positions with MuJoCo FK!")
        raise RuntimeError("Cannot proceed without feet position data")

    # --- 1. Infer initial position from feet height trajectory ---
    # We want to apply a transformation to root so that feet at their lowest point are at z=0.035
    
    # Find the minimum average feet height (lowest point)
    min_feet_height = np.min(feet_link_z)
    # NOTE (hsc): 它估计的不准，还是第几帧的平均值吧。先验在于，前面几帧和最后几帧应该是着地的
    min_feet_height = (feet_link_z[0:5].mean()+feet_link_z[-5:].mean())/2
    
    # Calculate the offset needed: we want min_feet_height to become 0.035
    # So the offset is: 0.035 - min_feet_height
    feet_offset = 0.035 - min_feet_height

    # Set initial position: xy at (0,0), z based on root position plus offset
    initial_pos_np = np.array([0.0, 0.0, root_pos[0, 2] + feet_offset])
    
    print(f"Minimum feet height: {min_feet_height:.4f}")
    print(f"Feet offset: {feet_offset:.4f}")
    print(f"Original root z: {root_pos[0, 2]:.4f}")
    print(f"Inferred initial position: {initial_pos_np}")
    
    root_pos_offset = initial_pos_np - root_pos[0]
    root_trans_offset = root_pos + root_pos_offset

    # --- 2. Process root and joint rotations to create pose_aa ---
    # Convert root quaternion (xyzw) to axis-angle
    root_rot = Rotation.from_quat(root_quat)  # (N,)
    
    # Calculate the yaw correction rotation
    # Get the forward direction from the first root rotation
    forward_dir = root_rot[0].apply([1, 0, 0])  # Apply rotation to [1,0,0] to get current forward
    forward_xy = forward_dir[:2]  # Take xy components
    current_yaw = np.arctan2(forward_xy[1], forward_xy[0])  # Calculate current yaw angle
    
    # We want to rotate by negative yaw to align forward direction to x-axis
    yaw_correction = Rotation.from_euler('z', -current_yaw)
    
    # Apply the yaw correction to all rotations
    root_rot_aligned = yaw_correction * root_rot  # (N,)
    root_aa = root_rot_aligned.as_rotvec()  # (N, 3)
    root_aa = root_aa[:, np.newaxis, :]  # Reshape to (N, 1, 3)

    # Apply the same rotation to root_trans_offset
    root_trans_offset = yaw_correction.apply(root_trans_offset)
    
    print(f"Original forward direction: {forward_dir}")
    print(f"Forward xy components: {forward_xy}")
    print(f"Current yaw angle: {np.degrees(current_yaw):.2f} degrees")
    print(f"Yaw correction: {np.degrees(-current_yaw):.2f} degrees")

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
