from __future__ import annotations

import time
from pathlib import Path
import h5py
import pickle
import os.path as osp

import jax
import jaxlie
import jaxls
import numpy as onp
import trimesh
import trimesh.ray
import tyro
import yourdfpy
from jax import numpy as jnp
import pyroki as pk
import jax_dataclasses as jdc
from typing import TypedDict


class RetargetingWeights(TypedDict):
    local_pose_cost_weight: float
    end_effector_cost_weight: float
    global_pose_cost_weight: float
    self_coll_factor_weight: float
    world_coll_factor_weight: float
    limit_cost_factor_weight: float
    smoothness_cost_factor_weight: float
    foot_skating_cost_weight: float
    ground_contact_cost_weight: float
    padding_norm_factor_weight: float
    hip_yaw_cost_weight: float
    hip_pitch_cost_weight: float
    hip_roll_cost_weight: float
    world_coll_margin: float


smpl_joint_names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

# Note that we now use _links_ instead of joints.
# g1_joint_names = (
#     'pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_foot_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'right_foot_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'logo_joint', 'head_joint', 'waist_support_joint', 'imu_joint', 'd435_joint', 'mid360_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_hand_palm_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_hand_palm_joint'
# )
g1_link_names = (
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_foot_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_foot_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "waist_support_link",
    "imu_link",
    "d435_link",
    "mid360_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)

smpl_joint_retarget_indices_to_g1 = []
g1_link_retarget_indices = []

for smpl_name, g1_name in [
    ("pelvis", "pelvis"),
    ("left_hip", "left_hip_roll_link"),
    ("right_hip", "right_hip_roll_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_wrist", "left_rubber_hand"),
    ("right_wrist", "right_rubber_hand"),
    ("left_ankle", "left_ankle_pitch_link"),
    ("right_ankle", "right_ankle_pitch_link"),
    ("left_shoulder", "left_shoulder_roll_link"),
    ("right_shoulder", "right_shoulder_roll_link"),
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
]:
    smpl_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_link_retarget_indices.append(g1_link_names.index(g1_name))

feet_link_pairs_g1 = [
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
]
ankle_link_pairs_g1 = [
    ("left_ankle", "left_ankle_pitch_link"),
    ("right_ankle", "right_ankle_pitch_link"),
]

smpl_feet_joint_retarget_indices_to_g1 = []
g1_feet_joint_retarget_indices = []

smpl_ankle_joint_retarget_indices_to_g1 = []
g1_ankle_joint_retarget_indices = []

for smpl_name, g1_name in feet_link_pairs_g1:
    smpl_feet_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_feet_joint_retarget_indices.append(g1_link_names.index(g1_name))

for smpl_name, g1_name in ankle_link_pairs_g1:
    smpl_ankle_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_ankle_joint_retarget_indices.append(g1_link_names.index(g1_name))


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


def save_dict_to_hdf5(h5file, dictionary, path="/"):
    """
    Recursively save a nested dictionary to an HDF5 file.

    Args:
        h5file: An open h5py.File object.
        dictionary: The nested dictionary to save.
        path: The current path in the HDF5 file.
    """
    for key, value in dictionary.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # If value is a dictionary, create a group and recurse
            if path == "/":
                group_path = key
            else:
                group_path = f"{path}/{key}"
            if group_path not in h5file:
                group = h5file.create_group(group_path)
            save_dict_to_hdf5(h5file, value, group_path)
        elif isinstance(value, onp.ndarray):
            if path == "/":
                dataset_path = key
            else:
                dataset_path = f"{path}/{key}"
            h5file.create_dataset(dataset_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            # Store scalars as attributes of the parent group
            if path == "/":
                h5file.attrs[key] = value
            else:
                h5file[path].attrs[key] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key}")


def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
    """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
    delta = delta * jnp.array([1, 1, 1, 0, 0, 0])  # Only update translation.
    return jaxls.SE3Var.retract_fn(transform, delta)


class SmplJointsScaleVarG1(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.ones(
        (len(smpl_joint_retarget_indices_to_g1), len(smpl_joint_retarget_indices_to_g1))
    ),
): ...

smpl_joint_scales_g1 = jnp.ones(
    (len(smpl_joint_retarget_indices_to_g1), len(smpl_joint_retarget_indices_to_g1))
)


def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Create a NxN connectivity matrix for N links.
    The matrix is marked Y if there is a direct kinematic chain connection
    between the two links, without bypassing the root link.
    """
    n = len(link_indices)
    conn_matrix = jnp.zeros((n, n))

    joint_indices = [
        robot.links.parent_joint_indices[link_indices[idx]] for idx in range(n)
    ]

    def is_direct_chain_connection(idx1: int, idx2: int) -> bool:
        """Check if two joints are connected in the kinematic chain without other retargeted joints between"""
        joint1 = joint_indices[idx1]
        joint2 = joint_indices[idx2]

        # Check path from joint2 up to root
        current = joint2
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint1:
                return True
            if parent in joint_indices:
                # Hit another retargeted joint before finding joint1
                break
            current = parent

        # Check path from joint1 up to root
        current = joint1
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint2:
                return True
            if parent in joint_indices:
                # Hit another retargeted joint before finding joint2
                break
            current = parent

        return False

    # Build symmetric connectivity matrix
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)  # Self-connection
        for j in range(i + 1, n):
            if is_direct_chain_connection(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix


def sanitize_joint_angles(
    joint_angles: jnp.ndarray,
    joint_limits_upper: jnp.ndarray,
    joint_limits_lower: jnp.ndarray,
):
    # joint_angles: (T, N)
    # joint_limits_upper: (N,)
    # joint_limits_lower: (N,)
    # return: (T, N)
    # Reshape to (T,N) if needed
    if len(joint_angles.shape) == 1:
        joint_angles = joint_angles.reshape(1, -1)

    # Broadcast limits to match joint angles shape
    joint_limits_upper = jnp.broadcast_to(joint_limits_upper, joint_angles.shape)
    joint_limits_lower = jnp.broadcast_to(joint_limits_lower, joint_angles.shape)

    # Assuming the joint angles are in the range of [-pi, pi]
    # If not, we need to normalize them to be within the range of [-pi, pi]
    joint_angles_mod = (joint_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # And then clip with the limits
    joint_angles_clipped = jnp.clip(
        joint_angles_mod, joint_limits_lower, joint_limits_upper
    )

    return joint_angles_clipped


@jaxls.Cost.create_factory(name="LocalPoseCost")
def local_pose_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    var_smpl_joints_scale: SmplJointsScaleVarG1,
    smpl_mask: jax.Array,
    robot: pk.Robot,
    keypoints: jax.Array,  # smpl joints --> keypoints
    local_pose_cost_weight: jax.Array,
) -> jax.Array:
    """Retargeting factor, with a focus on:
    - matching the relative joint/keypoint positions (vectors).
    - and matching the relative angles between the vectors.
    """
    robot_cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    T_world_root = var_values[var_T_world_root]
    T_world_link = T_world_root @ T_root_link

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    robot_joint_retarget_indices = jnp.array(g1_link_retarget_indices)
    smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
    robot_pos = T_world_link.translation()[jnp.array(robot_joint_retarget_indices)]

    # T_world_root = var_values[var_T_world_root]
    # robot_cfg = var_values[var_robot_cfg]
    # T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    # keypoints = T_world_root.inverse() @ keypoints

    # smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    # robot_joint_retarget_indices = jnp.array(g1_link_retarget_indices)
    # smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
    # robot_pos = T_root_link.translation()[jnp.array(robot_joint_retarget_indices)]

    # NxN grid of relative positions.
    delta_smpl = smpl_pos[:, None] - smpl_pos[None, :]
    delta_robot = robot_pos[:, None] - robot_pos[None, :]

    # Vector regularization.
    position_scale = var_values[var_smpl_joints_scale][..., None]
    residual_position_delta = (
        (delta_smpl - delta_robot * position_scale)
        * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
        * smpl_mask[..., None]
    )

    # Vector angle regularization.
    delta_smpl_normalized = delta_smpl  # / (jnp.linalg.norm(delta_smpl + 1e-6, axis=-1, keepdims=True) + 1e-6)
    delta_robot_normalized = delta_robot  # / (jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True) + 1e-6)
    residual_angle_delta = 1 - (delta_smpl_normalized * delta_robot_normalized).sum(
        axis=-1
    )

    # delta_smpl_normalized = delta_smpl / (jnp.linalg.norm(delta_smpl + 1e-6, axis=-1, keepdims=True) + 1e-6)
    # delta_robot_normalized = delta_robot / (jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True) + 1e-6)
    # residual_angle_delta = jnp.clip(1 - (delta_smpl_normalized * delta_robot_normalized).sum(axis=-1), min=0.1)

    residual_angle_delta = (
        residual_angle_delta * (1 - jnp.eye(residual_angle_delta.shape[0])) * smpl_mask
    )

    residual = (
        jnp.concatenate(
            [
                residual_position_delta.flatten(),
                residual_angle_delta.flatten(),
            ],
            axis=0,
        )
        * local_pose_cost_weight
    )
    return residual


@jaxls.Cost.create_factory(name="GlobalPoseCost")
def global_pose_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    var_smpl_joints_scale: SmplJointsScaleVarG1,
    smpl_mask: jax.Array,
    robot: pk.Robot,
    keypoints: jax.Array,  # smpl joints --> keypoints
    global_pose_cost_weight: jax.Array,
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)

    robot_joints = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    target_smpl_joint_pos = keypoints[jnp.array(smpl_joint_retarget_indices_to_g1)]
    target_robot_joint_pos = robot_joints.wxyz_xyz[..., 4:7][
        jnp.array(g1_link_retarget_indices)
    ]

    # center to the feetcenter
    center_between_smpl_feet_joint_pos = target_smpl_joint_pos.mean(
        axis=0, keepdims=True
    )
    center_between_robot_feet_joint_pos = target_robot_joint_pos.mean(
        axis=0, keepdims=True
    )
    recentered_target_smpl_joint_pos = (
        target_smpl_joint_pos - center_between_smpl_feet_joint_pos
    )
    recentered_target_robot_joint_pos = (
        target_robot_joint_pos - center_between_robot_feet_joint_pos
    )

    global_skeleton_scale = (var_values[var_smpl_joints_scale] * smpl_mask).mean()

    residual = jnp.abs(
        recentered_target_smpl_joint_pos * global_skeleton_scale
        - recentered_target_robot_joint_pos
    ).flatten()
    return residual * global_pose_cost_weight


@jaxls.Cost.create_factory(name="EndEffectorCost")
def end_effector_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    robot: pk.Robot,
    keypoints: jax.Array,  # smpl joints --> keypoints
    end_effector_cost_weight: jax.Array,
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)

    robot_joints = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    target_smpl_feet_joint_pos = keypoints[
        jnp.array(smpl_feet_joint_retarget_indices_to_g1)
    ]
    target_robot_feet_joint_pos = robot_joints.wxyz_xyz[..., 4:7][
        jnp.array(g1_feet_joint_retarget_indices)
    ]

    feet_residual = jnp.abs(
        target_smpl_feet_joint_pos - target_robot_feet_joint_pos
    ).flatten()
    residual = feet_residual

    return residual * end_effector_cost_weight


@jaxls.Cost.create_factory(name="FootSkatingCost")
def foot_skating_cost(
    var_values: jaxls.VarValues,
    var_robot_cfg_t0: jaxls.Var[jax.Array],
    var_robot_cfg_t1: jaxls.Var[jax.Array],
    robot: pk.Robot,
    contact_left_foot: jax.Array,
    contact_right_foot: jax.Array,
    foot_skating_cost_weight: jax.Array,
) -> jax.Array:
    robot_cfg_t0 = var_values[var_robot_cfg_t0]
    robot_cfg_t1 = var_values[var_robot_cfg_t1]
    Ts_root_links_t0 = robot.forward_kinematics(cfg=robot_cfg_t0)
    Ts_root_links_t1 = robot.forward_kinematics(cfg=robot_cfg_t1)

    feet_pos_t0 = Ts_root_links_t0[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)]
    feet_pos_t1 = Ts_root_links_t1[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)]
    ankle_pos_t0 = Ts_root_links_t0[..., 4:7][
        jnp.array(g1_ankle_joint_retarget_indices)
    ]
    ankle_pos_t1 = Ts_root_links_t1[..., 4:7][
        jnp.array(g1_ankle_joint_retarget_indices)
    ]

    left_foot_vel = jnp.abs(feet_pos_t1[0] - feet_pos_t0[0])
    right_foot_vel = jnp.abs(feet_pos_t1[1] - feet_pos_t0[1])

    residual = (left_foot_vel * contact_left_foot).flatten() + (
        right_foot_vel * contact_right_foot
    ).flatten()

    left_ankle_vel = jnp.abs(ankle_pos_t1[0] - ankle_pos_t0[0])
    right_ankle_vel = jnp.abs(ankle_pos_t1[1] - ankle_pos_t0[1])
    residual = (
        residual
        + (left_ankle_vel * contact_left_foot).flatten()
        + (right_ankle_vel * contact_right_foot).flatten()
    )

    return residual * foot_skating_cost_weight


@jaxls.Cost.create_factory(name="GroundContactCost")
def ground_contact_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    robot: pk.Robot,
    target_ground_z_for_frame: jax.Array,  # Shape (2,) - precomputed Z for left/right
    contact_mask_for_frame: jax.Array,  # Shape (2,) - 1 if contact, 0 otherwise
    ground_contact_cost_weight: jax.Array,
) -> jax.Array:
    """
    Penalizes vertical distance between robot feet/ankles and precomputed ground Z,
    only when contact is active.
    """

    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)
    robot_joints_world = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    # Get the indices of the relevant robot joints (feet or ankles)
    robot_joint_indices = jnp.array(g1_feet_joint_retarget_indices)  # Use feet for G1

    # Extract world positions of the relevant robot joints
    target_robot_joint_pos = robot_joints_world.wxyz_xyz[..., 4:7][
        robot_joint_indices
    ]  # (2, 3)

    # Extract Z coordinates
    robot_joint_z = target_robot_joint_pos[..., 2]  # (2,)

    # Calculate residual: (robot_z - target_ground_z) * contact_mask
    # We only penalize if the robot foot is above the target ground Z during contact.
    # Penalizing being below might push the foot through the visual mesh.
    # Let's use jnp.maximum(0, robot_joint_z - target_ground_z) to only penalize being above.
    # residual = jnp.maximum(0, robot_joint_z - target_ground_z_for_frame) * contact_mask_for_frame # Shape (2,)

    # Simpler: penalize absolute difference, masked by contact
    residual = (
        robot_joint_z - target_ground_z_for_frame
    ) * contact_mask_for_frame  # Shape (2,)

    return residual * ground_contact_cost_weight


@jaxls.Cost.create_factory(name="ScaleRegCost")
def scale_regularization(
    var_values: jaxls.VarValues,
    var_smpl_joints_scale: SmplJointsScaleVarG1,
) -> jax.Array:
    """Regularize the scale of the retargeted joints."""
    # Close to 1.
    res_0 = (var_values[var_smpl_joints_scale] - 1.0).flatten() * 1.0
    # Symmetric.
    res_1 = (
        var_values[var_smpl_joints_scale] - var_values[var_smpl_joints_scale].T
    ).flatten() * 100.0
    # Non-negative.
    res_2 = jnp.clip(-var_values[var_smpl_joints_scale], min=0).flatten() * 100.0
    return jnp.concatenate([res_0, res_1, res_2])


@jaxls.Cost.create_factory(name="HipYawCost")
def hip_yaw_and_pitch_cost(
    var_values: jaxls.VarValues,
    var_robot_cfg: jaxls.Var[jax.Array],
    hip_yaw_cost_weight: jax.Array,
    hip_pitch_cost_weight: jax.Array,
    hip_roll_cost_weight: jax.Array,
) -> jax.Array:
    """Regularize the hip yaw joints to be close to 0."""
    left_hip_pitch_joint_idx = 0
    left_hip_roll_joint_idx = 1
    left_hip_yaw_joint_idx = 2
    right_hip_pitch_joint_idx = 6
    right_hip_roll_joint_idx = 7
    right_hip_yaw_joint_idx = 8

    cfg = var_values[var_robot_cfg]
    residual = jnp.concatenate(
        [
            cfg[..., [left_hip_yaw_joint_idx]] * hip_yaw_cost_weight,
            cfg[..., [right_hip_yaw_joint_idx]] * hip_yaw_cost_weight,
            cfg[..., [left_hip_pitch_joint_idx]] * hip_pitch_cost_weight,
            cfg[..., [right_hip_pitch_joint_idx]] * hip_pitch_cost_weight,
            cfg[..., [left_hip_roll_joint_idx]] * hip_roll_cost_weight,
            cfg[..., [right_hip_roll_joint_idx]] * hip_roll_cost_weight,
        ],
        axis=-1,
    )
    return residual.flatten()


@jaxls.Cost.create_factory(name="RootSmoothnessCost")
def root_smoothness(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    root_smoothness_cost_weight: jax.Array,
) -> jax.Array:
    """Smoothness cost for the robot root pose."""
    return (
        var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
    ).log().flatten() * root_smoothness_cost_weight


@jdc.jit
def retarget_human_to_robot(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    heightmap: pk.collision.Heightmap,
    valid_timesteps: jax.Array,
    target_keypoints: jax.Array,
    Ts_world_root_smpl: jaxlie.SE3,
    smpl_mask: jax.Array,
    left_foot_contact: jax.Array,
    right_foot_contact: jax.Array,
    target_ground_z: jax.Array,
    weights: RetargetingWeights,
):
    num_timesteps = target_keypoints.shape[0]
    jnp_arange_timesteps = jnp.arange(num_timesteps)

    # Create variables for each timestep
    var_Ts_world_root = jaxls.SE3Var(jnp_arange_timesteps)
    var_joints = robot.joint_var_cls(jnp_arange_timesteps)
    var_smpl_joints_scale = SmplJointsScaleVarG1(0)
    default_smpl_joints_scale = smpl_joint_scales_g1

    costs: list[jaxls.Cost] = []

    costs.extend([
        local_pose_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["local_pose_cost_weight"] * valid_timesteps,
        ),
        end_effector_cost(
            var_Ts_world_root,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["end_effector_cost_weight"] * valid_timesteps,
        ),
        global_pose_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["global_pose_cost_weight"] * valid_timesteps,
        ),
    ])

    contact_mask = jnp.stack([left_foot_contact, right_foot_contact]).T # Shape (T, 2)
    target_ground_z = target_ground_z # Shape (T, 2)
    assert contact_mask.shape == (num_timesteps, 2)
    assert target_ground_z.shape == (num_timesteps, 2), (target_ground_z.shape, num_timesteps)

    costs.extend([
        ground_contact_cost(
            var_Ts_world_root,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            target_ground_z,
            contact_mask,
            weights["ground_contact_cost_weight"] * valid_timesteps,
        ),
        foot_skating_cost(
            robot.joint_var_cls(jnp.arange(1, num_timesteps)),
            robot.joint_var_cls(jnp.arange(0, num_timesteps-1)),
            jax.tree.map(lambda x: x[None], robot),
            left_foot_contact[1:],
            right_foot_contact[1:],
            weights["foot_skating_cost_weight"] * valid_timesteps[1:],
        ),
    ])

    costs.append(
        scale_regularization(
            var_smpl_joints_scale,
        )
    )

    @jaxls.Cost.create_factory(name="WorldCollisionCost")
    def world_collision_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        coll_weight: jax.Array,
    ) -> jax.Array:
        """
        World collision; we intentionally use a low weight --
        high enough to lift the robot up from the ground, but
        low enough to not interfere with the retargeting.
        """
        Ts_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        coll = robot_coll.at_config(robot, robot_cfg)
        coll = coll.transform(Ts_world_root)

        dist = pk.collision.collide(coll, heightmap)
        act = pk.collision.colldist_from_sdf(dist, activation_dist=weights["world_coll_margin"])
        return act.flatten() * coll_weight

    # Robot sanity costs:
    # - self-collision
    # - joint limits
    # - smoothness
    # - ...
    costs.extend([
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.1,
            weight=(
                weights["self_coll_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps,
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            weight=(
                weights["limit_cost_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, num_timesteps)),
            robot.joint_var_cls(jnp.arange(0, num_timesteps-1)),
            weight=(
                weights["smoothness_cost_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps[1:],
        ),
        # hip_yaw_and_pitch_cost(
        #     var_joints,
        #     weights["hip_yaw_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        #     weights["hip_pitch_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        #     weights["hip_roll_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        # ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, num_timesteps)),
            jaxls.SE3Var(jnp.arange(0, num_timesteps-1)),
            (
                2 * weights["smoothness_cost_factor_weight"]
            ) * (valid_timesteps[1:] * valid_timesteps[:-1]),
        ),
        world_collision_cost(
            var_Ts_world_root,
            var_joints,
            weights['world_coll_factor_weight'] * weights["padding_norm_factor_weight"] * valid_timesteps,
        )
    ])

    # Build and solve graph
    graph = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=[var_Ts_world_root, var_joints, var_smpl_joints_scale],
    ).analyze()

    solved_values, summary = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_Ts_world_root.with_value(Ts_world_root_smpl),
                var_joints,
                var_smpl_joints_scale.with_value(default_smpl_joints_scale),
            ]
        ),
        return_summary=True,
    )

    # Extract results
    optimized_T_world_root = solved_values[var_Ts_world_root]
    optimized_robot_cfg = solved_values[var_joints]
    optimized_scale = solved_values[var_smpl_joints_scale][0]

    return optimized_scale, optimized_robot_cfg, optimized_T_world_root, summary


def run_retargeting_core(
    urdf_path: Path,
    src_dir: Path,
    contact_dir: Path,
    output_path: Path,
    smpl_root_joint_idx: int = 0,
) -> None:

    print("--- 1. Loading Models and Data ---")
    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    with h5py.File(src_dir / "gravity_calibrated_keypoints.h5", "r") as f:
        keypoints_output = load_dict_from_hdf5(f)
    person_id = list(keypoints_output["joints"].keys())[0]
    unpadded_keypoints = jnp.array(keypoints_output["joints"][person_id])
    unpadded_root_orient = jnp.array(keypoints_output["root_orient"][person_id])
    num_timesteps = unpadded_keypoints.shape[0]

    print(f"--- 2. Preparing Data for Person {person_id} ({num_timesteps} frames) ---")

    initial_T_world_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(unpadded_root_orient[:, 0, :, :]),
        unpadded_keypoints[:, smpl_root_joint_idx, :],
    )

    megahunter_path = src_dir / "gravity_calibrated_megahunter.h5"
    if osp.exists(contact_dir) and osp.exists(megahunter_path):
        with h5py.File(megahunter_path, "r") as f:
            world_env = load_dict_from_hdf5(f)["our_pred_world_cameras_and_structure"]
        contact_estimation = {}
        for frame_name in world_env.keys():
            if osp.exists(osp.join(contact_dir, f'{frame_name}.pkl')):
                with open(osp.join(contact_dir, f'{frame_name}.pkl'), 'rb') as f:
                    contact_estimation[frame_name] = pickle.load(f) # Dict[person_id, Dict[left_foot_contact, right_foot_contact, frame_contact_vertices]]

        left_foot_contact_list = []
        right_foot_contact_list = []
        try:
            for frame_name in world_env.keys():
                if int(person_id) in contact_estimation.get(frame_name, {}):
                    left_foot_contact_list.append(contact_estimation[frame_name][int(person_id)]["left_foot_contact"])
                    right_foot_contact_list.append(contact_estimation[frame_name][int(person_id)]["right_foot_contact"])
            
            left_foot_contact = jnp.array(left_foot_contact_list)
            right_foot_contact = jnp.array(right_foot_contact_list)

            if len(left_foot_contact) > num_timesteps:
                left_foot_contact = left_foot_contact[:num_timesteps]
                right_foot_contact = right_foot_contact[:num_timesteps]
            else:
                left_foot_contact = jnp.pad(left_foot_contact, ((0, num_timesteps - len(left_foot_contact)),), mode="edge")
                right_foot_contact = jnp.pad(right_foot_contact, ((0, num_timesteps - len(right_foot_contact)),), mode="edge")

            assert left_foot_contact.shape == right_foot_contact.shape == (num_timesteps,)
            
        except Exception as e:
            print(f"\033[93mWarning: No feet contact estimation found for person {person_id}. Set to zeros.\033[0m")
            left_foot_contact = jnp.zeros((num_timesteps,))
            right_foot_contact = jnp.zeros((num_timesteps,))
    else:
        print(f"[Warning]: No feet contact estimation found for person {person_id}. Set to zeros.")
        left_foot_contact = jnp.zeros((num_timesteps,))
        right_foot_contact = jnp.zeros((num_timesteps,))

    # Then, pad all temporal data.
    padded_num_timesteps = ((num_timesteps - 1) // 100 + 1) * 100
    valid_timesteps = jnp.pad(
        jnp.ones(num_timesteps),
        (0, padded_num_timesteps - num_timesteps),
        constant_values=0,
    )
    padded_keypoints = jnp.pad(
        unpadded_keypoints,
        ((0, padded_num_timesteps - num_timesteps), (0, 0), (0, 0)),
        "edge",
    )
    padded_left_contact = jnp.pad(
        left_foot_contact,
        (0, padded_num_timesteps - num_timesteps),
        "edge",
    )
    padded_right_contact = jnp.pad(
        right_foot_contact,
        (0, padded_num_timesteps - num_timesteps),
        "edge",
    )

    background_mesh = None
    bg_mesh_path = src_dir / "background_mesh.obj"
    if bg_mesh_path.exists():
        background_mesh = trimesh.load(str(bg_mesh_path), force='mesh')
        if not isinstance(background_mesh, trimesh.Trimesh) or len(background_mesh.faces) == 0:
            print(f"\033[93mWarning: Background mesh {bg_mesh_path} could not be loaded or is empty.\033[0m")
            background_mesh = None
    else:
        print(f"\033[93mWarning: Background mesh file not found at {bg_mesh_path}.\033[0m")
        
    padded_target_ground_z = onp.zeros((padded_num_timesteps, 2))
    if background_mesh is not None:
        print("Precomputing target ground Z using raycasting...")
        
        relevant_smpl_indices = smpl_feet_joint_retarget_indices_to_g1
        print("Using SMPL feet for ground projection.")
        assert len(relevant_smpl_indices) == 2, "Expected 2 relevant SMPL indices (left/right)"

        human_joint_positions = onp.array(padded_keypoints[:, relevant_smpl_indices, :])  # (T, 2, 3)

        # Prepare ray origins slightly above the joints and flatten
        ray_origins_flat = (human_joint_positions + onp.array([0, 0, 0.1])).reshape(-1, 3) # (T*2, 3)
        num_rays = ray_origins_flat.shape[0]

        # --- Downward Raycast ---
        down_directions = onp.full_like(ray_origins_flat, [0, 0, -1.0]) # (T*2, 3)
        print("Performing downward raycast...")
        index_tri_down = background_mesh.ray.intersects_first(
            ray_origins=ray_origins_flat,
            ray_directions=down_directions
        ) # Shape: (T*2,) Returns -1 on miss

        # Initialize index_tri with downward results
        index_tri = index_tri_down.copy()

        # --- Upward Raycast for Misses ---
        missed_indices_down = onp.where(index_tri == -1)[0]
        if len(missed_indices_down) > 0:
            print(f"Downward raycast missed {len(missed_indices_down)} rays. Performing upward raycast for misses...")
            origins_for_upward = ray_origins_flat[missed_indices_down]
            up_directions = onp.full_like(origins_for_upward, [0, 0, 1.0])

            index_tri_up = background_mesh.ray.intersects_first(
                ray_origins=origins_for_upward,
                ray_directions=up_directions
            ) # Shape: (len(missed_indices_down),)

            # Update index_tri with upward hits
            valid_up_hits = onp.where(index_tri_up != -1)[0]
            if len(valid_up_hits) > 0:
                 print(f"Upward raycast found hits for {len(valid_up_hits)} previously missed rays.")
                 update_indices = missed_indices_down[valid_up_hits]
                 index_tri[update_indices] = index_tri_up[valid_up_hits]
            else:
                 print("Upward raycast did not find any additional hits.")

        # --- Calculate Triangle Center Z and Handle Final Misses ---
        final_missed_indices = onp.where(index_tri == -1)[0]
        if len(final_missed_indices) > 0:
             print(f"\033[93mWarning: {len(final_missed_indices)} rays missed in both directions.\033[0m")

        # Calculate fallback Z based on minimum *human* joint height for stability
        fallback_z = onp.min(human_joint_positions[..., 2]) if human_joint_positions.size > 0 else 0.0
        print(f"Using fallback Z: {fallback_z:.3f} for final missed rays.")

        target_ground_z_flat = onp.zeros(num_rays)
        valid_hit_indices = onp.where(index_tri != -1)[0]

        if len(valid_hit_indices) > 0:
            hit_triangle_indices = index_tri[valid_hit_indices]
            # Ensure triangle indices are within bounds
            valid_triangle_indices = hit_triangle_indices < len(background_mesh.triangles)
            if not onp.all(valid_triangle_indices):
                print(f"\033[91mError: Found invalid triangle indices after raycast! Check mesh integrity.\033[0m")
                 # Handle error appropriately, e.g., use fallback for invalid indices
                invalid_original_indices = valid_hit_indices[~valid_triangle_indices]
                index_tri[invalid_original_indices] = -1 # Mark as missed
                # Recalculate valid hits
                valid_hit_indices = onp.where(index_tri != -1)[0]
                hit_triangle_indices = index_tri[valid_hit_indices]

            projected_triangles = background_mesh.triangles[hit_triangle_indices] # (num_valid_hits, 3, 3)
            projected_triangles_center = projected_triangles.mean(axis=1) # (num_valid_hits, 3)
            projected_triangles_center_z = projected_triangles_center[:, 2] # (num_valid_hits,)
            target_ground_z_flat[valid_hit_indices] = projected_triangles_center_z

        # Apply fallback Z to all missed indices
        target_ground_z_flat[index_tri == -1] = fallback_z

        # Reshape back to (T, 2)
        padded_target_ground_z = target_ground_z_flat.reshape(-1, 2) # Reshape to (T, 2)

        print(f"Raycasting complete. Final Misses: {len(final_missed_indices)}")

    else:
        # If no mesh, fill with zeros (or fallback Z)
        fallback_z = onp.min(onp.array(padded_keypoints)[..., 2]) if padded_keypoints.size > 0 else 0.0
        padded_target_ground_z.fill(fallback_z)
        print("Skipping ground Z precomputation (no mesh).")

    padded_target_ground_z = jnp.array(padded_target_ground_z) # Convert to JAX array

    smpl_mask = create_conn_tree(robot, jnp.array(g1_link_retarget_indices))
    heightmap = (
        pk.collision.Heightmap.from_trimesh(background_mesh, x_bins=500, y_bins=500)
        if background_mesh is not None
        else pk.collision.Heightmap.from_trimesh(trimesh.Trimesh())
    )
    weights: RetargetingWeights = {
        "local_pose_cost_weight": 8.0,
        "end_effector_cost_weight": 5.0,
        "global_pose_cost_weight": 2.0,
        "self_coll_factor_weight": 1.0,
        "world_coll_factor_weight": 0.1,
        "limit_cost_factor_weight": 1000.0,
        "smoothness_cost_factor_weight": 10.0,
        "foot_skating_cost_weight": 10.0,
        "ground_contact_cost_weight": 1.0,
        "padding_norm_factor_weight": float(
            valid_timesteps.sum() / padded_num_timesteps
        ),
        "hip_yaw_cost_weight": 5.0,
        "hip_pitch_cost_weight": 0.0,
        "hip_roll_cost_weight": 0.0,
        "world_coll_margin": 0.01,
    }

    print("--- 3. Starting Full Optimization ---")
    start_time = time.time()
    # LOGIC FIX: Pass the UNPADDED initial_T_world_root to the solver.
    optimized_scale, raw_robot_cfg, optimized_T_world_root, summary = retarget_human_to_robot(
        robot,
        robot_coll,
        heightmap,
        valid_timesteps,
        padded_keypoints,
        initial_T_world_root,
        smpl_mask,
        padded_left_contact,
        padded_right_contact,
        padded_target_ground_z,
        weights,
    )
    jax.block_until_ready((optimized_scale, raw_robot_cfg, optimized_T_world_root, summary))
    print(f"Optimization finished in {time.time() - start_time:.2f}s")
    print(summary)

    print("--- 4. Sanitizing and Saving Results ---")
    sanitized_cfg = sanitize_joint_angles(
        raw_robot_cfg, robot.joints.upper_limits, robot.joints.lower_limits
    )
    export_data = {
        "joints": onp.array(sanitized_cfg[:num_timesteps]),
        "root_pos": onp.array(optimized_T_world_root.translation()[:num_timesteps]),
        "root_quat": onp.array(optimized_T_world_root.rotation().wxyz[:num_timesteps])[
            :, [1, 2, 3, 0]
        ],
    }
    with h5py.File(output_path, "w") as f:
        save_dict_to_hdf5(f, export_data)
    print(f"✅ Successfully saved results to {output_path}")


if __name__ == "__main__":
    tyro.cli(run_retargeting_core)
