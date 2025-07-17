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

# ===================================================================================
# 1. UTILITIES & SETUP (Verbatim from original)
# ===================================================================================


def load_dict_from_hdf5(h5file, path="/"):
    result = {}
    current_group = h5file[path] if path != "/" else h5file
    for key in current_group.keys():
        key_path = f"{path}/{key}" if path != "/" else key
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path)
        else:
            result[key] = h5file[key_path][:]
    for attr_key, attr_value in current_group.attrs.items():
        result[attr_key] = attr_value
    return result


def save_dict_to_hdf5(h5file, dictionary, path="/"):
    for key, value in dictionary.items():
        if value is None:
            continue
        full_path = f"{path}/{key}" if path != "/" else key
        if isinstance(value, dict):
            group = h5file.create_group(full_path)
            save_dict_to_hdf5(h5file, value, full_path)
        elif isinstance(value, onp.ndarray):
            h5file.create_dataset(full_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            parent_group = h5file[path] if path != "/" else h5file
            parent_group.attrs[key] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key}")


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

smpl_joint_retarget_indices_to_g1, g1_link_retarget_indices = [], []
key_links = [
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
]
for smpl_name, g1_name in key_links:
    smpl_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_link_retarget_indices.append(g1_link_names.index(g1_name))
smpl_feet_joint_retarget_indices_to_g1, g1_feet_joint_retarget_indices = [], []
for smpl_name, g1_name in [
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
]:
    smpl_feet_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_feet_joint_retarget_indices.append(g1_link_names.index(g1_name))
smpl_ankle_joint_retarget_indices_to_g1, g1_ankle_joint_retarget_indices = [], []
for smpl_name, g1_name in [
    ("left_ankle", "left_ankle_pitch_link"),
    ("right_ankle", "right_ankle_pitch_link"),
]:
    smpl_ankle_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_ankle_joint_retarget_indices.append(g1_link_names.index(g1_name))


def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    n, joint_indices = len(link_indices), [
        robot.links.parent_joint_indices[link_indices[idx]]
        for idx in range(len(link_indices))
    ]
    conn_matrix = jnp.zeros((n, n))

    def is_direct_chain(i1, i2):
        j1, j2 = joint_indices[i1], joint_indices[i2]
        c = j2
        while c != -1:
            p = robot.joints.parent_indices[c]
            if p == j1:
                return True
            if p in joint_indices:
                break
            c = p
        c = j1
        while c != -1:
            p = robot.joints.parent_indices[c]
            if p == j2:
                return True
            if p in joint_indices:
                break
            c = p
        return False

    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)
        for j in range(i + 1, n):
            if is_direct_chain(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0).at[j, i].set(1.0)
    return conn_matrix


def sanitize_joint_angles(joint_angles, upper, lower):
    mod = (joint_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.clip(
        mod, jnp.broadcast_to(lower, mod.shape), jnp.broadcast_to(upper, mod.shape)
    )


# ===================================================================================
# 2. CORE OPTIMIZATION (Faithful Reproduction)
# ===================================================================================
class SmplJointsScaleVarG1(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.ones(
        (len(smpl_joint_retarget_indices_to_g1), len(smpl_joint_retarget_indices_to_g1))
    ),
): ...


@jaxls.Cost.create_factory
def local_pose_cost(
    var_values,
    var_T_world_root,
    var_robot_cfg,
    var_smpl_joints_scale,
    smpl_mask,
    robot,
    keypoints,
    weight,
):
    robot_cfg, T_world_root = var_values[var_robot_cfg], var_values[var_T_world_root]
    T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    smpl_pos, robot_pos = (
        keypoints[jnp.array(smpl_joint_retarget_indices_to_g1)],
        T_world_link.translation()[jnp.array(g1_link_retarget_indices)],
    )
    delta_smpl, delta_robot = (
        smpl_pos[:, None] - smpl_pos[None, :],
        robot_pos[:, None] - robot_pos[None, :],
    )
    position_scale = var_values[var_smpl_joints_scale][..., None]
    residual_pos = (
        (delta_smpl - delta_robot * position_scale)
        * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
        * smpl_mask[..., None]
    )
    delta_smpl_norm, delta_robot_norm = delta_smpl / (
        jnp.linalg.norm(delta_smpl, axis=-1, keepdims=True) + 1e-6
    ), delta_robot / (jnp.linalg.norm(delta_robot, axis=-1, keepdims=True) + 1e-6)
    residual_ang = (
        (1 - (delta_smpl_norm * delta_robot_norm).sum(axis=-1))
        * (1 - jnp.eye(delta_smpl.shape[0]))
        * smpl_mask
    )
    return jnp.concatenate([residual_pos.flatten(), residual_ang.flatten()]) * weight


@jaxls.Cost.create_factory
def global_pose_cost(
    var_values,
    var_T_world_root,
    var_robot_cfg,
    var_smpl_joints_scale,
    smpl_mask,
    robot,
    keypoints,
    weight,
):
    robot_cfg, T_world_root = var_values[var_robot_cfg], var_values[var_T_world_root]
    robot_joints = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    smpl_pos, robot_pos = (
        keypoints[jnp.array(smpl_joint_retarget_indices_to_g1)],
        robot_joints.translation()[jnp.array(g1_link_retarget_indices)],
    )
    smpl_center, robot_center = smpl_pos.mean(axis=0, keepdims=True), robot_pos.mean(
        axis=0, keepdims=True
    )
    scale = (var_values[var_smpl_joints_scale] * smpl_mask).mean()
    return (
        jnp.abs((smpl_pos - smpl_center) * scale - (robot_pos - robot_center)).flatten()
        * weight
    )


@jaxls.Cost.create_factory
def end_effector_cost(
    var_values, var_T_world_root, var_robot_cfg, robot, keypoints, weight
):
    robot_cfg, T_world_root = var_values[var_robot_cfg], var_values[var_T_world_root]
    robot_joints = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    smpl_pos = keypoints[jnp.array(smpl_feet_joint_retarget_indices_to_g1)]
    robot_pos = robot_joints.translation()[jnp.array(g1_feet_joint_retarget_indices)]
    return (smpl_pos - robot_pos).flatten() * weight


@jaxls.Cost.create_factory
def foot_skating_cost(
    var_values, var_cfg_t0, var_cfg_t1, robot, contact_left, contact_right, weight
):
    fk_t0, fk_t1 = robot.forward_kinematics(
        cfg=var_values[var_cfg_t0]
    ), robot.forward_kinematics(cfg=var_values[var_cfg_t1])
    feet_pos_t0, feet_pos_t1 = (
        fk_t0[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)],
        fk_t1[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)],
    )
    ankle_pos_t0, ankle_pos_t1 = (
        fk_t0[..., 4:7][jnp.array(g1_ankle_joint_retarget_indices)],
        fk_t1[..., 4:7][jnp.array(g1_ankle_joint_retarget_indices)],
    )
    residual = (jnp.abs(feet_pos_t1[0] - feet_pos_t0[0]) * contact_left) + (
        jnp.abs(feet_pos_t1[1] - feet_pos_t0[1]) * contact_right
    )
    residual += (jnp.abs(ankle_pos_t1[0] - ankle_pos_t0[0]) * contact_left) + (
        jnp.abs(ankle_pos_t1[1] - ankle_pos_t0[1]) * contact_right
    )
    return residual.flatten() * weight


@jaxls.Cost.create_factory
def ground_contact_cost(
    var_values, var_T_world_root, var_robot_cfg, robot, target_z, contact_mask, weight
):
    T_world_links = var_values[var_T_world_root] @ jaxlie.SE3(
        robot.forward_kinematics(cfg=var_values[var_robot_cfg])
    )
    robot_z = T_world_links.translation()[jnp.array(g1_feet_joint_retarget_indices)][
        ..., 2
    ]
    return ((robot_z - target_z) * contact_mask).flatten() * weight


@jaxls.Cost.create_factory
def hip_yaw_and_pitch_cost(var_values, var_robot_cfg, yaw_w, pitch_w, roll_w):
    cfg = var_values[var_robot_cfg]
    return jnp.concatenate(
        [
            cfg[..., [2, 8]] * yaw_w,
            cfg[..., [0, 6]] * pitch_w,
            cfg[..., [1, 7]] * roll_w,
        ]
    ).flatten()


@jaxls.Cost.create_factory
def root_smoothness(var_values, var_t0, var_t1, weight):
    return (var_values[var_t0].inverse() @ var_values[var_t1]).log().flatten() * weight


@jaxls.Cost.create_factory
def scale_regularization(var_values, var_scale):
    s = var_values[var_scale]
    return jnp.concatenate(
        [
            (s - 1.0).flatten(),
            (s - s.T).flatten() * 100.0,
            jnp.clip(-s, min=0).flatten() * 100.0,
        ]
    )


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
    ts = jnp.arange(target_keypoints.shape[0])
    var_Ts, var_joints, var_scale = (
        jaxls.SE3Var(ts),
        robot.joint_var_cls(ts),
        SmplJointsScaleVarG1(0),
    )
    contact_mask = jnp.stack([left_foot_contact, right_foot_contact]).T

    @jaxls.Cost.create_factory
    def world_collision_cost(var_values, var_Ts_world_root, var_robot_cfg, coll_weight):
        coll = robot_coll.at_config(robot, var_values[var_robot_cfg]).transform(
            var_values[var_Ts_world_root]
        )
        dist = pk.collision.collide(coll, heightmap)
        return (
            pk.collision.colldist_from_sdf(
                dist, activation_dist=weights["world_coll_margin"]
            ).flatten()
            * coll_weight
        )

    costs = [
        local_pose_cost(
            var_Ts,
            var_joints,
            var_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            target_keypoints,
            weights["local_pose_cost_weight"] * valid_timesteps,
        ),
        global_pose_cost(
            var_Ts,
            var_joints,
            var_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            target_keypoints,
            weights["global_pose_cost_weight"] * valid_timesteps,
        ),
        end_effector_cost(
            var_Ts,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            target_keypoints,
            weights["end_effector_cost_weight"] * valid_timesteps,
        ),
        ground_contact_cost(
            var_Ts,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            target_ground_z,
            contact_mask,
            weights["ground_contact_cost_weight"] * valid_timesteps,
        ),
        foot_skating_cost(
            robot.joint_var_cls(ts[:-1]),
            robot.joint_var_cls(ts[1:]),
            jax.tree.map(lambda x: x[None], robot),
            left_foot_contact[1:],
            right_foot_contact[1:],
            weights["foot_skating_cost_weight"] * valid_timesteps[1:],
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.1,
            weight=weights["self_coll_factor_weight"]
            * weights["padding_norm_factor_weight"]
            * valid_timesteps,
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            weight=weights["limit_cost_factor_weight"]
            * weights["padding_norm_factor_weight"]
            * valid_timesteps,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(ts[:-1]),
            robot.joint_var_cls(ts[1:]),
            weight=weights["smoothness_cost_factor_weight"]
            * weights["padding_norm_factor_weight"]
            * valid_timesteps[1:],
        ),
        root_smoothness(
            jaxls.SE3Var(ts[:-1]),
            jaxls.SE3Var(ts[1:]),
            (2 * weights["smoothness_cost_factor_weight"])
            * (valid_timesteps[1:] * valid_timesteps[:-1]),
        ),
        hip_yaw_and_pitch_cost(
            var_joints,
            weights["hip_yaw_cost_weight"],
            weights["hip_pitch_cost_weight"],
            weights["hip_roll_cost_weight"],
        ),
        scale_regularization(var_scale),
        world_collision_cost(
            var_Ts,
            var_joints,
            weights["world_coll_factor_weight"]
            * weights["padding_norm_factor_weight"]
            * valid_timesteps,
        ),
    ]

    graph = jaxls.LeastSquaresProblem(
        costs=costs, variables=[var_Ts, var_joints, var_scale]
    ).analyze()
    solved_vals, summary = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_Ts.with_value(Ts_world_root_smpl),
                var_joints,
                var_scale.with_value(
                    jnp.ones(
                        (
                            len(smpl_joint_retarget_indices_to_g1),
                            len(smpl_joint_retarget_indices_to_g1),
                        )
                    )
                ),
            ]
        ),
        return_summary=True,
    )
    print(summary)
    return solved_vals[var_joints], solved_vals[var_Ts]


# ===================================================================================
# 3. MAIN EXECUTION (Correcting NameError)
# ===================================================================================
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
    with h5py.File(megahunter_path, "r") as f:
        world_env = load_dict_from_hdf5(f)["our_pred_world_cameras_and_structure"]
    contacts = {
        fn: pickle.load(open(contact_dir / f"{fn}.pkl", "rb"))
        for fn in world_env.keys()
    }
    left_foot_contact = onp.array(
        [contacts[fn][int(person_id)]["left_foot_contact"] for fn in world_env.keys()]
    )
    right_foot_contact = onp.array(
        [contacts[fn][int(person_id)]["right_foot_contact"] for fn in world_env.keys()]
    )

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
    padded_initial_T_world_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(
            jnp.pad(
                initial_T_world_root.rotation().as_matrix(),
                ((0, padded_num_timesteps - num_timesteps), (0, 0), (0, 0)),
                "edge",
            )
        ),
        jnp.pad(
            initial_T_world_root.translation(),
            ((0, padded_num_timesteps - num_timesteps), (0, 0)),
            "edge",
        ),
    )
    padded_left_contact = jnp.pad(
        jnp.array(left_foot_contact),
        (0, padded_num_timesteps - len(left_foot_contact)),
        "edge",
    )
    padded_right_contact = jnp.pad(
        jnp.array(right_foot_contact),
        (0, padded_num_timesteps - len(right_foot_contact)),
        "edge",
    )

    background_mesh = (
        trimesh.load(src_dir / "background_mesh.obj", force="mesh")
        if (src_dir / "background_mesh.obj").exists()
        else None
    )
    padded_target_ground_z = onp.zeros((padded_num_timesteps, 2))
    if background_mesh is not None and not background_mesh.is_empty:
        human_feet_pos = onp.array(
            padded_keypoints[
                :,
                [
                    smpl_joint_names.index("left_foot"),
                    smpl_joint_names.index("right_foot"),
                ],
                :,
            ]
        )
        ray_origins_flat = (human_feet_pos + onp.array([0, 0, 0.1])).reshape(-1, 3)
        num_rays = ray_origins_flat.shape[0]
        down_directions = onp.full_like(ray_origins_flat, [0, 0, -1.0])
        index_tri = background_mesh.ray.intersects_first(
            ray_origins=ray_origins_flat, ray_directions=down_directions
        )
        missed_indices_down = onp.where(index_tri == -1)[0]
        if len(missed_indices_down) > 0:
            origins_for_upward = ray_origins_flat[missed_indices_down]
            up_directions = onp.full_like(origins_for_upward, [0, 0, 1.0])
            index_tri_up = background_mesh.ray.intersects_first(
                ray_origins=origins_for_upward, ray_directions=up_directions
            )
            valid_up_hits = onp.where(index_tri_up != -1)[0]
            if len(valid_up_hits) > 0:
                update_indices = missed_indices_down[valid_up_hits]
                index_tri[update_indices] = index_tri_up[valid_up_hits]

        # BUG FIX: Use the correct variable name defined within this scope.
        fallback_z = onp.min(human_feet_pos[..., 2])

        target_ground_z_flat = onp.zeros(num_rays)
        valid_hit_indices = onp.where(index_tri != -1)[0]
        if len(valid_hit_indices) > 0:
            hit_triangle_indices = index_tri[valid_hit_indices]
            valid_mask = hit_triangle_indices < len(background_mesh.triangles)
            valid_hit_indices = valid_hit_indices[valid_mask]
            hit_triangle_indices = hit_triangle_indices[valid_mask]
            projected_triangles_center_z = background_mesh.triangles[
                hit_triangle_indices
            ].mean(axis=1)[:, 2]
            target_ground_z_flat[valid_hit_indices] = projected_triangles_center_z
        target_ground_z_flat[index_tri == -1] = fallback_z
        padded_target_ground_z = jnp.array(target_ground_z_flat.reshape(-1, 2))

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
    raw_robot_cfg, optimized_T_world_root = retarget_human_to_robot(
        robot,
        robot_coll,
        heightmap,
        valid_timesteps,
        padded_keypoints,
        padded_initial_T_world_root,
        smpl_mask,
        padded_left_contact,
        padded_right_contact,
        padded_target_ground_z,
        weights,
    )
    jax.block_until_ready((raw_robot_cfg, optimized_T_world_root))
    print(f"Optimization finished in {time.time() - start_time:.2f}s")

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
