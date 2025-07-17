import tyro
import h5py
import os.path as osp
import numpy as onp
import mujoco
import imageio
import os
import numpy as np

from tqdm import trange
from loguru import logger

from pathlib import Path
from typing import List, Tuple


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


def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    return result


def set_free_joint_pose(
    joint_name: str,
    position: onp.ndarray,
    orientation_xyzw: onp.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Sets the pose of a free joint by its name."""
    # MuJoCo requires qpos to be [x, y, z, w, x, y, z] order for quaternion
    orientation_wxyz = onp.array(
        [
            orientation_xyzw[3],
            orientation_xyzw[0],
            orientation_xyzw[1],
            orientation_xyzw[2],
        ],
        dtype=onp.double,
    )
    # Get the joint ID and its starting address in qpos
    jid = model.joint(joint_name).id
    qpos_start = model.jnt_qposadr[jid]
    # Write position + quaternion to the qpos array
    data.qpos[qpos_start : qpos_start + 3] = position
    data.qpos[qpos_start + 3 : qpos_start + 7] = orientation_wxyz


def main(
    postprocessed_dir: Path,
    output_video_path: Path | None = None,
    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.793),
):
    """
    Script to visualize retargeted G1 robot motion using MuJoCo.
    """
    robot_name = "g1"

    if output_video_path is None:
        output_video_path = postprocessed_dir / f'{robot_name}_kinematics_only.mp4'
        logger.info(f"Output video path not provided. Using default: {output_video_path}")

    # --- 1. Load Motion Data (G1) ---
    retargeted_poses_path = postprocessed_dir / f'retarget_poses_{robot_name}.h5'
    with h5py.File(retargeted_poses_path, 'r') as f:
        retargeted_poses = load_dict_from_hdf5(f)
        joint_names = f.attrs['joint_names'].tolist()
        link_names = f.attrs['link_names'].tolist()
        fps = f.attrs['fps']

    link_pos= retargeted_poses["link_pos"]

    num_frames = retargeted_poses["root_pos"].shape[0]
    
    logger.info(f"Loaded {num_frames} frames of motion data.")

    first_frame_root_pos = retargeted_poses["root_pos"][0]
    initial_pos_np = onp.array(initial_pos)

    # --- 2. Setup MuJoCo Simulation ---
    xml_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_disable_collision.xml")
    spec = mujoco.MjSpec.from_file(xml_path)

    model = spec.compile()
    data = mujoco.MjData(model)
    mj_qpos_indices, _, _ = get_ordered_joint_indices(model, joint_names)
    model.opt.gravity[:] = 0

    # --- 3. Set Initial State for Camera Initialization ---
    final_root_pos_t0 = initial_pos_np + (retargeted_poses["root_pos"][0] - first_frame_root_pos)
    
    # Set initial robot pose
    data.qpos[0:3] = final_root_pos_t0
    data.qpos[3:7] = onp.concatenate([retargeted_poses["root_quat"][0][3:], retargeted_poses["root_quat"][0][:3]])
    for i, angle in enumerate(onp.array(retargeted_poses["joints"][0])):
        data.qpos[mj_qpos_indices[i]] = angle
    
    mujoco.mj_forward(model, data)

    # --- 4. Setup Renderer and Camera ---
    renderer = mujoco.Renderer(model, height=1920, width=1080)
    camera = mujoco.MjvCamera()
    tracking_body_name = "pelvis"
    try:
        tracking_body_id = model.body(tracking_body_name).id
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = tracking_body_id
        camera.lookat = data.body(tracking_body_id).xpos.copy()
        logger.info(f"Camera initialized to track body: '{tracking_body_name}'")
    except KeyError:
        logger.warning(f"Body '{tracking_body_name}' not found. Falling back to FREE camera.")
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat = onp.array([0.0, 0.0, 0.75])

    camera.distance = 3.0
    camera.elevation = -20.0
    camera.azimuth = 90.0
    frames = []

    feet_heights:List[np.ndarray] = []
    feet_names = ['left_ankle_roll_link','right_ankle_roll_link']

    # --- 5. Pure Kinematic Replay Loop ---
    for t in trange(num_frames, desc="Kinematic Replay"):
        # --- Update Robot Pose ---
        final_root_pos = initial_pos_np + (retargeted_poses["root_pos"][t] - first_frame_root_pos)
        data.qpos[0:3] = final_root_pos
        data.qpos[3:7] = onp.concatenate([retargeted_poses["root_quat"][t][3:], retargeted_poses["root_quat"][t][:3]])
        for i, angle in enumerate(onp.array(retargeted_poses["joints"][t])):
            data.qpos[mj_qpos_indices[i]] = angle
            
        mujoco.mj_forward(model, data)

        left_foot_height = data.body(feet_names[0]).xpos[2]
        right_foot_height = data.body(feet_names[1]).xpos[2]
        feet_heights.append(np.array([left_foot_height, right_foot_height]))

        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

    feet_heights = np.array(feet_heights)
    print(f"First frame feet heights: {feet_heights[0]}")

    min_feet_height = feet_heights.min(axis=0)
    print(f"Minimum feet heights: {min_feet_height}")

    # --- 6. Save the Video ---
    output_video_path = output_video_path.resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving video to {output_video_path}")
    imageio.mimsave(output_video_path, frames, fps=fps)
    logger.info("Video saving complete.")

    renderer.close()


if __name__ == "__main__":
    tyro.cli(main)