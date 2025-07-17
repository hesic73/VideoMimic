import tyro
import h5py
import os.path as osp
import numpy as onp
import mujoco
import imageio
import os
import torch
import smplx
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
    megahunter_path: Path | None = None,
    gender: str = 'male',
):
    """
    Script to visualize retargeted G1 robot motion and the original SMPL
    motion using MuJoCo.
    """
    robot_name = "g1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if output_video_path is None:
        output_video_path = postprocessed_dir / f'{robot_name}_kinematics_only.mp4'
        logger.info(f"Output video path not provided. Using default: {output_video_path}")

    # --- 1. Load All Motion Data (G1 and SMPL) ---
    if megahunter_path is None:
        megahunter_path = postprocessed_dir.parent.parent / 'output_smpl_and_points' / f'{postprocessed_dir.name}.h5'
        if not megahunter_path.exists():
            raise FileNotFoundError(f"SMPL data file not found at the default path: {megahunter_path}")

    retargeted_poses_path = postprocessed_dir / f'retarget_poses_{robot_name}.h5'
    with h5py.File(retargeted_poses_path, 'r') as f:
        retargeted_poses = load_dict_from_hdf5(f)
        joint_names = f.attrs['joint_names'].tolist()
        link_names = f.attrs['link_names'].tolist()
        fps = f.attrs['fps']

    logger.info(f"Loading SMPL data from: {megahunter_path}")
    with h5py.File(megahunter_path, 'r') as f:
        megahunter_data = load_dict_from_hdf5(f)

    link_pos= retargeted_poses["link_pos"]

    rotated_keypoints_path = postprocessed_dir / 'gravity_calibrated_keypoints.h5'
    with h5py.File(rotated_keypoints_path, 'r') as f:
        world_rotation = load_dict_from_hdf5(f)['world_rotation']

    human_params = megahunter_data['our_pred_humans_smplx_params']
    person_id = list(human_params.keys())[0]
    num_frames = human_params[person_id]['body_pose'].shape[0]
    
    smpl_layer = smplx.create(model_path='./assets/body_models', model_type='smpl', gender=gender, num_betas=10, batch_size=num_frames).to(device)
    smpl_betas = torch.from_numpy(human_params[person_id]['betas']).to(device)
    if smpl_betas.ndim == 1: smpl_betas = smpl_betas.repeat(num_frames, 1)

    smpl_output = smpl_layer(body_pose=torch.from_numpy(human_params[person_id]['body_pose']).to(device), betas=smpl_betas, global_orient=torch.from_numpy(human_params[person_id]['global_orient']).to(device), pose2rot=False)
    
    smpl_joints = smpl_output['joints'].detach().cpu().numpy()
    smpl_root_joint = smpl_joints[:, 0:1, :]
    smpl_joints3d = (smpl_joints - smpl_root_joint) + human_params[person_id]['root_transl']
    smpl_joints3d_aligned = smpl_joints3d @ world_rotation.T
    num_smpl_joints = smpl_joints3d_aligned.shape[1]
    
    logger.info(f"Loaded {num_frames} frames of motion data.")
    logger.info(f"Visualizing {num_smpl_joints} SMPL joints using bodies with free joints.")

    first_frame_root_pos = retargeted_poses["root_pos"][0]
    initial_pos_np = onp.array(initial_pos)
    identity_quat_xyzw = onp.array([0, 0, 0, 1])

    # --- 2. Setup MuJoCo Simulation with Bodies and Free Joints ---
    xml_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_disable_collision.xml")
    spec = mujoco.MjSpec.from_file(xml_path)

    # For each SMPL joint, add a body with a free joint and a sphere geom
    for i in range(num_smpl_joints):
        body = spec.worldbody.add_body(name=f'smpl_body_{i}')
        # The joint controls the body's 6 DoF pose
        body.add_joint(name=f'smpl_joint_{i}', type=mujoco.mjtJoint.mjJNT_FREE)
        # The geom is attached to the body, its pose is relative to the body
        body.add_geom(
            name=f'smpl_geom_{i}',
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            rgba=[0.8, 0.2, 0.8, 1],
            pos=[0, 0, 0], # Geom is at the body's origin
            contype=0,
            conaffinity=0,
        )

    model = spec.compile()
    debug_inspect_mjcf_model(model)
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
    
    # Set initial SMPL body poses using the free joints
    for i in range(num_smpl_joints):
        smpl_pos_t0 = smpl_joints3d_aligned[0, i] - first_frame_root_pos + initial_pos_np
        set_free_joint_pose(f'smpl_joint_{i}', smpl_pos_t0, identity_quat_xyzw, model, data)
    
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

        # --- Update SMPL Body Poses ---
        for i in range(num_smpl_joints):
            final_smpl_pos = smpl_joints3d_aligned[t, i] - first_frame_root_pos + initial_pos_np
            set_free_joint_pose(f'smpl_joint_{i}', final_smpl_pos, identity_quat_xyzw, model, data)
            
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


def debug_inspect_mjcf_model(model: mujoco.MjModel):
    """
    Prints detailed information about joints and actuators in the loaded MuJoCo model.
    """
    print("\n" + "=" * 70)
    print("MuJoCo Model Inspection Report")
    print("=" * 70)
    print(f"Model Info: njnt={model.njnt}, nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("-" * 70)

    print("\nJOINTS DETAILS:")
    print("-" * 70)
    if model.njnt == 0:
        print("No joints found in the model.")
    else:
        header = f"{'ID':<5} {'Name':<30} {'Type':<15} {'Body Name':<25} {'Qpos Adr':<10} {'DoF Adr':<10} {'Nq':<5} {'Nv':<5}"
        print(header)
        print("-" * len(header))
        for j_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if name is None:
                name = f"Unnamed_Joint_{j_id}"

            joint_type_int = model.jnt_type[j_id]
            joint_type_str = mujoco.mjtJoint(joint_type_int).name

            body_id = model.jnt_bodyid[j_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                body_name = f"Unnamed_Body_{body_id}"

            qpos_adr = model.jnt_qposadr[j_id]
            dof_adr = model.jnt_dofadr[j_id]
            nq_j = (
                model.jnt_qposnum[j_id]
                if hasattr(model, "jnt_qposnum")
                else (model.jnt_qposadr[j_id + 1] - qpos_adr if j_id < model.njnt - 1 else model.nq - qpos_adr)
            )
            nv_j = (
                model.jnt_dofnum[j_id]
                if hasattr(model, "jnt_dofnum")
                else (model.jnt_dofadr[j_id + 1] - dof_adr if j_id < model.njnt - 1 else model.nv - dof_adr)
            )
            print(
                f"{j_id:<5} {name:<30} {joint_type_str:<15} {body_name:<25} {qpos_adr:<10} {dof_adr:<10} {nq_j:<5} {nv_j:<5}"
            )

    print("\nACTUATOR DETAILS:")
    print("-" * 70)
    if model.nu == 0:
        print("No actuators found in the model.")
    else:
        header_act = f"{'Act ID':<7} {'Act Name':<30} {'Target Type':<15} {'Target ID':<10} {'Target Name':<25}"
        print(header_act)
        print("-" * len(header_act))
        for act_id in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            if act_name is None:
                act_name = f"Unnamed_Actuator_{act_id}"

            target_obj_type_int = model.actuator_trntype[act_id]
            target_obj_type_str = mujoco.mjtTrn(target_obj_type_int).name

            target_id = model.actuator_trnid[act_id, 0]
            target_name = "N/A"
            if target_obj_type_int != mujoco.mjtObj.mjOBJ_UNKNOWN:
                target_name = mujoco.mj_id2name(model, target_obj_type_int, target_id)
                if target_name is None:
                    target_name = f"Unnamed_Target_{target_id}"

            print(f"{act_id:<7} {act_name:<30} {target_obj_type_str:<15} {target_id:<10} {target_name:<25}")

    print("\n" + "=" * 70)
    print("End of MuJoCo Model Inspection Report")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    tyro.cli(main)