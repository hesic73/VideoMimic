# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import viser
import tyro
import torch
import smplx
import numpy as onp
import time
import h5py
import os.path as osp
import yourdfpy

from pathlib import Path
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R


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

    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path):]] = attr_value

    return result


def main(
    postprocessed_dir: Path,
    megahunter_path: Path | None = None,
    gender: str = 'male'
):
    """
    Simplified script to visualize SMPL joints and G1 robot motion.

    Args:
        postprocessed_dir: Path to the directory with postprocessed robot motion data.
        megahunter_path: Path to the megahunter HDF5 file containing SMPL data.
        gender: Gender of the SMPL model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    robot_name = "g1"  # Hardcoded robot name

    if megahunter_path is None:
        # try to retrieve the megahunter path from the postprocessed dir
        megahunter_path = postprocessed_dir / '..' / '..' / 'output_smpl_and_points' / f'{postprocessed_dir.name}.h5'
        if not megahunter_path.exists():
            megahunter_path = None
            print(f"[Warning] The megahunter path {megahunter_path} does not exist")

    # --- 1. Load Human (SMPL) Data ---
    with h5py.File(megahunter_path, 'r') as f:
        megahunter_data = load_dict_from_hdf5(f)

    human_params_in_world = megahunter_data['our_pred_humans_smplx_params']
    person_id = list(human_params_in_world.keys())[0]

    num_frames = human_params_in_world[person_id]['body_pose'].shape[0]
    smpl_batch_layer = smplx.create(
        model_path='./assets/body_models', model_type='smpl', gender=gender,
        num_betas=10, batch_size=num_frames
    ).to(device)

    smpl_betas = torch.from_numpy(
        human_params_in_world[person_id]['betas']).to(device)
    if smpl_betas.ndim == 1:
        smpl_betas = smpl_betas.repeat(num_frames, 1)

    smpl_output_batch = smpl_batch_layer(
        body_pose=torch.from_numpy(
            human_params_in_world[person_id]['body_pose']).to(device),
        betas=smpl_betas,
        global_orient=torch.from_numpy(
            human_params_in_world[person_id]['global_orient']).to(device),
        pose2rot=False
    )

    smpl_joints = smpl_output_batch['joints']
    smpl_root_joint = smpl_joints[:, 0:1, :]
    smpl_joints3d = smpl_joints.detach().cpu().numpy() - smpl_root_joint.detach().cpu().numpy() + \
        human_params_in_world[person_id]['root_transl']

    # --- 2. Load Robot (G1) Data ---
    rotated_keypoints_path = postprocessed_dir / 'gravity_calibrated_keypoints.h5'
    with h5py.File(rotated_keypoints_path, 'r') as f:
        rotated_keypoints = load_dict_from_hdf5(f)

    retargeted_poses_path = postprocessed_dir / \
        f'retarget_poses_{robot_name}.h5'
    with h5py.File(retargeted_poses_path, 'r') as f:
        retargeted_poses = load_dict_from_hdf5(f)

    # Align human motion to the robot's coordinate system
    world_rotation = rotated_keypoints['world_rotation']
    smpl_joints3d = smpl_joints3d @ world_rotation.T

    # --- 3. Setup Viser Server and UI ---
    server = viser.ViserServer(port=8081)

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15)

    # --- 4. Initialize Visualization Objects ---

    # Initialize SMPL joints point cloud
    smpl_joints_handle = server.scene.add_point_cloud(
        "/smpl_joints",
        points=smpl_joints3d[0],
        colors=onp.array([[128, 0, 128]] *
                         smpl_joints3d[0].shape[0]),  # Purple color
        point_size=0.03,
    )

    # Initialize G1 Robot URDF
    urdf_path = osp.join(osp.dirname(
        __file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
    urdf = yourdfpy.URDF.load(urdf_path)
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_viser = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot")

    # --- 5. Define Animation Update Logic ---
    @gui_timestep.on_update
    def _(_) -> None:
        t = gui_timestep.value

        with server.atomic():
            # Update SMPL joints
            smpl_joints_handle.points = smpl_joints3d[t]

            # Update Robot pose
            root_quat_xyzw = retargeted_poses["root_quat"][t]
            root_pos_xyz = retargeted_poses["root_pos"][t]

            robot_frame.wxyz = onp.concatenate(
                [root_quat_xyzw[3:], root_quat_xyzw[:3]])
            robot_frame.position = root_pos_xyz

            joint_angles = onp.array(retargeted_poses["joints"][t])
            joint_angles[8] = 0.0  # TEMP FIX
            urdf_viser.update_cfg(joint_angles)

    # --- 6. Main Animation Loop ---
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
