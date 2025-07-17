from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf


def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.
    """
    result = {}
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
    motion_h5_path: Path,
    urdf_path: Path=Path("assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf"),
) -> None:
    """
    Load G1 motion data from an H5 file and visualize it using Viser.

    Args:
        motion_h5_path: Path to the input .h5 file containing the motion data.
                        This file can contain single-person or multi-person motion.
        urdf_path: Path to the G1 robot's URDF file.
    """
    urdf = yourdfpy.URDF.load(str(urdf_path))

    with h5py.File(motion_h5_path, 'r') as f:
        motion_data = load_dict_from_hdf5(f)

    server = viser.ViserServer(port=8081)
    # Set Y-axis as the up direction to match the original script's visualization setting.
    server.scene.set_up_direction("+y")

    all_person_data = {}
    if "persons" in motion_data:
        print(f"Found {len(motion_data['persons'])} persons in the file.")
        all_person_data = motion_data["persons"]
    else:
        print("Found 1 person in the file.")
        all_person_data["person_0"] = motion_data

    robot_vis_handles = {}
    max_timesteps = 0
    for person_id, data in all_person_data.items():
        robot_frame = server.scene.add_frame(f"/robot_{person_id}", axes_length=0.2, axes_radius=0.01)
        urdf_viser = ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/robot_{person_id}")
        robot_vis_handles[person_id] = {"frame": robot_frame, "urdf": urdf_viser}
        
        num_frames = data["joints"].shape[0]
        if num_frames > max_timesteps:
            max_timesteps = num_frames
    
    # Add a ground plane for better visualization.
    server.scene.add_grid(
        "/ground",
        width=20.0,
        height=20.0,
        width_segments=40,
        height_segments=40,
        plane="xz",
    )

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=max_timesteps - 1, step=1, initial_value=0
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=120, step=1, initial_value=int(motion_data.get("fps", 30))
        )

    def update_robot_poses(t: int):
        for person_id, data in all_person_data.items():
            handles = robot_vis_handles[person_id]
            if t >= data["joints"].shape[0]:
                handles["frame"].visible = False
                continue
            
            handles["frame"].visible = True
            
            root_pos = data["root_pos"][t]
            root_quat_xyzw = data["root_quat"][t]
            root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])
            
            handles["frame"].position = root_pos
            handles["frame"].wxyz = root_quat_wxyz
            handles["urdf"].update_cfg(data["joints"][t])

    @gui_timestep.on_update
    def _(_) -> None:
        update_robot_poses(gui_timestep.value)

    # Set initial pose at t=0
    update_robot_poses(0)

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % max_timesteps
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)