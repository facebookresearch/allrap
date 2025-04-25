# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset


def anime_read(filename):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], riangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, "rb")
    num_frames = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_vertices = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_faces = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=num_vertices * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=num_faces * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((num_frames - 1, num_vertices, 3))
    return num_frames, num_vertices, num_faces, vert_data, face_data, offset_data


def read_anime_list(files_path):
    nfs, nvs, nts, verts, faces, offsets = [], [], [], [], [], []
    for file in files_path:
        (
            num_frames,
            num_vertices,
            num_faces,
            vert_data,
            face_data,
            offset_data,
        ) = anime_read(file)
        face_data = face_data.transpose()[None, :]
        vert_data = np.repeat(vert_data.transpose()[None, :], (num_frames - 1), axis=0)
        offset_data = offset_data.transpose((0, 2, 1))
        nfs.append(num_frames)
        nvs.append(num_vertices)
        nts.append(num_faces)
        verts.append(vert_data)
        faces.append(face_data)
        offsets.append(offset_data)

    verts = torch.tensor(np.concatenate((verts), 0))
    offsets = torch.tensor(np.concatenate((offsets), 0))
    faces = torch.tensor(np.concatenate((faces), 0))
    return nfs, nvs, nts, verts, faces, offsets


class AnimeThings(Dataset):
    def __init__(
        self,
        data_name: str,
        transform=None,
        base_dir=None,
        data_path=None,
        deform_type="animals",
        cam_rot_euler=[0.0, 0.0, 0.0],
        cam_tran=[0.0, 0.0, 0.0],
        blur_rad=0.0,
        fperpix=1,
        im_size=512,
        focal_length=1.0,
        principal_point=(0.0, 0.0),
        obj_tran=torch.tensor([0.0, 0.0, 5.0]),
        flip_yz=True,
        traj="static",
    ):
        random_seed = 0
        np.random.seed(random_seed)
        self.base_dir = base_dir
        self.data_path = data_path
        self.data_name = data_name
        self.deform_type = deform_type
        self.flip_yz = flip_yz
        self.obj_tran = obj_tran
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.load_data(data_name, deform_type)
        self.transform = transform

        cameras, rasterizer, raster_settings = init_cam_pt3d(
            im_size=im_size,
            cam_tran=cam_tran,
            blur_rad=blur_rad,
            fperpix=fperpix,
            euler=cam_rot_euler,
            focal_length=focal_length,
            principal_point=principal_point,
        )

        self.raster_settings = raster_settings
        self.rasterizer = rasterizer
        self.cameras = cameras
        self.im_size = im_size
        self.cam_tran = cam_tran
        self.cam_rot_euler = cam_rot_euler
        self.blur_rad = blur_rad
        self.fperpix = fperpix
        self.traj = traj
        self.create_static_data_trajectory()
        self.min_axes, self.max_axes = self.calculate_min_max_axes_lims()

        self.indices = torch.as_tensor(list(range(self.num_vertices)))
        self.num_points = self.num_vertices

    def create_static_data_trajectory(self):
        if self.traj == "static":
            T_xyz_N = self.create_static_cam_trajectory(self.obj_tran)
            (
                self.all_vis,
                self.all_zbuf,
                self.visible_points_10_percent,
            ) = find_permanently_occluded_points(
                self.static_cam_trajectory,
                self.face_data[0],
                self.rasterizer,
                self.raster_settings,
                self.focal_length,
            )

        self.T_xyz_N, self.T_N_xyv, self.T_N_xy, self.cam_coords = create_2d_tracks(
            T_xyz_N=T_xyz_N, all_vis=self.all_vis, cameras=self.cameras
        )
        pass

    def calculate_min_max_axes_lims(self):
        TN_xy = self.T_N_xy.flatten(0, 1)
        M = torch.max(TN_xy, dim=0)[0] + 0.5
        m = torch.min(TN_xy, dim=0)[0] - 0.5
        q = (M - m).max() / 2
        m, M = (m + M) / 2 - q, (m + M) / 2 + q
        return m, M

    def __getitem__(self, item):
        sample = {
            "pointcloud": self.vert_data[item][:, :],
            "idx": np.array(item),
            "faces": self.face_data[0],
            "verts": self.vert_data[item],
            "vert_deformation_data": self.vert_deformation_data[item],
            "pose_3d": self.T_xyz_N[item, :, :],
            "sampled_verts": self.indices[: self.num_points],
            "kp_loc": self.T_N_xy[item][:],
            "N_xyv": self.T_N_xyv[item],
            "cam_coords": self.cam_coords[item],
            "kp_vis": self.all_vis[item][:],
            "im_size": self.im_size,
            "obj_tran": self.obj_tran,
            "cam_rot_euler": self.cam_rot_euler,
            "blur_rad": self.blur_rad,
            "fperpix": self.fperpix,
            "zbuf": self.all_zbuf[item],
            "cam_tran": self.cam_tran,
            "focal_length": self.focal_length,
            "principal_point": self.principal_point,
            "min_axes": self.min_axes,
            "max_axes": self.max_axes,
            "traj": self.traj,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def create_static_cam_trajectory(self, obj_tran):
        vert_deformation_data = torch.transpose(self.vert_deformation_data, 2, 1)
        center_xy_trans_min_zby_obj_tran = obj_tran - torch.cat(
            (
                vert_deformation_data.flatten(0, 1).mean(0)[:2],
                vert_deformation_data.flatten(0, 1).min(0)[0][2:],
            )
        )
        vert_deformation_data = vert_deformation_data + center_xy_trans_min_zby_obj_tran
        self.static_cam_trajectory = torch.transpose(vert_deformation_data, 2, 1)
        return self.static_cam_trajectory

    def __len__(self):
        return self.vert_data.shape[0]

    def load_data(self, data_name, deform_type):
        files_path = []
        files_name = self.data_path[deform_type][data_name]
        for fn in files_name:
            files_path.append(f"{self.base_dir}/{fn}/{fn}.anime")
        (
            self.num_frames,
            self.num_vertices,
            self.num_faces,
            self.vert_data,
            self.face_data,
            self.offset_data,
        ) = read_anime_list(files_path)
        self.vert_data = self.vert_data / self.vert_data.max()
        self.num_frames = np.array(self.num_frames).sum()
        self.num_vertices = self.num_vertices[0]
        self.num_faces = self.num_faces[0]
        self.vert_deformation_data = self.vert_data + self.offset_data
        self.vert_deformation_data = torch.cat(
            (self.vert_data[0][None], self.vert_deformation_data)
        )
        if self.flip_yz:
            # T, V, 3
            vert_deformation_data = torch.transpose(self.vert_deformation_data, 2, 1)
            vert_deformation_data = vert_deformation_data[:, :, [1, 2, 0]]
            self.vert_deformation_data = torch.transpose(vert_deformation_data, 2, 1)


def create_2d_tracks(T_xyz_N, all_vis, cameras):
    T_N_xy, cam_coords = transform_pointcloud_cam(T_xyz_N, cameras)
    vis = torch.stack(all_vis).to(device="cuda")
    T_N_xyv = torch.cat((T_N_xy, vis), dim=2)
    return T_xyz_N, T_N_xyv, T_N_xy, cam_coords


def find_permanently_occluded_points(vert_deformation, faces, rasterizer):
    """
    Return indices of points in the frame that are visible at least
    in 10% of the frames.
    """
    occlusion_bool = torch.zeros(vert_deformation.shape[2], 1)
    len_seq = len(vert_deformation)
    occlusion_threshold = 0.1 * len_seq
    all_vis = []
    all_z_buf = []
    for n in range(len_seq):
        mesh = convert_to_mesh_pt3d(vert_deformation[n], faces)
        fragments = rasterizer(mesh)
        vis = torch.as_tensor(
            generate_visibility_information(meshes=mesh, rastered=fragments)
        )[:, None]
        all_vis.append(vis)
        all_z_buf.append(fragments.zbuf)
        occlusion_bool += vis
    return (
        all_vis,
        all_z_buf,
        np.where(occlusion_bool.reshape(-1) >= occlusion_threshold)[0],
    )


def convert_to_mesh_pt3d(verts, faces):
    verts = torch.transpose(verts, 1, 0).to(device="cuda")
    faces = torch.transpose(faces, 1, 0).to(device="cuda")
    verts_rgb = 0.5 * torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb)
    trg_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return trg_mesh


def init_cam_pt3d(
    euler=[0.0, 0.0, 0.0],
    cam_tran=[0.0, 0.0, 0.0],
    im_size=512,
    blur_rad=0.0,
    fperpix=1,
    focal_length=1.0,
    principal_point=(0.0, 0.0),
):
    R = pytorch3d.transforms.axis_angle_to_matrix(torch.as_tensor(euler))

    principal_point = torch.tensor([principal_point])
    T = torch.tensor([cam_tran]).to(device="cuda")
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R[None, :, :],
        T=T,
        device="cuda",
    )
    raster_settings = RasterizationSettings(
        image_size=im_size,
        blur_radius=blur_rad,
        faces_per_pixel=fperpix,
        bin_size=0,
        max_faces_per_bin=100,
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return cameras, rasterizer, raster_settings


def transform_pointcloud_cam(pointclouds, cameras):
    pointclouds = pointclouds.to(device="cuda")
    T_N_xy = []
    T_cam_coords = []
    for pointcloud in pointclouds:
        cam_coords = cameras.get_world_to_view_transform().transform_points(
            pointcloud.T
        )
        t_points_projected = cameras.transform_points(pointcloud.T)
        T_N_xy.append(t_points_projected[:, :2])
        T_cam_coords.append(cam_coords)

    return torch.stack(T_N_xy), torch.stack(T_cam_coords)


def generate_visibility_information(meshes, rastered):
    with torch.no_grad():
        vertex_visibility_map = torch.zeros(meshes.verts_packed().shape[0])  # (V,)
        unique_visible_verts_idx = torch.unique(
            meshes.faces_packed()[rastered.pix_to_face.unique()]
        )  # (num_visible_verts, )
        vertex_visibility_map[unique_visible_verts_idx] = 1.0
        del unique_visible_verts_idx

    return vertex_visibility_map.tolist()


def create_xy_save_pth_files(train_set, outfolder):
    # Time, 3, N
    T_N_xyz_file = os.path.join(outfolder, "T_N_xyz_{}.pth".format(train_set.traj))
    T_N_xyv_file = os.path.join(outfolder, "T_N_xyv_{}.pth".format(train_set.traj))
    cam_params_file = os.path.join(outfolder, "cam_params.pth")
    T_N_xyz = torch.transpose(train_set.T_xyz_N, 2, 1)
    torch.save(T_N_xyz.cpu(), T_N_xyz_file)
    torch.save(train_set.T_N_xyv.cpu(), T_N_xyv_file)
    torch.save(
        {
            "focal_length": train_set.focal_length,
            "principal_point": train_set.principal_point,
            "cam_translation": train_set.cam_tran,
            "object_translation": train_set.obj_tran.cpu(),
            "rasterizer_resolution": train_set.im_size,
            "min_axes": train_set.min_axes.cpu(),
            "max_axes": train_set.max_axes.cpu(),
        },
        cam_params_file,
    )
    return T_N_xyz_file, T_N_xyv_file, cam_params_file


def main(cfg):
    data_path = {
        "animals": {
            "bullMJ6_SwimTurnR": ["bullMJ6_SwimTurnR"],
            "foxWDFS_Swim9": ["foxWDFS_Swim9"],
            "pumaRW_Damaged1": ["pumaRW_Damaged1"],
            "lionessHTR42_action1": ["lionessHTR42_action1"],
        }
    }

    deform_types = ["animals"]
    out_base_dir = "/path/to/output/deformingThings4D/"
    trajectories = ["static"]
    for deform_type in deform_types:
        BASE_DIR = "/path/to/output/deformingThings4D//DeformingThings4D/{}/".format(
            deform_type
        )  # # Set me # #
        base_dir = f"{BASE_DIR}"
        for seq_type in data_path[deform_type].keys():
            for trajectory in trajectories:
                train_set = AnimeThings(
                    data_name=seq_type,
                    base_dir=base_dir,
                    data_path=data_path,
                    deform_type=deform_type,
                    cam_rot_euler=[0.0, 0.0, 0.0],
                    cam_tran=[0, 0.0, 0.0],
                    blur_rad=0.0,
                    fperpix=1,
                    im_size=2**12,
                    focal_length=1.0,
                    principal_point=(0.0, 0.0),
                    obj_tran=torch.tensor([0.0, 0.0, 3.0]),
                    flip_yz=True,
                    traj=trajectory,
                )

                files_dir = os.path.join(out_base_dir, deform_type, seq_type)

                pth_write_folder = os.path.join(files_dir, "pth_files")
                input_folder_viz = os.path.join(files_dir, "gt_images", trajectory)

                for dire in [pth_write_folder, input_folder_viz]:
                    if os.path.exists(dire):
                        pass
                    else:
                        os.makedirs(dire, exist_ok=True)

                (
                    TNxyz_pth_file,
                    TNxyv_pth_file,
                    cam_params_file,
                ) = create_xy_save_pth_files(train_set, pth_write_folder)
                print(
                    f"Generated {TNxyz_pth_file}, {TNxyv_pth_file}"
                    f" and {cam_params_file}"
                )


if __name__ == "__main__":
    main()
