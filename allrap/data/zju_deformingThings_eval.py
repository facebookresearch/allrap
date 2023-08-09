import torch
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from pytorch3d.renderer.cameras import PerspectiveCameras

root_dir = '/private/home/benjamingraham/repos/nrsfm-sbl/c3dpo_up3d79_ben'
def mpjpe(pred, gt):
    assert pred.dim() == 3, "(Time,Keypoints,3)"
    assert pred.shape == gt.shape
    assert pred.device == gt.device
    pred = pred.double()
    gt = gt.double()
    r = {}

    beta = torch.dot(pred.flatten(), gt.flatten()) / pred.pow(2).sum()
    r["1df"] = (beta * pred - gt).norm(p=2, dim=2).mean().item()

    x0 = pred.view(-1, 1)
    x1 = torch.nn.functional.normalize(pred, p=2, dim=2).view(-1, 1)
    x = torch.cat([x0, x1], 1)
    y = gt.view(-1, 1)
    beta = torch.inverse(x.mT @ x) @ x.mT @ y
    r["2dfA"] = (x @ beta - y).view(-1, 3).norm(p=2, dim=1).mean().item()

    x0 = pred.view(-1, 1)
    x1 = (pred / pred[:, :, 2:]).view(-1, 1)
    x = torch.cat([x0, x1], 1)
    y = gt.view(-1, 1)
    beta = torch.inverse(x.mT @ x) @ x.mT @ y
    r["2dfB"] = (x @ beta - y).view(-1, 3).norm(p=2, dim=1).mean()

    x = pred - pred.mean(1, keepdim=True)
    y = gt - gt.mean(1, keepdim=True)
    beta = torch.dot(x.flatten(), y.flatten()) / x.pow(2).sum()
    r["4df"] = (beta * x - y).norm(p=2, dim=-1).mean().item()
    return r


def mpjpe_warp(pred, gt):
    r = {}

    beta = torch.dot(pred.flatten(), gt.flatten()) / pred.pow(2).sum()
    r["1df"] = beta * pred

    # x0 = pred.view(-1,1)
    # x1 = torch.nn.functional.normalize(pred,p=2, dim=2).view(-1,1)
    # x = torch.cat([x0, x1], 1)
    # y = gt.view(-1, 1)
    # beta = torch.inverse(x.mT @ x) @ x.mT @ y
    # r["2dfA"] = (x @ beta).view_as(gt)

    # x0 = pred.view(-1,1)
    # x1 = (pred / pred[:, :, 2:]).view(-1,1)
    # x = torch.cat([x0, x1], 1)
    # y = gt.view(-1, 1)
    # beta = torch.inverse(x.mT @ x) @ x.mT @ y
    # r["2dfB"] = (x @ beta).view_as(gt)

    x = pred - pred.mean(1, keepdim=True)
    gtm = gt.mean(1, keepdim=True)
    y = gt - gtm
    beta = torch.dot(x.flatten(), y.flatten()) / x.pow(2).sum()
    r["4df"] = (beta * x).view_as(gt) + gtm
    return r


def load_scene(root, static_camera=True, n_kpts=100, visibility_threshold=0.3):
    if "zju_fastcapture" in root:
        assert static_camera
        scene = torch.load(            f"{root_dir}/data/{root}.pth"        )
        scene["traj_3d_world"] = scene["traj_3d_world"].permute(2, 0, 1)
        scene["traj_2d"] = 1 - scene["traj_2d"].permute(2, 0, 1) / 512
        scene["visibility"] = scene["visibility"].mT[:, :, None]
        cam = scene["camera"]
        print(cam.focal_length, cam.principal_point)

        frame = 0
        x = cam.transform_points(scene["traj_3d_world"][frame, :, :])[:, :2]
        y = scene["traj_2d"][frame, :, :]
        assert torch.isclose(x, y, atol=1e-6).all()

        xyv = torch.cat(
            [scene["traj_2d"] * scene["visibility"], scene["visibility"]], 2
        )
        xyz = scene["traj_3d_world"] @ cam.R[0] + cam.T[0]
        cam.R[0].copy_(torch.eye(3))
        cam.T[0].copy_(torch.zeros(3))
        assert torch.isclose(
            cam.transform_points(xyz)[..., :2] * scene["visibility"],
            xyv[:, :, :2],
            atol=1e-3,
        ).all()
        del scene
    else:

        def l(x):
            return torch.load(
                f"{root_dir}/data/{root}/pth_files/{x}.pth",
                map_location="cpu",
            )

        if static_camera:
            xyz = l("T_N_xyz_static")
            xyv = l("T_N_xyv_static")
        else:
            xyz = l("T_N_xyz_track")
            xyv = l("T_N_xyv_track")

        cam_param = l("cam_params")
        cam = PerspectiveCameras(
            focal_length=torch.ones(1, 2) * cam_param["focal_length"],
            principal_point=torch.tensor(cam_param["principal_point"])[None],
        )
        print(cam.focal_length, cam.principal_point)

        frame = 0
        visible = xyv[frame, :, 2].bool()
        if not static_camera:
            xyz = xyz - xyz.mean(1, keepdim=True) + cam_param["object_translation"]
        assert torch.isclose(
            cam.transform_points(xyz[frame, visible, :])[:, :2],
            xyv[frame, visible, :2],
            atol=1e-06,
        ).all()

    print("Initial shape", xyz.shape)
    vis = xyv[:, :, 2]
    mean_vis = vis.mean(0)
    xyz = xyz[:, mean_vis >= visibility_threshold, :]
    xyv = xyv[:, mean_vis >= visibility_threshold, :]
    vis = xyv[:, :, 2]
    print("remove mostly occluded points", xyz.shape, vis.shape)
    idxs = sample_farthest_points(
        xyz.transpose(0, 1).mT.reshape(1, xyz.size(1), -1),
        K=torch.Tensor([n_kpts]),
        random_start_point=False,
    )[1].flatten()
    xyz = xyz[:, idxs]
    xyv = xyv[:, idxs]
    vis = xyv[:, :, 2]

    xyv[:, :, :2] *= xyv[:, :, 2:]
    print(xyz.shape, vis.shape)
    return xyv, xyz, cam


if __name__ == "__main__":
    for root in [
        "animals/bearVGG_Turn3",
        "animals/bullMJ6_SwimTurnR",
        "animals/foxWDFS_Swim9",
        "animals/pumaRW_Damaged1",
        "animals/lionessHTR42_action1",
        "zju_fastcapture/377",
        "zju_fastcapture/386",
        "zju_fastcapture/387",
        "zju_fastcapture/392",
        "zju_fastcapture/393",
        "zju_fastcapture/394",
    ]:
        print(root)
        load_scene(root, static_camera=True)
