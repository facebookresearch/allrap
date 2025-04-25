# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras


def mpjpe(pred, gt):
    assert pred.dim() == 3, "(Time,Keypoints,3)"
    assert pred.shape == gt.shape
    assert pred.device == gt.device
    pred = pred.double()
    gt = gt.double()
    r = {}

    beta = torch.dot(pred.flatten(), gt.flatten()) / pred.pow(2).sum()
    r["scale"] = (beta * pred - gt).norm(p=2, dim=2).mean().item()

    x = pred - pred.mean(1, keepdim=True)
    y = gt - gt.mean(1, keepdim=True)
    beta = torch.dot(x.flatten(), y.flatten()) / x.pow(2).sum()
    r["center(3d)_and_scale"] = (beta * x - y).norm(p=2, dim=-1).mean().item()

    x = pred - pred.mean((0, 1), keepdim=True)
    y = gt - gt.mean((0, 1), keepdim=True)
    beta = torch.dot(x.flatten(), y.flatten()) / x.pow(2).sum()
    r["center(4d)_and_scale"] = (beta * x - y).norm(p=2, dim=-1).mean().item()

    return r


def load_scene(root, n_kpts=100):
    root_dir = os.path.dirname(os.path.realpath(__file__))

    def load(x):
        return torch.load(
            f"{root_dir}/df4d/{root}/{x}.pth",
            map_location="cpu",
        )

    xyv = load(f"T_{n_kpts}_xyv")
    xyz = load(f"T_{n_kpts}_xyz")
    cam_param = load("cam_params")
    cam = PerspectiveCameras(
        focal_length=torch.ones(1, 2) * cam_param["focal_length"],
        principal_point=torch.tensor(cam_param["principal_point"])[None],
    )
    frame = 0
    visible = xyv[frame, :, 2].bool()
    assert torch.isclose(
        cam.transform_points(xyz[frame, visible, :])[:, :2],
        xyv[frame, visible, :2],
        atol=1e-06,
    ).all()
    return {
        "xyv": xyv,
        "xy": xyv[:, :, :2],
        "visible": xyv[:, :, 2],
        "xyz": xyz,
        "camera": cam,
    }


if __name__ == "__main__":
    for root in [
        "bullMJ6_SwimTurnR",
        "foxWDFS_Swim9",
        "pumaRW_Damaged1",
        "lionessHTR42_action1",
    ]:
        print(root)
        print(load_scene(root))
