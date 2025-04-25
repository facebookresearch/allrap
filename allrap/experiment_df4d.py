# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import sys

import model.mlp_mixer
import timm
import torch
from data.deforming_things_4d import load_scene, mpjpe
from loss.subset import nearby_tuple_aligned_coord_loss
from loss.visibility import weak_negative_correlation

import visdom, os
vis=visdom.Visdom(
            env='df4d',
            port=int(os.environ["VISDOM_PORT"]),
            server=os.environ["VISDOM_SERVER"],
)
# vis=None
@torch.no_grad()
def visdom_scatter(vis, xyz, rgb=None, win="3d", markersize=3, title=None, webgl=False):
    if vis is None:
        return
    if title is None:
        title = win
    xyz = xyz.cpu()
    m = xyz.min(0)[0]
    M = xyz.max(0)[0]
    rng = (M - m).max()
    xyz = torch.cat([xyz, torch.zeros(2, 3)])
    xyz[-2] = (M + m - rng) * 0.5
    xyz[-1] = (M + m + rng) * 0.5
    opts = {
        "markersize": markersize,
        "title": title,
        "caption": title,
        "markerborderwidth": 0,
        "webgl": webgl,
    }
    if rgb is not None:
        rgb = rgb.cpu()
        rgb = torch.cat([rgb, torch.zeros(2, 3)])
        rgb = rgb.mul(255.99).floor().numpy()
        opts["markercolor"] = rgb
    vis.scatter(
        xyz.numpy(),
        opts=opts,
        win=win,
    )



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)
n_kpts = 100

root = {1: "bullMJ6_SwimTurnR",
        2: "foxWDFS_Swim9",
        3: "pumaRW_Damaged1",
        4: "lionessHTR42_action1",
}[int(sys.argv[1])]

data = load_scene(root, n_kpts)
for k in data:
    data[k] = data[k].to(device)
xy = data["xy"]
visible = data["visible"]
v = visible[:, :, None]

net = model.mlp_mixer.MLPMixer(
    hidden_layers=32,
    dim=32,
    n_kpts=n_kpts,
    camera=data["camera"],
    activation=torch.nn.Hardswish,
    dropout = 0.01,
    training_noise=0,
)
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-5)


ll = []
for i in range(50_000):
    with torch.no_grad():
        R = torch.rand(len(xy), device=device) * 2 * torch.pi
        Rz = R * 0
        Ro = Rz + 1
        R = torch.stack(
            [
                R.cos(),
                R.sin(),
                Rz,
                -R.sin(),
                R.cos(),
                Rz,
                Rz,
                Rz,
                Ro,
            ],
            1,
        ).reshape(-1, 3, 3)
    pred_xyz = net(xy @ R[:, :2, :2], visible) @ R.mT

    loss = (
        nearby_tuple_aligned_coord_loss(pred_xyz, 10, 32, scaling="perspective")
        +
        weak_negative_correlation(visible, pred_xyz[:, :, 2])
    )
    ll.append(loss.item())
    loss.backward()
    timm.utils.adaptive_clip_grad(net.parameters(), clip_factor=0.1)
    optim.step()
    optim.zero_grad()

    if (i + 1) % 1000 == 0:
        print(
            root, i,
            "loss",
            torch.tensor(ll).mean().item(),
        )
        ll = []
        net.eval()
        with torch.no_grad():
            pred_xyz = net(xy, visible)
            m = mpjpe(pred_xyz, data["xyz"])
            print(i, m)
            torch.save(
                {
                    "iteration": i,
                    "root": root,
                    "n_kpts": n_kpts,
                    "cam": data["camera"],
                    "pose_2d": data["xyv"],
                    "pose_3d": data["xyz"],
                    "pred_xyz": pred_xyz,
                    "mpjpe": m,
                },
                f"{root}_{n_kpts}_{i}.pth",
            )
        visdom_scatter(vis,pred_xyz.view(-1,3),torch.rand(1,100,3).expand_as(pred_xyz).reshape(-1,3))
        vis.text(str(i),win='a')
        net.train()
