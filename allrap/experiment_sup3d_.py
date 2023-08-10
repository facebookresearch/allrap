from dataset.dataset_zoo import dataset_zoo
from dataset.eval_zoo import eval_zoo

import os
import torch
import tuples
import mlp_mixer_classic
import tuples
import losses.subset
import losses.visibility
import model.mlp_mixer
import data.sup3d
import random
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataloader = data.sup3d.loaders()

seed = 0
random.seed(seed)
torch.manual_seed(seed)

net = mlp_mixer_classic.MLPMixerPoseNet(dropout=0,hidden_layers=32//2, dim=32,activation=torch.nn.ReLU)
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-4)

filename = "experiment_sup3d_state_dict.pth"
if 0:  # os.path.exists(filename):
    print(f"Resume training from {filename}")
    epoch, state = torch.load(filename)
    net.load_state_dict(state, strict=True)
else:
    epoch = 0

for epoch in range(epoch, 300):
    net.train()
    for s in dataloader["train"]:
            with torch.no_grad():
                pose_2d = (
                    torch.cat(
                        [s["kp_loc"].transpose(-2, -1), s["kp_vis"][:, :, None]], 2
                    )
                    .to(device)
                )
                # random 2d rotation
                R = torch.rand(len(pose_2d)) * 2 * torch.pi
                R = torch.stack([R.cos(), R.sin(), -R.sin(), R.cos()], 1).reshape(
                    -1, 2, 2
                )
                R = R.to(device)
                pose_2d[:, :, :2] = pose_2d[:, :, :2] @ R
            optim.zero_grad()
            pred_pose_3d = net(pose_2d)

            z0 = pose_2d[:, :, 2].reshape(-1)
            z1 = pred_pose_3d[:, :, 2].reshape(-1)
            loss = (
                torch.nn.functional.cosine_similarity(         z0 - z0.mean(), z1 - z1.mean(), dim=0   ).clamp(min=-0.05)
                + tuples.nearby_tuple_aligned_coord_loss(pred_pose_3d, 10, 32, False)
                )
            timm.utils.adaptive_clip_grad(net.parameters(), clip_factor=0.1)
            loss.backward()
            optim.step()

    torch.save([epoch + 1, net.state_dict()], filename)

    # if epoch > 5 and epoch % 5 < 4:
    #    continue
    net.eval()
    gt = []
    pred = []
    visi = []
    with torch.no_grad():
        for s in dataloader["test"]:
            pose_2d = torch.cat(
                [s["kp_loc"].transpose(-2, -1), s["kp_vis"][:, :, None]], 2
                ).to(device)
            pred_pose_3d = net(pose_2d)
            gt.append(s["kp_loc_3d"])
            visi.append(s["kp_vis"])
            pred.append(pred_pose_3d.transpose(-2, -1).cpu().detach())
        gt = torch.cat(gt)
        pred = torch.cat(pred)
        visi = torch.cat(visi)
        pred[:, 2, :] -= pred[:, 2, :].mean(1, keepdim=True)
        results, _ = data.sup3d.eval_up3d_79kp(
            {"kp_loc_3d": gt, "shape_image_coord": pred}
        )
        print(f"Epoch: {epoch} MPJPE: {results}")
