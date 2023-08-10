import os
import torch
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

net = model.mlp_mixer.MLPMixer(hidden_layers=32, dim=32)
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

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
            xy = s["kp_loc"].mT.to(device)
            visible = s["kp_vis"].to(device)

            # random 2d rotation
            R = torch.rand(len(xy)) * 2 * torch.pi
            R = torch.stack([R.cos(), R.sin(), -R.sin(), R.cos()], 1).reshape(-1, 2, 2)
            R = R.to(device)
            xy = xy @ R

        pred_pose_3d = net(xy, visible)

        loss = losses.visibility.weak_negative_correlation(
            visible, pred_pose_3d[:, :, 2]
        ) + losses.subset.nearby_tuple_aligned_coord_loss(pred_pose_3d, 10, 32, False)
        timm.utils.adaptive_clip_grad(net.parameters(), clip_factor=0.1)
        loss.backward()
        optim.step()
        optim.zero_grad()

    torch.save([epoch + 1, net.state_dict()], filename)

    # if epoch > 5 and epoch % 5 < 4:
    #    continue
    net.eval()
    gt = []
    pred = []
    visi = []
    with torch.no_grad():
        for s in dataloader["test"]:
            xy = s["kp_loc"].mT.to(device)
            visible = s["kp_vis"].to(device)
            pred_pose_3d = net(xy, visible)
            gt.append(s["kp_loc_3d"])
            visi.append(s["kp_vis"])
            pred.append(pred_pose_3d.mT.cpu())
        gt = torch.cat(gt)
        pred = torch.cat(pred)
        visi = torch.cat(visi)
        print(pred.shape)
        pred[:, 2, :] -= pred[:, 2, :].mean(1, keepdim=True)
        results, _ = data.sup3d.eval_up3d_79kp(
            {"kp_loc_3d": gt, "shape_image_coord": pred}
        )
        print(f"Epoch: {epoch} MPJPE: {results}")
