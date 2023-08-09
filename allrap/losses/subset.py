import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import inspect
import os
import random

import faiss
import faiss.contrib.torch_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pixar_replay.experimental.models.syncmatch.rotation_sync import tb3_to_se3

res = faiss.StandardGpuResources()  # use a single GPU


def faiss_knn(x, y, k):
    """
    Wrapper for FAISS functions to find for x
    the k nearest neighbors in y.
    """
    if x.is_cuda:
        return faiss.knn_gpu(res, x, y, k)
    else:
        d, i = faiss.knn(x.numpy(), y.numpy(), k)
        d = torch.from_numpy(d)
        i = torch.from_numpy(i)
        return d, i


def align_centered_point_clouds(A, B):
    Atb = A.mT @ B / A.size(-2)
    U, _, V_t = torch.linalg.svd(Atb)
    det = torch.linalg.det(U @ V_t)
    if (det < 0).any():
        det = det[..., None, None]
        U = torch.cat([U[..., :2, :], det * U[..., 2:, :]], -2)
    R = U @ V_t
    return R


def align_BK3(BK3):
    """
    BK3 @R ~= K3
    """
    B, K, _ = BK3.shape
    BK3 = BK3 - BK3.mean(1, keepdim=True)
    B3K = BK3.mT
    U, S, V_t = torch.linalg.svd(B3K.reshape(B * 3, K), full_matrices=False)
    V = V_t.mT
    # Sort-of rotation matrices
    B33 = U[:, :3].view(B, 3, 3)
    # Sort-of prototyical shape
    K3 = V[:, :3] * S[None, :3]
    # Mirror-image the prototypical shape when the sort-of rotations
    # are more like mirror-images of rotation matrices
    flip = torch.det(B33).sum(-1).sign()
    K3 = K3 * flip[None, None]
    bK3 = K3[None, :, :].expand(B, -1, -1)
    R = align_centered_point_clouds(BK3, bK3)
    return R, K3


def tuple_aligned_coord_loss(bnk3, cpa_with_grad=False, z_scale=False):
    B, N, K, _ = bnk3.shape
    nb3k = bnk3.permute(1, 0, 3, 2).double()  # (N,B,3,K)
    if z_scale:
        s2 = nb3k[:, :, 2, :].flatten(1).mean(1, keepdim=True)
    # center the tuples before aligning them by rotation
    nb3k = nb3k - nb3k.mean(3, keepdim=True)
    nbk3 = nb3k.mT

    with torch.no_grad():
        # SVD: use first 3 components to get a mean shape;
        # s.U: (N,B3,3+...)  s.S: (N,3+...)   s.V: (N,3*K,3+...)
        U, S, V_t = torch.linalg.svd(nb3k.reshape(N, B * 3, K), full_matrices=False)
        V = V_t.mT
        # Sort-of rotation matrices
        NB33 = U[:, :, :3].view(N, B, 3, 3)
        # Sort-of prototyical shape
        NK3 = V[:, :, :3] * S[..., None, :3]  # NK3
        # Mirror-image the prototypical shape when the sort-of rotations
        # are more like mirror-images of rotation matrices
        flip = torch.det(NB33).sum(-1).sign()
        NK3 = NK3 * flip[..., None, None]
        N_K3 = NK3[:, None, :, :].expand(-1, B, -1, -1)
        if not cpa_with_grad:
            R = align_centered_point_clouds(nbk3, N_K3)
    if cpa_with_grad:
        R = align_centered_point_clouds(nbk3, N_K3)

    # aligned = nbk3 @ R
    # s = torch.linalg.svdvals(aligned.view(N, B, K * 3))
    # f=(s[:,1:]>1e-12).double()

    residual = nbk3 @ R - N_K3
    s = torch.linalg.svdvals(residual.view(N, B, K * 3))[:, : min(B, K * 2)]
    if not z_scale:
        s2 = NK3.flatten(1).std(1, keepdim=True)
    s = (s / s2).log().mul(s > 1e-12).mean().float()
    return s


def random_k_tuples(n, k, n_skeleton):
    tuples = []
    while n > 0:
        tuples.extend(
            torch.randperm(n_skeleton)[: min(n, n_skeleton // k) * k].split(k)
        )
        n -= n_skeleton // k
    return torch.stack(tuples)


def nearby_tuple_aligned_coord_loss(poses, n, k, cpa_with_grad=False, z_scale=False):
    B, K, THREE = poses.shape
    with torch.no_grad():
        x = poses.transpose(0, 1).flatten(1, 2)
        tuples = faiss_knn(x[torch.randperm(K)[:n].to(poses.device)], x, k)[1]
    bnk3 = poses[:, tuples]
    return tuple_aligned_coord_loss(bnk3, cpa_with_grad=cpa_with_grad, z_scale=z_scale)


def random_tuple_aligned_coord_loss(poses, n, k, cpa_with_grad=False, z_scale=False):
    B, K, THREE = poses.shape
    c = random_k_tuples(n, k, K)
    bnk3 = poses[:, c]
    return tuple_aligned_coord_loss(bnk3, cpa_with_grad=cpa_with_grad, z_scale=z_scale)


def diluted_nearby_tuple_aligned_coord_loss(
    poses, split, k, cpa_with_grad=False, z_scale=False
):
    if split < 2:
        return 0
    B, K, THREE = poses.shape
    s = split
    with torch.no_grad():
        x = poses.transpose(0, 1).flatten(1, 2)
        p = torch.randperm(K, device=poses.device)[: K // s * s]
        xp = x[p].unflatten(0, (s, -1))
        p = p.unflatten(0, (s, -1))
        pd = torch.cdist(xp[:, :1], xp).squeeze(1)
        tuples = torch.topk(pd, largest=False, k=k)[1]
    bnk3 = poses[:, tuples]
    return tuple_aligned_coord_loss(bnk3, cpa_with_grad=cpa_with_grad, z_scale=z_scale)


def pairwise_distances(a, b=None):
    """
    Equivalent to torch.cdist(a,b).pow(2)
    """
    a2 = (a ** 2).sum(dim=-1, keepdim=True)
    if b is None:
        b = a
        b2 = a2.mT
    else:
        b2 = (b ** 2).sum(dim=-1, keepdim=True).mT
    return a2 - 2 * a @ b.mT + b2


def outlier_loss(pred_pose_3d, threshold=1.0):
    if threshold == -1:
        return 0
    x = pred_pose_3d  # BN3
    x = x - x.mean(dim=1, keepdim=True)
    with torch.no_grad():
        xs = x.flatten(1).std(1)[:, None, None]
    x = x / xs

    x = (
        pairwise_distances(x)
        .topk(2, dim=2, largest=False)[0][:, :, 1:2]
        .sub(threshold)
        .relu()
    )
    return x.mean()

    # x=torch.cdist(x,x).topk(2,dim=2,largest=False)[0][:,:,1:2]
    # x=x.reshape(-1)
    # x=x[x>0.5]
    # if len(x) and random.randint(0,1000)==1:
    #     print('outlier', len(x),x.min(),x.mean(),x.max())
    #     return x.sum()/(pred_pose_3d.size(0)*pred_pose_3d.size(1))
    # else:
    #     return 0


class PoseNet(torch.nn.Module):
    def __init__(
        self, n_skeleton, activ="lr25", n=1024, hidden_layers=3, bn=False, all_vis=False
    ):
        self.n_skeleton = n_skeleton
        self.all_vis = all_vis
        torch.nn.Module.__init__(self)
        if activ == "hardswish":
            act = torch.nn.Hardswish
        elif activ == "swish":
            act = torch.nn.SiLU
        elif activ == "gelu":
            act = torch.nn.GELU
        elif activ == "relu":
            act = torch.nn.ReLU
        elif activ == "mish":
            act = Mish
        elif activ == "lr25":
            act = lambda: torch.nn.LeakyReLU(negative_slope=1 / 4)
        elif activ == "selu":
            act = torch.nn.SELU
        elif activ == "tanh":
            act = torch.nn.Tanh
        else:
            assert False
        layers = []
        layers.append(
            torch.nn.Linear(n_skeleton * (2 if all_vis else 3), n, bias=not bn)
        )
        if bn:
            layers.append(torch.nn.BatchNorm1d(n))
        layers.append(act())
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(n, n, bias=not bn))
            if bn:
                layers.append(torch.nn.BatchNorm1d(n))
            layers.append(act())
        layers.append(torch.nn.Linear(n, n_skeleton * (1 if all_vis else 3)))
        self.net = torch.nn.Sequential(*layers)
        print(self.net)

    def forward(self, x):
        if self.all_vis:
            poses_ = torch.cat(
                [
                    x,
                    self.net(x.reshape(x.size(0), 2 * self.n_skeleton)).reshape(
                        x.size(0), self.n_skeleton, 1
                    ),
                ],
                2,
            )
            return poses_
        else:
            with torch.no_grad():
                vcm = (x * x[:, :, 2:]).sum(1, keepdim=True) / x[:, :, 2:].sum(
                    1, keepdim=True
                )
                if self.training:
                    vcm += torch.randn_like(vcm) * 0.05
                vcm[:, :, 2].zero_()
            x = x - vcm
            poses = self.net(x.reshape(x.size(0), -1)).reshape(-1, self.n_skeleton, 3)
            visible = x[:, :, 2:]
            poses = (
                torch.cat(
                    [
                        x[:, :, :2] * visible + poses[:, :, :2] * (1 - visible),
                        poses[:, :, 2:],
                    ],
                    2,
                )
                + vcm
            )
            return poses


class Residual(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return x + self.model(x)


def MLP(n, expand=2):
    m = torch.nn.Sequential(
        torch.nn.Linear(n, expand * n),
        torch.nn.BatchNorm1d(expand * n),
        torch.nn.ReLU(),
        torch.nn.Linear(expand * n, n),
    )
    with torch.no_grad():
        m[-1].weight.zero_()
        m[-1].bias.zero_()
    return Residual(m)


class RPoseNet(torch.nn.Module):
    def __init__(self, n_skeleton=79, n=256, expand=2, hidden_layers=12):
        super().__init__()
        self.n_skeleton = n_skeleton
        layers = []
        layers.append(torch.nn.Linear(n_skeleton * 3, n))
        layers.append(torch.nn.BatchNorm1d(n))
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(MLP(n, expand))
        layers.append(torch.nn.Linear(n, n_skeleton * 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        poses = self.net(x.reshape(x.size(0), -1)).reshape(-1, self.n_skeleton, 3)
        visible = x[:, :, 2:]
        poses_ = torch.cat(
            [x[:, :, :2] * visible + poses[:, :, :2] * (1 - visible), poses[:, :, 2:]],
            2,
        )
        return poses_


@torch.no_grad()
def scatter(xyz, rgb, win="3d", markersize=3, title=None):
    if title is None:
        title = win
    xyz, rgb = xyz.cpu(), rgb.cpu()
    m = xyz.min(0)[0]
    M = xyz.max(0)[0]
    rng = (M - m).max()
    xyz = torch.cat([xyz, torch.zeros(2, 3)])
    rgb = torch.cat([rgb, torch.zeros(2, 3)])
    xyz[-2] = (M + m - rng) * 0.5
    xyz[-1] = (M + m + rng) * 0.5
    rgb = rgb.mul(255.99).floor().numpy()
    vis.scatter(
        xyz.numpy(),
        opts={
            "markersize": markersize,
            "markercolor": rgb,
            "title": title,
            "caption": title,
            "markerborderwidth": 0,
        },
        win=win,
    )


import visdom

if os.environ["HOME"] == "/private/home/benjamingraham":
    vis = visdom.Visdom(env="q3", port=8097, server="100.97.72.139")  # devfair0301
else:
    vis = visdom.Visdom(port=8097)


@torch.no_grad()
def visdom_scatter(vis, xyz, rgb=None, win="3d", markersize=3, title=None):
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


def pad_to_multiple(x, dims, n=16):
    z = torch.zeros(*[1 for _ in x.shape], device=x.device, dtype=x.dtype)
    for d in dims:
        s = x.shape
        if s[d] % n != 0:
            p = n - (s[d] % n)
            p0 = p // 2
            p1 = p - p0
            x = torch.cat(
                [
                    z.expand(*s[:d], p0, *s[d + 1 :]),
                    x,
                    z.expand(*s[:d], p1, *s[d + 1 :]),
                ],
                dim=d,
            )
    return x


class VideoWriter:
    def __init__(self, name="foo", fps=30, trim=False):
        if name[-4:] not in [".mp4", ".gif"]:
            name = name + ".mp4"
        self.name = name
        self.fps = fps
        self.trim = trim

    def __enter__(self):
        self.frames = []
        return self

    def add_frame(self, frame):
        frame = frame.cpu()
        frame = pad_to_multiple(frame, [1, 2], 16)
        frame = frame.permute(1, 2, 0).mul(255.9).byte()
        self.frames.append(frame)

    def add_figure(self, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(torch.tensor(data))
        plt.close()

    def __exit__(self, exc_type, *args, **kwargs):
        if exc_type is None:
            self.write()

    def write(self):
        frames = torch.stack(self.frames)
        if self.trim:
            frames = trim_boring_rows(frames, (2, 3), 16)
        if self.name[-4:] == ".mp4":
            imageio.mimwrite(self.name, frames, fps=self.fps, quality=6)
            # torchvision.io.write_video('_'+self.name,frames.permute(0,3,1,2).float().div(255),fps=self.fps)
        else:
            imageio.mimwrite(self.name, frames, fps=self.fps)


class function_to_dataset(torch.utils.data.Dataset):
    def __init__(self, fn, n=1000):
        super().__init__()
        self.fn = fn
        self.n = n
        self.k = len(inspect.signature(fn).parameters)

    def __getitem__(self, k):
        if self.k:
            return self.fn(k)
        else:
            return self.fn()

    def __len__(self):
        return self.n


def function_to_dataloader(
    fn, num_batches, batch_size=None, collate_fn=None, num_workers=10
):
    return torch.utils.data.DataLoader(
        function_to_dataset(
            fn, num_batches * (1 if batch_size is None else batch_size)
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


@torch.no_grad()
def render(root, pose_2d, pose_3d, pred_pose_3d, cam):
    print("rendering")

    rgb = pose_3d
    rgb = rgb - rgb.mean(1, keepdim=True)
    rgb = rgb.transpose(0, 1).flatten(1)  # (N,B3)
    rgb = torch.linalg.svd(rgb, full_matrices=False).U[:, :3].cpu()
    rgb -= rgb.min(0)[0]
    rgb /= rgb.max(0)[0]

    xs = []
    Ms = []
    for x in [pose_3d, pred_pose_3d]:
        x = x - x.flatten(0, 1).mean()
        y = x - x.mean(1, keepdim=True)
        R, _ = align_BK3(y)
        x = [x, x.roll(1, dims=2), x.roll(2, dims=2), y @ R @ R[0].mT]
        # 2 3
        # 0 1
        x[1][:, :, 0] += x[0][:, :, 0].max() - x[1][:, :, 0].min()
        x[2][:, :, 1] += x[0][:, :, 1].max() - x[2][:, :, 1].min()
        x[3][:, :, 0] += x[2][:, :, 0].max() - x[3][:, :, 0].min()
        x[3][:, :, 1] += x[1][:, :, 1].max() - x[3][:, :, 1].min()
        x = torch.cat(x, 1)
        x -= x.flatten(0, 1).mean()
        M = x[:, :, :2].abs().max().item()
        xs.append(x.cpu())
        Ms.append(M)

    with VideoWriter(root.replace("/", "_"), fps=10) as vw:
        for i in range(len(pose_2d)):
            fig = plt.figure(figsize=(20.48, 10.24))
            rgb[:, 2].copy_(pose_2d[i, :, 2])
            rgb4 = torch.cat([rgb for _ in range(4)]).numpy()
            for j, (x, M) in enumerate(zip(xs, Ms)):
                ax = fig.add_subplot(1, 2, j + 1)
                ax.scatter(x[i, :, 0], x[i, :, 1], color=rgb4)
                ax.set_xlabel("Ground truth" if j == 0 else "Reconstruction")
                ax.set_xlim(-M, M)
                ax.set_ylim(-M, M)
                ax.text(-M / 2, -M * 0.9, "Front")
                ax.text(-M / 2, M * 0.9, "Side")
                ax.text(M / 2, -M * 0.9, "Top")
                ax.text(M / 2, M * 0.9, "Aligned to first frame")
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            vw.add_figure(fig)
    print("done")


import math

import numpy as np
import pyrender
import trimesh


def render_pcl(xyz, rgb, radius=0.1, view="F", res=512):
    if isinstance(radius, (int, float)):
        radius = [radius for _ in xyz]
    scene = pyrender.Scene()
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.5,
        # emissiveFactor=(1,1,1),
        alphaMode="OPAQUE",
        baseColorFactor=(0.9, 0.9, 0.9, 1.0),
    )

    for xyz, rgb, radius in zip(xyz, rgb, radius):
        sm = trimesh.creation.icosphere(radius=radius)
        sm.visual.vertex_colors = rgb
        tfs = torch.eye(4)[None]
        tfs[:, :3, 3] = xyz
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(m)

    sm = trimesh.creation.box(
        extents=(10, 0, 10),
    )
    sm.visual.vertex_colors = torch.tensor([1, 1, 1])
    scene.add(pyrender.Mesh.from_trimesh(sm, material=material), pose=torch.eye(4))
    if view == "F":
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
        s, c = math.sin(math.pi / 12), math.cos(math.pi / 12)
        camera_pose = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, c, s, s * 17 + 4],
                [0, -s, c, c * 17],
                [0, 0, 0, 1],
            ]
        )
    elif view == "S":
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
        s, c = math.sin(math.pi / 12), math.cos(math.pi / 12)
        camera_pose = torch.tensor(
            [
                [0, s, -c, -c * 17],
                [0, c, s, s * 17 + 4],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
    elif view == "T":
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0, aspectRatio=1.0)
        s, c = math.sin(math.pi / 2), math.cos(math.pi / 2)
        camera_pose = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, c, s, s * 17 + 4],
                [0, -s, c, c * 17],
                [0, 0, 0, 1],
            ]
        )
    elif view == "FO":
        camera = pyrender.OrthographicCamera(xmag=5.5, ymag=5.5)
        s, c = math.sin(math.pi / 12), math.cos(math.pi / 12)
        camera_pose = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, c, s, s * 17 + 4],
                [0, -s, c, c * 17],
                [0, 0, 0, 1],
            ]
        )
    scene.add(camera, pose=camera_pose)

    light_pose = tb3_to_se3(torch.tensor([3.14 / 2, 0, 0.2, 2, 10, 0])).mT
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100.0,
        innerConeAngle=np.pi / 3,
        outerConeAngle=np.pi / 2,
    )
    scene.add(light, pose=light_pose)

    light_pose = tb3_to_se3(torch.tensor([3.14 / 2, 0, -0.2, -2, 10, 0])).mT
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100.0,
        innerConeAngle=np.pi / 3,
        outerConeAngle=np.pi / 2,
    )
    scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(res, res)
    color, depth = r.render(scene)
    del r
    color = torch.from_numpy(np.ascontiguousarray(np.transpose(color, (2, 0, 1))))
    depth = torch.from_numpy(np.ascontiguousarray(depth))
    return color, depth
