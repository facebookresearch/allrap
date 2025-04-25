# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import icosphere
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import pytorch3d
import pytorch3d.io
import torch
import torchvision


def cam(dist=1, at=torch.zeros(3), azim=0, elev=0, device="cpu"):
    cam = pytorch3d.renderer.PerspectiveCameras().to(device)
    at = at.to(device)
    cam.R, cam.T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, at=at[None, :], azim=azim, elev=elev, device=device
    )
    return cam


@torch.no_grad()
def chessboard(r=1, z=0, n=8, device="cpu"):
    V, F, C = [], [], []
    t = torch.linspace(-r, r, n + 1).tolist()
    V.append([t[0], z - 1e-3, t[0]])
    V.append([t[0], z - 1e-3, t[-1]])
    V.append([t[-1], z - 1e-3, t[0]])
    V.append([t[-1], z - 1e-3, t[-1]])
    F.extend([[0, 1, 2], [2, 3, 1]])
    C.append([0, 0, 0])
    C.append([0, 0, 0])
    C.append([0, 0, 0])
    C.append([0, 0, 0])
    for i in range(n):
        for j in range((i + 1) % 2, n, 2):
            q = len(V)
            F.append((q, q + 1, q + 2))
            F.append((q + 1, q + 3, q + 2))
            for a in [t[i], t[i + 1]]:
                for b in [t[j], t[j + 1]]:
                    V.append((a, z, b))
                    C.append((1, 1, 1))
    return [
        torch.tensor(V, device=device),
        torch.tensor(F, device=device, dtype=torch.int64),
        torch.tensor(C, device=device, dtype=torch.float32),
    ]


@torch.no_grad()
def cylinder(x, y, r, N, rgb, device="cpu"):
    x = x.to(device)
    y = y.to(device)
    n = y - x
    n = torch.stack(
        [n, torch.randn_like(n, device=device), torch.randn_like(n, device=device)], 1
    )
    n = torch.linalg.qr(n).Q.mT
    assert torch.dot(x - y, n[0]) != 0
    assert torch.dot(n[0], n[1]).abs() < 1e-6
    assert torch.dot(n[0], n[2]).abs() < 1e-6
    assert torch.dot(n[1], n[2]).abs() < 1e-6

    t0 = torch.linspace(0, 2 * torch.pi, N + 1)[:N].to(device)
    t1 = t0 + torch.pi / N
    V = torch.cat(
        [
            x + r * torch.cos(t0)[:, None] * n[1] + r * torch.sin(t0)[:, None] * n[2],
            y + r * torch.cos(t1)[:, None] * n[1] + r * torch.sin(t1)[:, None] * n[2],
        ]
    )
    F = []
    for i in range(N):
        F.append([i, (i + 1) % N, i + N])
        F.append(
            [
                (i + 1) % N,
                (i + 1) % N + N,
                i + N,
            ]
        )
    F = torch.tensor(F).to(device)
    C = rgb[None, :].to(device).expand_as(V).contiguous()
    return [V, F, C]


@torch.no_grad()
def sphere(xyz, r, rgb, device="cpu", invert=False, n=4):
    if isinstance(xyz, (list, tuple)):
        xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
    if isinstance(rgb, (list, tuple)):
        rgb = torch.tensor(rgb, dtype=torch.float32, device=device)
    V, F = icosphere.icosphere(n)
    V = torch.from_numpy(V * r).float().to(device) + xyz.to(device)
    F = torch.from_numpy(F).long().to(device)
    if invert:
        F = torch.flip(F, (1,))
    C = rgb[None, :].expand_as(V).contiguous()
    return [V, F, C]


def make_mesh(objs, opacity=None):
    V, F, C = [], [], []
    offset = 0
    for v, f, c in objs:
        V.append(v)
        F.append(f + offset)
        C.append(c)
        offset += len(v)
    V = torch.cat(V)
    F = torch.cat(F)
    C = torch.cat(C)
    if opacity is not None:
        C = torch.cat([C, torch.ones_like(C[:, :1]) * opacity], 1)
    return pytorch3d.structures.Meshes(
        [V],
        [F],
        textures=pytorch3d.renderer.mesh.textures.TexturesVertex([C]),
    )


def render_mesh2(mesh, cam, resolution=512, pad=0, light=[0, 6, 0], a=0, b=1):
    device = cam.R.device
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=resolution + 2 * pad, blur_radius=a, faces_per_pixel=b, bin_size=0
    )

    lights = pytorch3d.renderer.PointLights(
        device=device,
        location=[light],
        # ambient_color=((0.9, 0.9, 0.9), ),
        # diffuse_color=((0.1, 0.1, 0.1), ),
        # specular_color=((0.0, 0.0, 0.0), )
    ).to(device)
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cam, raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device, cameras=cam, lights=lights
        ),
    )
    images = renderer(mesh)[0][:, :, :3]
    if pad > 0:
        images = images[pad:-pad, pad:-pad]
    return images.permute(2, 0, 1)


def annotate(image, msg):
    fnt = PIL.ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=32
    )
    image = torchvision.transforms.functional.to_pil_image(image)
    d = PIL.ImageDraw.Draw(image)
    d.text((10, 10), msg, font=fnt, fill=(10, 10, 10))
    return torchvision.transforms.functional.to_tensor(image)


class OpacityVertexShader(torch.nn.Module):
    def __init__(
        self,
        device="cpu",
    ) -> None:
        super().__init__()
        self.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        colors = meshes.sample_textures(fragments)
        w = colors[:, :, :, 1:, 3:] * torch.cumprod(1 - colors[:, :, :, :-1, 3:], dim=3)
        c = colors[:, :, :, 0, :3] * colors[:, :, :, 0, 3:] + (
            colors[:, :, :, 1:, :3] * w
        ).sum(dim=3)
        opacity = colors[:, :, :, 0, 3:] + w.sum(-2)
        return c, opacity


def render_mesh(mesh, cam, resolution=512, pad=0):
    device = cam.R.device
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=resolution + 2 * pad,
        blur_radius=0.0,
        faces_per_pixel=25,
    )
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cam, raster_settings=raster_settings
        ),
        shader=OpacityVertexShader(
            device=device,
        ),
    )
    images = renderer(mesh)
    images = images[0][0]
    if pad > 0:
        images = images[pad:-pad, pad:-pad, :3]
    return images.permute(2, 0, 1)
