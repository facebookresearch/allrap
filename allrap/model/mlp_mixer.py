# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import torch


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, activation=None):
        super().__init__()
        self.add_module("l", torch.nn.Linear(a, b, bias=False))
        if bn_weight_init == "na":
            bn = torch.nn.BatchNorm2d(b, affine=False)
        else:
            bn = torch.nn.BatchNorm1d(b)
            torch.nn.init.constant_(bn.weight, bn_weight_init)
            torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if activation is not None:
            if len(inspect.signature(activation).parameters):
                self.add_module("activation", activation(b))
            else:
                self.add_module("activation", activation())

    def forward(self, x):
        x = self[0](x)
        x = self[1](x.flatten(0, -2)).reshape_as(x)
        if len(self) == 3:
            x = self[2](x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return x + self.model(x)


class Drop(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.training and self.dropout > 0:
            #x = x * torch.rand_like(x[:, :1, :]).ge(self.dropout) / (1 - self.dropout)
            x = x * torch.randn_like(x[:, :, :]).mul(self.dropout).add(1)
        return x

    def __repr__(self):
        return f"Drop({self.dropout})"


def MLP(
    n,
    expand,
    activation=torch.nn.ReLU,
    init_zero=True,
    residual=True,
    dropout=0.1,
):
    mlp = torch.nn.Sequential(
        Linear_BN(n, int(n * expand), activation=activation),
        Drop(dropout),
        torch.nn.Linear(int(n * expand), n),
    )
    if init_zero:
        with torch.no_grad():
            mlp[2].weight.zero_()
            mlp[2].bias.zero_()
    if residual:
        mlp = Residual(mlp)
    return mlp


def MixerBlock(
    n_kpts=79,
    dim=32,
    expand=2,
    activation=torch.nn.ReLU,
    init_zero=True,
    dropout=0,
):
    if isinstance(expand, (int, float)):
        expand = (expand, expand)
    return torch.nn.Sequential(
        Transpose(),
        MLP(
            n=n_kpts,
            expand=expand[0],
            init_zero=init_zero,
            dropout=dropout,
            activation=activation,
        ),
        Transpose(),
        MLP(
            n=dim,
            expand=expand[1],
            init_zero=init_zero,
            dropout=dropout,
            activation=activation,
        ),
    )


class Transpose(torch.nn.Module):
    def forward(self, x):
        return torch.transpose(x, 1, 2)  # x.mT


class MLPMixer(torch.nn.Module):
    def __init__(
        self,
        n_kpts=79,
        activation=torch.nn.ReLU,
        expand=2,
        hidden_layers=32,
        dropout=0,
        dim=32,
        init_zero=True,
        camera="orthographic",
        training_noise=0.05,
    ):
        super().__init__()
        self.camera = camera
        self.training_noise = training_noise
        self.n_kpts = n_kpts

        # Increase the dimension of the visual input data to the transformer dimensions
        self.input_projection = torch.nn.Linear(3, dim)
        self.output_projection = torch.nn.Linear(dim, 3)
        layers = []

        for _ in range(hidden_layers):
            layers.append(
                MixerBlock(
                    n_kpts,
                    dim,
                    expand,
                    activation,
                    init_zero,
                    dropout,
                )
            )
        self.mlp_mixer = torch.nn.Sequential(*layers)

    def forward(self, xy, visible):
        """
        xy: tensor (batch, n_keypoints, 2)
        visible: tensor (batch, n_keypoints)
        """

        v = visible[:, :, None]

        if self.camera == "orthographic":
            # Center using visible keypoints
            vcm = (xy * v).sum(1, keepdim=True) / v.sum(1, keepdim=True)
            if self.training and self.training_noise > 0:
                vcm += torch.randn_like(vcm) * self.training_noise
            xy_centered = xy - vcm

            X = torch.cat([xy_centered, v], dim=2)
            X = self.input_projection(X)
            X = self.mlp_mixer(X)
            X = self.output_projection(X)

            # For visible points: use xy
            # For occluded point: use predicted_xy
            predicted_xy = X[:, :, :2] + vcm
            predicted_z = X[:, :, 2:]
            xyz = torch.cat([xy * v + predicted_xy * (1 - v), predicted_z], 2)

        else:  # i.e. perspective camera
            if self.training and self.training_noise > 0:
                noise = torch.randn_like(xy[:, :1, :]) * self.training_noise
                xy_n = xy + noise
            else:
                xy_n = xy

            X = torch.cat([xy_n, v], dim=2)

            X = self.input_projection(X)
            X = self.mlp_mixer(X)
            X = self.output_projection(X)

            # For visible points: use xy
            # For occluded point: use predicted_xy
            predicted_xy = X[:, :, :2]
            if self.training and self.training_noise > 0:
                predicted_xy = predicted_xy - noise
            predicted_z = torch.nn.functional.softplus(X[:, :, 2:])
            xyz = self.camera.unproject_points(
                torch.cat([xy, predicted_z], 2)
            ) * v + torch.cat(
                [
                    predicted_xy,
                    predicted_z,
                ],
                2,
            ) * (
                1 - v
            )
        return xyz
