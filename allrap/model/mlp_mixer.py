import torch
import inspect


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
            x = (
                x
                * torch.rand(len(x), 1, 1, device=x.device).ge(self.dropout)
                / (1 - self.dropout)
            )
        return x

    def __repr__(self):
        return f"Drop({self.dropout})"


def MLP(
    n,
    expand,
    activation=torch.nn.GELU,
    init_zero=True,
    residual=True,
    dropout=0.1,
):
    mlp = torch.nn.Sequential(
        Linear_BN(n, int(n * expand), activation=activation),
        torch.nn.Linear(int(n * expand), n),
        Drop(dropout),
    )
    if init_zero:
        with torch.no_grad():
            mlp[-2].weight.zero_()
            mlp[-2].bias.zero_()
    result = mlp
    if residual:
        result = Residual(mlp)
    return result


def MixerBlock(
    n_kpts=79,
    dim=32,
    expand=2,
    activation=torch.nn.GELU,
    init_zero=True,
    dropout=0.1,
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


class Attention(torch.nn.Module):
    def __init__(
        self,
        tokens,
        dim,
        key_dim=16,
        num_heads=4,
        attn_ratio=1,
        attn_bias=False,
        activation=torch.nn.GELU,
        init_zero=True,
    ):
        super().__init__()
        self.num_heads = num_heads if num_heads > 0 else dim // key_dim
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h)
        self.proj = torch.nn.Sequential(
            activation(),
            torch.nn.Linear(self.dh, dim),
        )
        if init_zero:
            with torch.no_grad():
                self.proj[1].weight.zero_()
        if attn_bias:
            self.register_buffer("attn_bias", torch.zeros(num_heads, tokens, tokens))

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # replace with k.mT
        if hasattr(self, "attn_bias"):
            attn = attn + self.attn_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def TransformerBlock(
    tokens=79,
    dim=64,
    key_dim=16,
    num_heads=4,
    mlp_ratio=2,
    attn_ratio=1,
    attn_bias=False,
    activation=torch.nn.GELU,
    dropout=0,
    init_zero=True,
):
    assert dropout == 0, dropout
    return torch.nn.Sequential(
        Residual(
            Attention(
                tokens,
                dim,
                key_dim,
                num_heads,
                attn_ratio,
                attn_bias,
                activation,
            )
        ),
        MLP(
            dim,
            mlp_ratio,
            activation,
            init_zero=init_zero,
            residual=True,
            dropout=0,
        ),
    )


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
            xy_centered = xy - vcm * v

            X = torch.cat([xy_centered, v], dim=2)
            X = self.input_projection(X)
            X = self.mlp_mixer(X)
            X = self.output_projection(X)

            # For visible points: use xy
            # For occluded point: use predicted_xy
            predicted_xy = X[:, :, :2] + vcm
            predicted_z = X[:, :, 2:]
            xyz = torch.cat([xy * v + predicted_xy * (1 - v), predicted_z], 2)

        else:  # Perspective camera
            X = torch.cat([xy, v], dim=2)

            X = self.input_projection(X)
            X = self.mlp_mixer(X)
            X = self.output_projection(X)

            # For visible points: use xy
            # For occluded point: use predicted_xy
            predicted_xy = X[:, :, :2]
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


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_kpts=79,
        dim=64,
        key_dim=32,
        num_heads=2,
        mlp_ratio=2,
        attn_ratio=1,
        attn_bias=True,
        pos_embed=True,
        activation=torch.nn.Hardswish,
        dropout=0,
        init_zero=True,
        hidden_layers=6,
        initial_nonlinearity=False,
        training_noise=0.05,
    ):
        super().__init__()
        self.n_kpts = n_kpts
        self.training_noise = training_noise

        if pos_embed:
            self.register_buffer(
                "position_bias", torch.randn(n_kpts, dim) * (dim**-0.5) * 0.1
            )

        # Increase the dimension of the visual input data to the transformer dimensions
        self.input_projection = torch.nn.Linear(3, dim)
        self.output_projection = torch.nn.Linear(dim, 3)
        layers = []

        if initial_nonlinearity:
            layers.append(activation())
        for _ in range(hidden_layers):
            layers.append(
                TransformerBlock(
                    n_kpts,
                    dim,
                    key_dim,
                    num_heads,
                    mlp_ratio,
                    attn_ratio,
                    attn_bias,
                    activation,
                    dropout,
                    init_zero,
                )
            )
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy, visible):
        """
        x: (B,skeleton,3)
        """
        v = visible[:, :, None]
        # Center using visible keypoints
        vcm = (xy * v).sum(1, keepdim=True) / v.sum(1, keepdim=True)
        if self.training and self.training_noise > 0:
            vcm += torch.randn_like(vcm) * self.training_noise
        xy_centered = xy - vcm * v

        X = torch.cat([xy_centered, v], dim=2)
        X = self.input_projection(X)
        if hasattr(self, "position_bias"):
            X = X + self.position_bias
        X = self.net(X)
        X = self.output_projection(X)

        # For visible points: use xy
        # For occluded point: use predicted_xy
        predicted_xy = X[:, :, :2] + vcm
        predicted_z = X[:, :, 2:]
        xyz = torch.cat([xy * v + predicted_xy * (1 - v), predicted_z], 2)
        return xyz
