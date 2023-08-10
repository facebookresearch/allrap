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
    norm_type="bn",
    dropout=0.1,
):
    assert norm_type in ["bn", "ln", "None"]
    # expand provides the ratio in which we increase the
    # TODO: Try initialising the second linear as 0 so that
    # the residual makes it an identity
    if norm_type == "bn":
        mlp = torch.nn.Sequential(
            Linear_BN(n, int(n * expand), activation=activation),
            torch.nn.Linear(int(n * expand), n),
            Drop(dropout),
        )

    elif norm_type == "ln":
        # layer_n = torch.nn.LayerNorm([lay_norm_dim_1, lay_norm_dim_2])
        layer_n = torch.nn.LayerNorm([n])
        mlp = torch.nn.Sequential(
            layer_n,
            torch.nn.Linear(n, n * expand),
            activation(),
            torch.nn.Linear(n * expand, n),
            Drop(dropout),
        )

    elif norm_type == "None":
        mlp = torch.nn.Sequential(
            torch.nn.Linear(n, n * expand),
            activation(),
            torch.nn.Linear(n * expand, n),
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


def NonMixerBlock(
    dim=384,
    expand=4,
    activation=torch.nn.GELU,
    init_zero=True,
    residual=True,
    norm_type="bn",
    dropout=0.1,
):
    return MLP(
        n=dim,
        expand=expand,
        init_zero=init_zero,
        residual=residual,
        dropout=dropout,
        norm_type=norm_type,
        activation=activation,
    )


def MixerBlock(
    n_kpts=79,
    dim=32,
    expand=2,
    activation=torch.nn.GELU,
    init_zero=True,
    residual=True,
    norm_type="bn",
    dropout=0.1,
):
    if isinstance(expand,(int,float)):
        expand=(expand,expand)
    return torch.nn.Sequential(
        Transpose(),
        MLP(
            n=n_kpts,
            expand=expand[0],
            init_zero=init_zero,
            residual=residual,
            dropout=dropout,
            norm_type=norm_type,
            activation=activation,
        ),
        Transpose(),
        MLP(
            n=dim,
            expand=expand[1],
            init_zero=init_zero,
            residual=residual,
            dropout=dropout,
            norm_type=norm_type,
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
        norm_type="bn",
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
    norm_type="bn",
    dropout=0,
    init_zero=True,
):
    assert dropout == 0, dropout
    assert norm_type == "bn"
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
                norm_type,
            )
        ),
        Residual(
            MLP(
                dim,
                mlp_ratio,
                activation,
                init_zero=init_zero,
                residual=True,
                norm_type=norm_type,
                dropout=0,
            )
        ),
    )


# FINAL
class MLPMixerPoseNet(torch.nn.Module):
    def __init__(
        self,
        n_kpts=79,
        activation=torch.nn.Hardswish,
        expand=2,
        hidden_layers=12,
        all_vis=False,
        dropout=0,
        norm_type="bn",
        residual=True,
        dim=32 * 2,
        init_zero=True,
        initial_nonlinearity=False,
    ):
        super(MLPMixerPoseNet, self).__init__()
        self.n_kpts = n_kpts
        self.pose_dim = 3
        self.dim = dim

        # Increase the dimension of the visual input data to the transformer dimensions
        self.input_projection = torch.nn.Linear(self.pose_dim, dim)
        self.output_projection = torch.nn.Linear(dim, self.pose_dim)
        layers = []

        if initial_nonlinearity:
            layers.append(activation())
        for _ in range(hidden_layers):
            layers.append(
                MixerBlock(
                    n_kpts,
                    dim,
                    expand,
                    activation,
                    init_zero,
                    residual,
                    norm_type,
                    dropout,
                )
            )
        self.net = torch.nn.Sequential(*layers)
        print('#param', sum([v.numel() for v in self.parameters()]))

    def forward(self, x):
        """
        x: (B,skeleton,3)
        """
        with torch.no_grad():
            vcm = (x*x[:,:,2:]).sum(1,keepdim=True)/x[:,:,2:].sum(1,keepdim=True)
            if self.training:
                vcm+=torch.randn_like(vcm)*0.05
            vcm[:,:,2].zero_()
        x=x-vcm
            
        # Input
        visible = x[:, :, 2:]
        pose_xy = x[:, :, :2]

        x = self.input_projection(x)
        x = self.net(x)
        x = self.output_projection(x)
        # Predicted
        predicted_xy = x[:, :, :2]
        predicted_z = x[:, :, 2:]

        # For xy: Mask out the occluded points and use only predicted values for
        # them and vice versa for the visible points, for z: concat predicted z to it
        poses = torch.cat(
            [pose_xy * visible + predicted_xy * (1 - visible), predicted_z], 2
        )
        return poses + vcm

    def forward_perspective(self, x, cam, scale_pred=False):
        # Input
        visible = x[:, :, 2:]
        pose_xy = x[:, :, :2]

        x = self.input_projection(x)
        x = self.net(x)
        x = self.output_projection(x)

        # Prediicted
        predicted_xy = x[:, :, :2]
        predicted_z = torch.nn.functional.softplus(x[:, :, 2:])

        # For xy: Mask out the occluded points and use only predicted values for
        # them and vice versa for the visible points, for z: concat predicted z to it
        if scale_pred:
            poses = torch.cat(
                [
                    pose_xy * visible + predicted_xy * (1 - visible),
                    predicted_z,
                ],
                2,
            )
            poses = cam.unproject_points(poses)
        else:
            poses = cam.unproject_points(
                torch.cat([pose_xy, predicted_z], 2)
            ) * visible + torch.cat(
                [
                    predicted_xy,
                    predicted_z,
                ],
                2,
            ) * (
                1 - visible
            )
        return poses



class MLPNonMixerPoseNet(torch.nn.Module):
    def __init__(
        self,
        n_kpts=79,
        activation=torch.nn.Hardswish,
        expand=4,
        hidden_layers=6,
        all_vis=False,
        dropout=0.0,
        norm_type="bn",
        residual=True,
        dim=512,
        init_zero=True,
        initial_nonlinearity=False,
    ):
        super(MLPNonMixerPoseNet, self).__init__()
        self.n_kpts = n_kpts
        self.pose_dim = 3
        self.dim = dim

        # Increase the dimension of the visual input data to the transformer dimensions
        self.input_projection = torch.nn.Linear(n_kpts * self.pose_dim, dim)
        self.output_projection = torch.nn.Linear(dim, n_kpts * self.pose_dim)
        layers = []
        if initial_nonlinearity:
            layers.append(activation())
        for _ in range(hidden_layers):
            layers.append(
                NonMixerBlock(
                    dim, expand, activation, init_zero, residual, norm_type, dropout
                )
            )
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B,skeleton,3)
        """

        # Input
        visible = x[:, :, 2:]
        pose_xy = x[:, :, :2]

        x = x.flatten(1, 2)
        x = self.input_projection(x)
        x = self.net(x)
        x = self.output_projection(x)
        x = x.reshape(-1, self.n_kpts, self.pose_dim)
        # Predicted
        predicted_xy = x[:, :, :2]
        predicted_z = x[:, :, 2:]

        # For xy: Mask out the occluded points and use only predicted values for
        # them and vice versa for the visible points, for z: concat predicted z to it
        poses = torch.cat(
            [pose_xy * visible + predicted_xy * (1 - visible), predicted_z], 2
        )
        return poses

    def forward_perspective(self, x, scale_pred=False):
        # Input
        visible = x[:, :, 2:]
        pose_xy = x[:, :, :2]

        x = x.flatten(1, 2)
        x = self.input_projection(x)
        x = self.net(x)
        x = self.output_projection(x)
        x = x.reshape(-1, self.n_kpts, self.pose_dim)

        # Prediicted
        predicted_xy = x[:, :, :2]
        predicted_z = torch.nn.functional.softplus(x[:, :, 2:])
        # predicted_z = torch.tanh(x[:, :, 2:])+1.1

        # For xy: Mask out the occluded points and use only predicted values for
        # them and vice versa for the visible points, for z: concat predicted z to it
        if scale_pred:
            poses = torch.cat(
                [
                    pose_xy * visible * predicted_z
                    + predicted_xy * (1 - visible) * predicted_z,
                    predicted_z,
                ],
                2,
            )
        else:
            poses = torch.cat(
                [
                    pose_xy * visible * predicted_z + predicted_xy * (1 - visible),
                    predicted_z,
                ],
                2,
            )
        return poses


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
        norm_type="bn",
        dropout=0,
        init_zero=True,
        hidden_layers=6,
        all_vis=False,
        initial_nonlinearity=False,
    ):
        super().__init__()
        self.n_kpts = n_kpts
        self.pose_dim = 3
        self.dim = dim

        if pos_embed:
            self.register_buffer(
                "position_bias", torch.randn(n_kpts, dim) * (dim**-0.5) * 0.1
            )

        # Increase the dimension of the visual input data to the transformer dimensions
        self.input_projection = torch.nn.Linear(self.pose_dim, dim)
        self.output_projection = torch.nn.Linear(dim, self.pose_dim)
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
                    norm_type,
                    dropout,
                    init_zero,
                )
            )
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B,skeleton,3)
        """

        # Input
        visible = x[:, :, 2:]
        pose_xy = x[:, :, :2]

        x = self.input_projection(x)
        if hasattr(self, "position_bias"):
            x = x + self.position_bias
        x = self.net(x)
        x = self.output_projection(x)
        # Predicted
        predicted_xy = x[:, :, :2]
        predicted_z = x[:, :, 2:]

        # For xy: Mask out the occluded points and use only predicted values for
        # them and vice versa for the visible points, for z: concat predicted z to it
        poses = torch.cat(
            [pose_xy * visible + predicted_xy * (1 - visible), predicted_z], 2
        )
        return poses
