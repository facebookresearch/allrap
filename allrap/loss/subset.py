# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import faiss.contrib.torch_utils
import torch

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


def tuple_aligned_coord_loss(bnk3, cpa_with_grad=False, scaling="perspective"):
    assert scaling in ["orthographic", "perspective", "none"]
    B, N, K, _ = bnk3.shape
    nb3k = bnk3.permute(1, 0, 3, 2).double()  # (N,B,3,K)
    if scaling == "perspective":
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
        R = align_centered_point_clouds(nbk3, N_K3)

    # aligned = nbk3 @ R
    # s = torch.linalg.svdvals(aligned.view(N, B, K * 3))
    # f=(s[:,1:]>1e-12).double()

    residual = nbk3 @ R - N_K3
    s = torch.linalg.svdvals(residual.view(N, B, K * 3))[:, : min(B, K * 2)]
    if scaling == "orthographic":
        s2 = NK3.flatten(1).std(1, keepdim=True)
    elif scaling == "none":
        s2 = 1
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


def nearby_tuple_aligned_coord_loss(poses, n, k, scaling="perspective"):
    B, K, THREE = poses.shape
    with torch.no_grad():
        x = poses.transpose(0, 1).flatten(1, 2)
        tuples = faiss_knn(x[torch.randperm(K)[:n].to(poses.device)], x, k)[1]
    bnk3 = poses[:, tuples]
    return tuple_aligned_coord_loss(bnk3, scaling=scaling)


def random_tuple_aligned_coord_loss(poses, n, k, scaling="perspective"):
    B, K, THREE = poses.shape
    c = random_k_tuples(n, k, K)
    bnk3 = poses[:, c]
    return tuple_aligned_coord_loss(bnk3, scaling=scaling)
