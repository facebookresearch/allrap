# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def weak_negative_correlation(v, z):
    v = v.flatten()
    z = z.flatten()
    v = v - v.mean()
    z = z - z.mean()
    cs = torch.nn.functional.cosine_similarity(v, z, dim=0)
    return cs.clamp(min=-0.05)
