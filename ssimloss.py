"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F

def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    #data_range = data_range[:, None, None, None]
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()
