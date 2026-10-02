#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
from torch import Tensor, einsum
from scipy.ndimage import distance_transform_edt

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.weights = kwargs.get('weights', None)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        if self.weights is not None:
            weights = self.weights[self.idk].to(mask.device).view(1, -1, 1, 1)
            mask = mask * weights

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = kwargs.get('smooth', 1e-8)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        intersection = einsum("bkwh,bkwh->bk", pred, target)
        union = einsum("bkwh->bk", pred) + einsum("bkwh->bk", target)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CEDiceLoss():
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(**kwargs)
        self.dice = DiceLoss(idk=kwargs['idk'])
        print(f"Initialized {self.__class__.__name__}")

    def __call__(self, pred_softmax, weak_target):
        return self.ce(pred_softmax, weak_target) + self.dice(pred_softmax, weak_target)


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # signed distance transform per class: negative inside the object, positive outside
    K = seg.shape[0]
    res = np.zeros_like(seg, dtype=np.float32)
    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = distance_transform_edt(negmask) * negmask \
                - (distance_transform_edt(posmask) - 1) * posmask
    return res


class BoundaryLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # distance maps are derived from the fixed ground truth, so no gradient needed here
        with torch.no_grad():
            target_np = weak_target.detach().cpu().numpy()
            dist_maps = np.stack([one_hot2dist(sample) for sample in target_np])
            dist_maps_t = torch.tensor(dist_maps, dtype=torch.float32, device=pred_softmax.device)

        pc = pred_softmax[:, self.idk, ...]
        dc = dist_maps_t[:, self.idk, ...]

        loss = einsum("bkwh,bkwh->bkwh", pc, dc).mean()

        return loss


class CEDiceBoundaryLoss():
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(**kwargs)
        self.dice = DiceLoss(idk=kwargs['idk'])
        self.boundary = BoundaryLoss(idk=kwargs['idk'])
        print(f"Initialized {self.__class__.__name__}")

    def __call__(self, pred_softmax, weak_target):
        return (self.ce(pred_softmax, weak_target)
               + self.dice(pred_softmax, weak_target)
               + self.boundary(pred_softmax, weak_target))
