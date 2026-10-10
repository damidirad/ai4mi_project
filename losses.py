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
from torch import Tensor, einsum, nn
from torch.nn import functional as F
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
        # foreground classes only to prevent the background to dominate the loss
        self.idk = [k for k in kwargs['idk'] if k != 0]
        print(f"Initialized {self.__class__.__name__} with {kwargs}, using classes {self.idk}")

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
        # weight of the boundary term is increased every epoch in main.py so the
        # region losses can first locate the organs
        self.boundary_weight = kwargs.get('boundary_weight', 1.0)
        print(f"Initialized {self.__class__.__name__}")

    def __call__(self, pred_softmax, weak_target):
        return (self.ce(pred_softmax, weak_target)
               + self.dice(pred_softmax, weak_target)
               + self.boundary_weight * self.boundary(pred_softmax, weak_target))


def validate_tiled_loss(name):
    if name != 'ce':
        raise ValueError('Tiled loss currently supports only ce; Dice needs a separate aggregation '
                         'definition and Boundary needs full-grid distance maps')


def inverse_frequency_weights(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError('Expected finite nonnegative class counts with positive total')
    weights = torch.rsqrt(counts / counts.sum() + 1e-10)
    return (weights / weights.mean()).float()


class TiledCrossEntropy(nn.Module):
    """Cross-entropy for uniformly sampled tiles, correcting overlap and padding.

    Pixel weights are inverse coverage, with zero weight on padding. A fixed
    dataset-based denominator preserves per-voxel contributions in expectation,
    including partial supervision and optional class weights.
    """

    def __init__(self, class_counts, sample_count, idk, class_weights=None):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float64).detach().clone()
        if counts.ndim != 1 or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Invalid dataset class counts')
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError('sample_count must be a positive integer')
        selected = list(idk)
        if not selected or len(set(selected)) != len(selected) or any(
                type(k) is not int or not 0 <= k < counts.numel() for k in selected):
            raise ValueError('Invalid supervised class indices')
        weights = torch.ones_like(counts) if class_weights is None else torch.as_tensor(
            class_weights, dtype=torch.float64).detach().clone()
        if weights.shape != counts.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid class weights')
        active = torch.zeros_like(counts)
        active[selected] = weights[selected]
        mass = (counts * active).sum()
        if mass <= 0:
            raise ValueError('Dataset has no positively weighted supervised pixels')
        self.register_buffer('class_weights', active)
        self.register_buffer('normalizer', mass / sample_count)

    def forward(self, logits, target, pixel_weights):
        if logits.ndim != 4 or target.shape != logits.shape or logits.shape[1] != self.class_weights.numel():
            raise ValueError('Expected matching BCHW logits and one-hot targets')
        if logits.shape[0] == 0 or pixel_weights.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError('Expected BHW pixel weights and a nonempty batch')
        if not torch.isfinite(logits).all() or not torch.isfinite(pixel_weights).all() or (pixel_weights < 0).any():
            raise ValueError('Logits must be finite and pixel weights finite/nonnegative')
        if not ((target == 0) | (target == 1)).all() or not (target.sum(1) == 1).all():
            raise ValueError('Expected one-hot target at every pixel')
        # Stable logits-based CE, with ignored classes contributing exactly zero.
        class_weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        weighted_target = target.to(logits) * class_weights[None, :, None, None]
        per_pixel = -(weighted_target * F.log_softmax(logits, dim=1)).sum(1)
        per_sample = (per_pixel * pixel_weights.to(logits)).sum((1, 2))
        return per_sample.mean() / self.normalizer.to(logits)


def make_tiled_loss(dataset, name, idk, class_weights=None, counts=None):
    validate_tiled_loss(name)
    counts = dataset.class_counts() if counts is None else counts
    return TiledCrossEntropy(counts, len(dataset), idk, class_weights)
