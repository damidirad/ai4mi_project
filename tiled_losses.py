"""Additive tiled CE for uniform sampling of dataset tile entries.

Pixel weights are 1/coverage (zero padding). The fixed denominator per sample
is the dataset's weighted supervised class mass / number of tile entries.
Unlike tile/batch-local normalization this preserves per-voxel contributions
in expectation, including partial supervision and optional class weights.
"""
import torch
from torch import nn
from torch.nn import functional as F


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
