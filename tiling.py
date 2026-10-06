"""Fixed XY tiles on the working voxel grid; no resizing or label-based selection."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def positive_pair(values, name):
    values = tuple(values)
    if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) or v <= 0 for v in values):
        raise ValueError(f'{name} must contain two positive integers')
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class TileLayout:
    shape: tuple[int, int]
    size: tuple[int, int] = (256, 256)
    stride: tuple[int, int] = (128, 128)

    def __post_init__(self):
        for name in ('shape', 'size', 'stride'):
            object.__setattr__(self, name, positive_pair(getattr(self, name), name))
        if any(step > size for step, size in zip(self.stride, self.size)):
            raise ValueError('Tile stride must not exceed tile size (would leave gaps)')

    @property
    def starts(self):
        axes = [range(0, max(0, (n - size + step - 1) // step) * step + 1, step)
                for n, size, step in zip(self.shape, self.size, self.stride)]
        return tuple((x, y) for x in axes[0] for y in axes[1])

    @property
    def padding_high(self):
        last = self.starts[-1]
        return tuple(start + size - n for start, size, n in zip(last, self.size, self.shape))

    def extract(self, plane, index, fill=0):
        """Return an owned tile and a boolean mask of real pixels."""
        if tuple(plane.shape) != self.shape:
            raise ValueError('Plane does not match tile layout')
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.starts):
            raise ValueError('Invalid tile index')
        x, y = self.starts[index]
        nx, ny = min(self.size[0], self.shape[0] - x), min(self.size[1], self.shape[1] - y)
        tile = np.full(self.size, fill, dtype=plane.dtype)
        valid = np.zeros(self.size, dtype=bool)
        tile[:nx, :ny] = plane[x:x+nx, y:y+ny]
        valid[:nx, :ny] = True
        return tile, valid

    def coverage(self):
        """Number of tiles covering each real working pixel."""
        result = np.zeros(self.shape, dtype=np.int32)
        for x, y in self.starts:
            result[x:x+self.size[0], y:y+self.size[1]] += 1
        return result

    def pixel_weights(self, index, coverage=None):
        """Inverse overlap count inside the scan, zero on padding."""
        if coverage is None:
            coverage = self.coverage()
        tile, valid = self.extract(coverage, index)
        weights = np.zeros(self.size, dtype=np.float32)
        np.divide(1., tile, out=weights, where=valid)
        return weights

    def to_dict(self):
        return {'size': list(self.size), 'stride': list(self.stride),
                'starts_xy': [list(p) for p in self.starts],
                'padding_low': [0, 0], 'padding_high': list(self.padding_high),
                'image_fill_uint8': 0, 'label_fill': 0,
                'valid_mask_values': [0, 255],
                'filename': 'z{z:04d}_t{tile:04d}.png',
                'mask_filename': 't{tile:04d}.png'}


def write_patient_tiles(destination, images, labels, layout):
    """Store image/label tiles and one valid mask per XY position (shared by Z).

    images are normalized uint8; labels are unencoded integer classes 0..4.
    Padding is zero in normalized intensity space, not an assumed HU value.
    """
    if images.ndim != 3 or images.shape != labels.shape or images.shape[:2] != layout.shape:
        raise ValueError('Images and labels must match the 3D working grid')
    if images.dtype != np.uint8 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Expected uint8 images and integer labels')
    if images.shape[2] < 1 or not np.isin(labels, [0, 1, 2, 3, 4]).all():
        raise ValueError('Expected nonempty Z and labels 0..4')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ('img', 'gt', 'valid'):
        (destination / folder).mkdir()
    for index in range(len(layout.starts)):
        for z in range(images.shape[2]):
            image, valid = layout.extract(images[:, :, z], index)
            label, _ = layout.extract(labels[:, :, z], index)
            name = f'z{z:04d}_t{index:04d}.png'
            Image.fromarray(image).save(destination / 'img' / name)
            Image.fromarray((label * 63).astype(np.uint8)).save(destination / 'gt' / name)
            if z == 0:
                Image.fromarray(valid.astype(np.uint8) * 255).save(destination / 'valid' / f't{index:04d}.png')
