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

import json
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

from preprocessing_metadata import load_metadata
from tiling import TileLayout


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 slices: int = 1):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.slices: int = slices

        assert self.slices > 0 and self.slices % 2 == 1, self.slices

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        self.slice_paths = [img_path for img_path, _ in self.files]
        if debug:
            self.files = self.files[:10]
            self.slice_paths = self.slice_paths[:10]

        print(f">> Created {subset} dataset with {len(self)} images and {self.slices} slice(s)...")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _patient_id(path: Path) -> str:
        return path.stem.rsplit("_", 1)[0]
    
    def _neighbor_img_path(self, index: int, offset: int) -> Path:
        center_id = self._patient_id(self.slice_paths[index])
        step = 1 if offset > 0 else -1
        for o in range(offset, 0, -step):  # return to the center if no neighbor is found
            j = index + o
            if 0 <= j < len(self.slice_paths) and self._patient_id(self.slice_paths[j]) == center_id:
                return self.slice_paths[j]

        return self.slice_paths[index]

    def _load_img_stack(self, index: int) -> Tensor:
        if self.slices == 1:
            return self.img_transform(Image.open(self.slice_paths[index]))

        radius = self.slices // 2
        imgs: list[Tensor] = []
        for offset in range(-radius, radius + 1):
            img_path = self._neighbor_img_path(index, offset)
            imgs.append(self.img_transform(Image.open(img_path)))

        return torch.cat(imgs, dim=0)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self._load_img_stack(index)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict


class TiledDataset(Dataset):
    """Load tiles with explicit patient, slice and tile coordinates."""

    def __init__(self, subset, root_dir, img_transform, gt_transform=None,
                 slices=1, debug=False):
        if subset not in ('train', 'val', 'test'):
            raise ValueError('Unknown dataset split')
        if isinstance(slices, bool) or not isinstance(slices, int) or slices <= 0 or slices % 2 == 0:
            raise ValueError('slices must be a positive odd integer')
        if img_transform is None or (subset != 'test' and gt_transform is None):
            raise ValueError('Image and training/validation label transforms are required')
        self.root = Path(root_dir) / subset
        self.test_mode = subset == 'test'
        self.slices = slices
        self.img_transform, self.gt_transform = img_transform, gt_transform
        self.records = {}
        self.layouts = {}
        self.coverage = {}
        self.samples = []
        paths = sorted((self.root / 'metadata').glob('*.json'))
        if not paths:
            raise ValueError(f'No tiled metadata found in {self.root}')
        for path in paths:
            record = load_metadata(path)
            pid = record['patient_id']
            if record['schema_version'] != 2 or record['split'] != subset:
                raise ValueError('Expected tiled metadata for the selected split')
            if pid != path.stem or Path(pid).name != pid or pid in ('.', '..') or pid in self.records:
                raise ValueError('Invalid or duplicate patient identifier')
            self.records[pid] = record
            layout = TileLayout(record['working']['shape'][:2], record['tiling']['size'], record['tiling']['stride'])
            self.layouts[pid] = layout
            self.coverage[pid] = layout.coverage()
            depth = record['working']['shape'][2]
            count = len(record['tiling']['starts_xy'])
            expected = {f'z{z:04d}_t{t:04d}.png' for z in range(depth) for t in range(count)}
            folder = self.root / 'tiles' / pid
            for kind in ('img',) if self.test_mode else ('img', 'gt'):
                actual = {p.name for p in (folder / kind).glob('*.png') if p.is_file()}
                if actual != expected:
                    raise ValueError(f'Missing or unexpected {kind} tiles for {pid}')
            masks = {p.name for p in (folder / 'valid').glob('*.png') if p.is_file()}
            if masks != {f't{t:04d}.png' for t in range(count)}:
                raise ValueError(f'Missing or unexpected validity masks for {pid}')
            self.samples.extend((pid, z, t) for z in range(depth) for t in range(count))
        folders = {p.name for p in (self.root / 'tiles').iterdir() if p.is_dir()}
        if folders != set(self.records):
            raise ValueError('Tile patients and metadata patients differ')
        sizes = {tuple(r['tiling']['size']) for r in self.records.values()}
        if len(sizes) != 1:
            raise ValueError('All patients must have the same tile size for batching')
        self.tile_size = next(iter(sizes))
        signatures = {json.dumps({key: record[key] for key in
            ('resample', 'requested_spacing_mm', 'intensity', 'interpolation')} |
            {'size': record['tiling']['size'], 'stride': record['tiling']['stride']},
            sort_keys=True) for record in self.records.values()}
        if len(signatures) != 1:
            raise ValueError('Patients have inconsistent preprocessing configurations')
        self.preprocessing_signature = next(iter(signatures))
        if debug:
            # Limit centres only; neighbouring slices still use the complete data.
            self.samples = self.samples[:10]

    def __len__(self):
        return len(self.samples)

    def _read(self, path):
        with Image.open(path) as image:
            array = np.array(image)
        if array.dtype != np.uint8 or array.shape != self.tile_size:
            raise ValueError(f'Expected uint8 tile of size {self.tile_size}: {path}')
        return array

    def __getitem__(self, index):
        pid, z, tile = self.samples[index]
        record = self.records[pid]
        folder = self.root / 'tiles' / pid
        radius = self.slices // 2
        images = []
        for offset in range(-radius, radius + 1):
            neighbor = min(max(z + offset, 0), record['working']['shape'][2] - 1)
            array = self._read(folder / 'img' / f'z{neighbor:04d}_t{tile:04d}.png')
            images.append(self.img_transform(Image.fromarray(array)))
        image = torch.cat(images, dim=0)
        if tuple(image.shape) != (self.slices, *self.tile_size):
            raise ValueError('Image transform must preserve tile dimensions and one channel per slice')
        mask = self._read(folder / 'valid' / f't{tile:04d}.png')
        x, y = record['tiling']['starts_xy'][tile]
        nx = min(self.tile_size[0], record['working']['shape'][0]-x)
        ny = min(self.tile_size[1], record['working']['shape'][1]-y)
        expected_mask = np.zeros(self.tile_size, dtype=np.uint8)
        expected_mask[:nx, :ny] = 255
        if not np.array_equal(mask, expected_mask):
            raise ValueError('Validity mask does not match tile geometry')
        sample = {'images': image, 'stems': f'{pid}_z{z:04d}_t{tile:04d}',
                  'patient_id': pid, 'z': z, 'tile_index': tile,
                  'tile_start': torch.tensor([x, y]),
                  'valid_mask': torch.from_numpy(mask == 255),
                  'pixel_weights': torch.from_numpy(self.layouts[pid].pixel_weights(tile, self.coverage[pid]))}
        if not self.test_mode:
            label = self._read(folder / 'gt' / f'z{z:04d}_t{tile:04d}.png')
            if not np.isin(label, [0, 63, 126, 189, 252]).all():
                raise ValueError('Invalid tile label encoding')
            target = self.gt_transform(Image.fromarray(label))
            if tuple(target.shape) != (5, *self.tile_size):
                raise ValueError('Label transform must preserve tile dimensions and five classes')
            sample['gts'] = target
        return sample

    def class_counts(self):
        """Count selected samples with inverse-overlap weights, excluding padding.

        Full datasets count every working voxel once. Debug subsets describe
        only their selected weighted samples. No CT or neighbouring images load.
        """
        if self.test_mode:
            raise ValueError('Class counts require labelled data')
        counts = np.zeros(5, dtype=np.float64)
        for pid, z, tile in self.samples:
            path = self.root / 'tiles' / pid / 'gt' / f'z{z:04d}_t{tile:04d}.png'
            encoded = self._read(path)
            if not np.isin(encoded, [0, 63, 126, 189, 252]).all():
                raise ValueError('Invalid tile label encoding')
            weights = self.layouts[pid].pixel_weights(tile, self.coverage[pid])
            counts += np.bincount((encoded // 63).ravel(), weights=weights.ravel(), minlength=5)
        return torch.from_numpy(counts)


def build_dataset(subset, root_dir, *, tiling=False, **kwargs):
    """Explicitly select the on-disk layout; never infer a training mode silently."""
    if tiling:
        return TiledDataset(subset, root_dir, **kwargs)
    if (Path(root_dir) / subset / 'tiles').exists():
        raise ValueError('Tiled data requires --tiling')
    return SliceDataset(subset, root_dir, **kwargs)
