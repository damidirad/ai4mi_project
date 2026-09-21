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

from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


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
        center_path = self.slice_paths[index]
        neighbor_index = min(max(index + offset, 0), len(self.slice_paths) - 1)
        neighbor_path = self.slice_paths[neighbor_index]

        if self._patient_id(neighbor_path) != self._patient_id(center_path):
            return center_path

        return neighbor_path

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
