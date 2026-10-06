"""Load stage-3 tiles with explicit patient/Z/tile coordinates."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing_metadata import load_metadata
from tiling import TileLayout


class TiledDataset(Dataset):
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
