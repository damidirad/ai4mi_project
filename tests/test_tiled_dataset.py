import io
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import build_dataset, SliceDataset
from geometry import Grid
from preprocessing_metadata import build_metadata, save_metadata
from tiling import TileLayout, write_patient_tiles
from main import img_transform, gt_transform, check_data, setup


class TiledDatasetTests(unittest.TestCase):
    def prepare(self, root, split='train'):
        layout = TileLayout((7, 5), (4, 4), (3, 3))
        for pid, base in [('Patient_01', 10), ('Patient_02', 100)]:
            x, y, z = np.indices((7, 5, 3))
            images = (base + x + 2*y + 20*z).astype(np.uint8)
            labels = ((x+y+z) % 5).astype(np.uint8)
            write_patient_tiles(root / split / 'tiles' / pid, images, labels, layout)
            record = build_metadata(pid, split, 'source.nii.gz', Grid(images.shape, np.eye(4)),
                                    images.shape, None, images.shape[:2], None, tile_layout=layout)
            save_metadata(root / split / 'metadata' / f'{pid}.json', record)

    def load(self, root, **kwargs):
        return build_dataset('train', root, tiling=True, img_transform=img_transform,
                             gt_transform=partial(gt_transform, 5), **kwargs)

    def test_25d_neighbors_and_center_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            dataset = self.load(root, slices=3)
            self.assertEqual(len(dataset), 24)
            for index in [0, 4, 8, 12, 23]:
                sample = dataset[index]
                base = 10 if sample['patient_id'] == 'Patient_01' else 100
                x, y = sample['tile_start'].tolist()
                z = sample['z']
                expected = [base + x + 2*y + 20*min(max(z+offset, 0), 2) for offset in [-1, 0, 1]]
                np.testing.assert_allclose(sample['images'][:, 0, 0].numpy()*255, expected, atol=1e-5)
                self.assertEqual(sample['gts'][:, 0, 0].argmax().item(), (x+y+z)%5)
                self.assertEqual(sample['valid_mask'].dtype, torch.bool)
            batch = next(iter(DataLoader(dataset, batch_size=3, num_workers=0)))
            self.assertEqual(tuple(batch['images'].shape), (3, 3, 4, 4))
            self.assertEqual(tuple(batch['gts'].shape), (3, 5, 4, 4))
            self.assertEqual(tuple(batch['valid_mask'].shape), (3, 4, 4))

    def test_debug_keeps_neighbor_context_and_single_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            full, debug = self.load(root, slices=5), self.load(root, slices=5, debug=True)
            self.assertEqual(len(debug), 10)
            torch.testing.assert_close(full[9]['images'], debug[9]['images'])
            self.assertEqual(tuple(self.load(root, slices=1)[0]['images'].shape), (1, 4, 4))

    def test_missing_files_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            with self.assertRaisesRegex(ValueError, 'requires --tiling'):
                build_dataset('train', root)
            (root / 'train/tiles/Patient_01/img/z0001_t0000.png').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or unexpected'):
                self.load(root)

    def test_corrupt_mask_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            dataset = self.load(root)
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(root / 'train/tiles/Patient_01/valid/t0000.png')
            with self.assertRaisesRegex(ValueError, 'Validity mask'):
                dataset[0]

    def test_test_split_has_no_label_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, 'test')
            for path in (root / 'test').glob('tiles/*/gt/*.png'):
                path.unlink()
            dataset = build_dataset('test', root, tiling=True, img_transform=img_transform, slices=3)
            self.assertNotIn('gts', dataset[0])

    def test_check_data_does_not_write_results_and_training_is_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val']:
                self.prepare(root / 'SEGTHOR', split)
            args = SimpleNamespace(dataset='SEGTHOR', data_root=root, tiling=True,
                slices=3, debug=False, workers=0, dest=root / 'results')
            with redirect_stdout(io.StringIO()) as output:
                check_data(args)
            self.assertIn('train: 24 samples', output.getvalue())
            self.assertIn('val: 24 samples', output.getvalue())
            self.assertFalse(args.dest.exists())
            with self.assertRaisesRegex(ValueError, 'reconstruction'):
                setup(args)

    def test_mismatched_split_configuration_rejected(self):
        from preprocessing_metadata import load_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val']:
                self.prepare(root / 'SEGTHOR', split)
            for path in (root / 'SEGTHOR/val/metadata').glob('*.json'):
                record = load_metadata(path)
                record['intensity'] = {'method': 'hu_window', 'window': [-400, 400]}
                save_metadata(path, record)
            args = SimpleNamespace(dataset='SEGTHOR', data_root=root, tiling=True,
                slices=1, debug=False, workers=0)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'configurations differ'):
                check_data(args)

    def test_legacy_factory_and_wrong_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ['img', 'gt']:
                folder = root / 'train' / kind
                folder.mkdir(parents=True)
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(folder / 'Patient_01_0000.png')
            dataset = build_dataset('train', root, img_transform=img_transform, gt_transform=partial(gt_transform, 5))
            self.assertIsInstance(dataset, SliceDataset)
            self.assertEqual(tuple(dataset[0]['images'].shape), (1, 4, 4))
            with self.assertRaisesRegex(ValueError, 'No tiled metadata'):
                self.load(root)


if __name__ == '__main__':
    unittest.main()
