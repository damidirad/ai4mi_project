import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from geometry import Grid
from preprocessing_metadata import build_metadata
from reconstruct_tiled import SliceAccumulator, reconstruct_working_volume, predict_working_volume
from tiling import TileLayout
import test_tiled_dataset as fixtures


class ReconstructionTests(unittest.TestCase):
    def test_probability_average_precedes_argmax(self):
        layout = TileLayout((3, 1), (2, 1), (1, 1))
        acc = SliceAccumulator(layout, classes=2)
        # First tile weakly chooses 0; second strongly chooses 1.
        acc.add(1, np.broadcast_to(np.array([.01, .99])[:, None, None], (2, 2, 1)))
        acc.add(0, np.broadcast_to(np.array([.6, .4])[:, None, None], (2, 2, 1)))
        np.testing.assert_allclose(acc.probabilities()[:, 1, 0], [.305, .695])
        np.testing.assert_array_equal(acc.labels()[:, 0], [0, 1, 1])

    def test_coordinate_pattern_roundtrip_with_padding_and_slice_order(self):
        for shape in [(7, 11, 3), (2, 3, 2), (8, 8, 1)]:
            with self.subTest(shape=shape):
                layout = TileLayout(shape[:2], (4, 4), (3, 2))
                x, y, z = np.indices(shape)
                expected = ((x + 2*y + 3*z) % 5).astype(np.uint8)
                metadata = build_metadata('Patient_01', 'val', 'source.nii.gz',
                    Grid(shape, np.eye(4)), shape, None, shape[:2], None, tile_layout=layout)
                def stream():
                    for z in range(shape[2]):
                        for t in reversed(range(len(layout.starts))):
                            labels, valid = layout.extract(expected[:, :, z], t, fill=4)
                            # Padded pixels confidently predict a foreground class.
                            probabilities = np.eye(5, dtype=np.float32)[labels].transpose(2, 0, 1)
                            yield z, t, probabilities
                actual = reconstruct_working_volume(metadata, stream())
                self.assertEqual(actual.shape, shape)
                self.assertEqual(actual.dtype, np.uint8)
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_duplicate_and_missing_predictions(self):
        layout = TileLayout((3, 3), (2, 2), (1, 1))
        p = np.full((5, 2, 2), .2, dtype=np.float32)
        for invalid in [p.astype(int), p[:, :, :1], np.full_like(p, np.nan), p*2,
                        np.stack([np.full((2, 2), -1.), *[np.full((2, 2), .5)]*4])]:
            with self.assertRaises(ValueError):
                SliceAccumulator(layout).add(0, invalid)
        acc = SliceAccumulator(layout)
        with self.assertRaises(ValueError):
            acc.add(-1, p)
        acc.add(0, p)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            acc.add(0, p)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            acc.labels()
        metadata = build_metadata('Patient_01', 'val', 'source.nii.gz',
            Grid((3, 3, 2), np.eye(4)), (3, 3, 2), None, (3, 3), None, tile_layout=layout)
        for stream in [[], [(1, 0, p)], [(2, 0, p)], [(0, t, p) for t in range(4)],
                       [(0, t, p) for t in range(4)] + [(1, 0, p), (0, 1, p)]]:
            with self.assertRaises(ValueError):
                reconstruct_working_volume(metadata, stream)

    def test_model_inference_2d_25d_and_mode_restoration(self):
        fixture = fixtures.TiledDatasetTests()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Dropout()
                self.observed = []
            def forward(self, images):
                self.observed.append((self.training, torch.is_grad_enabled(), images.shape[1]))
                logits = torch.zeros(images.shape[0], 5, *images.shape[2:])
                logits[:, 3] = 2
                return logits
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture.prepare(root)
            for slices in [1, 3]:
                dataset = fixture.load(root, slices=slices)
                model = Model()
                model.train()
                model.child.eval()
                output = predict_working_volume(model, dataset, 'Patient_02', batch_size=3)
                np.testing.assert_array_equal(output, np.full((7, 5, 3), 3, dtype=np.uint8))
                self.assertTrue(model.training)
                self.assertFalse(model.child.training)
                self.assertTrue(all(state == (False, False, slices) for state in model.observed))
            truncated = fixture.load(root, debug=True)
            with self.assertRaisesRegex(ValueError, 'all patient tiles'):
                predict_working_volume(model, truncated, 'Patient_01')

    def test_model_mode_restored_on_failure(self):
        fixture = fixtures.TiledDatasetTests()
        class Broken(torch.nn.Module):
            def forward(self, image):
                raise RuntimeError('synthetic failure')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture.prepare(root)
            model = Broken().train()
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                predict_working_volume(model, fixture.load(root), 'Patient_01')
            self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
