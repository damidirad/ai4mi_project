import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tiling import TileLayout
from tiled_losses import TiledCrossEntropy, inverse_frequency_weights, make_tiled_loss
import test_tiled_dataset as fixtures
from main import check_data


class TiledLossTests(unittest.TestCase):
    def test_overlap_weights_sum_to_one_on_real_pixels(self):
        for shape, size, stride in [((7, 11), (4, 5), (2, 3)), ((2, 3), (4, 5), (2, 3)),
                                   ((8, 8), (4, 4), (4, 4))]:
            layout = TileLayout(shape, size, stride)
            summed = np.zeros(shape)
            coverage = layout.coverage()
            for t, (x, y) in enumerate(layout.starts):
                weights = layout.pixel_weights(t, coverage)
                nx, ny = min(size[0], shape[0]-x), min(size[1], shape[1]-y)
                summed[x:x+nx, y:y+ny] += weights[:nx, :ny]
                _, valid = layout.extract(coverage, t)
                self.assertTrue(np.all(weights[~valid] == 0))
            np.testing.assert_allclose(summed, 1, atol=1e-7)

    def test_loss_and_gradient_match_global_reference_across_layouts(self):
        # Two unequal scans: verify global voxel weighting, not equal tile means.
        for stride in [(4, 4), (2, 2), (3, 2)]:
            for idk in [list(range(5)), [0, 1, 3, 4]]:
                with self.subTest(stride=stride, idk=idk):
                    torch.manual_seed(7)
                    sources = [torch.randn(5, *shape, dtype=torch.float64, requires_grad=True)
                               for shape in [(7, 5), (3, 2)]]
                    labels = [(torch.arange(v.shape[1]*v.shape[2]).reshape(v.shape[1:]) % 5) for v in sources]
                    counts = sum((torch.bincount(v.flatten(), minlength=5) for v in labels))
                    class_weights = torch.tensor([1., 2., .5, 3., 1.], dtype=torch.float64)
                    active = torch.zeros(5, dtype=torch.float64)
                    active[idk] = class_weights[idk]
                    numerator = sum((-(F.log_softmax(v, dim=0).gather(0, y[None])[0]) * active[y]).sum()
                                    for v, y in zip(sources, labels))
                    reference = numerator / (counts * active).sum()
                    tiles, targets, weights = [], [], []
                    for source, label in zip(sources, labels):
                        layout = TileLayout(tuple(label.shape), (4, 4), stride)
                        for t, (x, y) in enumerate(layout.starts):
                            patch = source[:, x:x+4, y:y+4]
                            tiles.append(F.pad(patch, (0, 4-patch.shape[2], 0, 4-patch.shape[1])))
                            target, _ = layout.extract(label.numpy(), t)
                            targets.append(F.one_hot(torch.tensor(target), 5).permute(2, 0, 1))
                            weights.append(torch.from_numpy(layout.pixel_weights(t)))
                    criterion = TiledCrossEntropy(counts, len(tiles), idk, class_weights)
                    actual = criterion(torch.stack(tiles), torch.stack(targets), torch.stack(weights))
                    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
                    expected_grad = torch.autograd.grad(reference, sources, retain_graph=True)
                    actual_grad = torch.autograd.grad(actual, sources)
                    for a, b in zip(actual_grad, expected_grad):
                        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
                    # A short last batch must be weighted by its sample count in reporting.
                    values = [criterion(torch.stack(tiles[i:i+3]), torch.stack(targets[i:i+3]),
                                        torch.stack(weights[i:i+3])) * len(tiles[i:i+3])
                              for i in range(0, len(tiles), 3)]
                    torch.testing.assert_close(sum(values)/len(tiles), actual)

    def test_padding_and_ignored_classes_have_zero_gradient(self):
        logits = torch.randn(1, 5, 3, 3, requires_grad=True)
        labels = torch.zeros(1, 3, 3, dtype=torch.long)
        labels[0, 0, 0] = 2
        target = F.one_hot(labels, 5).permute(0, 3, 1, 2)
        weights = torch.ones(1, 3, 3)
        weights[:, 2, :] = 0
        criterion = TiledCrossEntropy([5, 0, 1, 0, 0], 1, [0, 1, 3, 4])
        value = criterion(logits, target, weights)
        value.backward()
        self.assertTrue(torch.all(logits.grad[:, :, 2, :] == 0))
        self.assertTrue(torch.all(logits.grad[:, :, 0, 0] == 0))
        altered = logits.detach().clone()
        altered[:, :, 2, :] = torch.tensor([100., -100., 50., 25., -20.])[None, :, None]
        altered[:, :, 0, 0] = 50
        torch.testing.assert_close(criterion(altered, target, weights), value)

    def test_dataset_counts_and_diagnostic_loss(self):
        fixture = fixtures.TiledDatasetTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                fixture.prepare(root / 'SEGTHOR', split)
            dataset = fixture.load(root / 'SEGTHOR')
            x, y, z = np.indices((7, 5, 3))
            expected = np.bincount(((x+y+z)%5).ravel(), minlength=5)*2
            np.testing.assert_allclose(dataset.class_counts().numpy(), expected)
            self.assertEqual(tuple(dataset[0]['pixel_weights'].shape), (4, 4))
            args = SimpleNamespace(dataset='SEGTHOR', data_root=root, tiling=True,
                check_loss=True, loss='ce', mode='full', slices=3, debug=False, workers=0)
            with redirect_stdout(io.StringIO()) as output:
                check_data(args)
            self.assertEqual(output.getvalue().count('padding gradients zero'), 2)
            for name in ('dice', 'ce_dice', 'boundary', 'ce_dice_boundary'):
                with self.assertRaisesRegex(ValueError, 'only ce'):
                    make_tiled_loss(dataset, name, list(range(5)))

    def test_optimizer_step_and_invalid_inputs(self):
        model = torch.nn.Conv2d(3, 5, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        image = torch.ones(1, 3, 4, 4)
        target = F.one_hot(torch.zeros(1, 4, 4, dtype=torch.long), 5).permute(0, 3, 1, 2)
        weights = torch.ones(1, 4, 4)
        criterion = TiledCrossEntropy([16, 0, 0, 0, 0], 1, list(range(5)))
        before = criterion(model(image), target, weights)
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        self.assertLess(criterion(model(image), target, weights).item(), before.item())
        with self.assertRaises(ValueError):
            TiledCrossEntropy([0, 1], 1, [0])
        with self.assertRaises(ValueError):
            criterion(model(image), target, -weights)
        with self.assertRaises(ValueError):
            inverse_frequency_weights([0, 0])


if __name__ == '__main__':
    unittest.main()
