import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from geometry import Grid
from preprocessing_metadata import build_metadata
from tiling import TileLayout
from evaluate_tiled import (restore_original_grid, save_original_prediction, volume_metrics,
                            evaluate_original_prediction)
import slice_segthor as pipeline
import main


class OriginalGridTests(unittest.TestCase):
    def record(self, target=None):
        affine = np.array([[0, -2., 0, 13.], [1., 0, 0, -7.], [0, 0, -3., 29.], [0, 0, 0, 1.]])
        source = Grid((5, 6, 7), affine)
        shape = list(source.shape)
        if target is not None:
            for axis, value in enumerate(target):
                shape[axis] = round(shape[axis]*source.spacing[axis]/value)
        layout = TileLayout(tuple(shape[:2]), (4, 4), (2, 2))
        return build_metadata('Patient_01', 'val', 'source.nii.gz', source,
                              tuple(shape), target, tuple(shape[:2]), None, tile_layout=layout)

    def test_none_xy_xyz_restore_exact_original_shape_and_coordinates(self):
        for target in [None, (.5, 3.), (.5, 3., 1.5), (2., 3., 6.)]:
            with self.subTest(target=target):
                record = self.record(target)
                shape = record['working']['shape']
                x, y, z = np.indices(shape)
                working = ((x+2*y+3*z)%5).astype(np.uint8)
                actual = restore_original_grid(working, record)
                positions = [np.floor(np.linspace(0, size-1, original)+.5).astype(int)
                             for size, original in zip(shape, [5, 6, 7])]
                expected = working[np.ix_(*positions)]
                self.assertEqual(actual.shape, (5, 6, 7))
                np.testing.assert_array_equal(actual, expected)
                if target is None:
                    np.testing.assert_array_equal(actual, working)

    def test_export_preserves_header_geometry_and_discrete_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self.record()
            affine = np.array(record['original']['affine_ras_mm'])
            reference = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.int16), affine)
            reference.header.set_xyzt_units('mm')
            reference.set_qform(affine, 1)
            reference.set_sform(affine, 2)
            nib.save(reference, root / 'ct.nii.gz')
            labels = (np.arange(210).reshape(5, 6, 7)%5).astype(np.uint8)
            save_original_prediction(labels, record, root / 'out.nii.gz', root / 'ct.nii.gz')
            output = nib.load(root / 'out.nii.gz')
            np.testing.assert_array_equal(np.asarray(output.dataobj), labels)
            np.testing.assert_allclose(output.affine, reference.affine)
            np.testing.assert_allclose(output.get_qform(), reference.get_qform())
            self.assertEqual(int(output.header['qform_code']), 1)
            self.assertEqual(int(output.header['sform_code']), 2)
            self.assertEqual(output.header.get_data_dtype(), np.dtype('uint8'))
            with self.assertRaises(FileExistsError):
                save_original_prediction(labels, record, root / 'out.nii.gz', root / 'ct.nii.gz')
            wrong = nib.Nifti1Image(np.zeros((5, 6, 7)), np.eye(4))
            wrong.header.set_xyzt_units('mm')
            nib.save(wrong, root / 'GT.nii.gz')
            with self.assertRaisesRegex(ValueError, 'same physical grid'):
                evaluate_original_prediction(labels, record, root / 'ct.nii.gz')

    def test_surface_distances_and_empty_status(self):
        grid = Grid((5, 5, 5), np.diag([2., 3., 4., 1.]))
        gt = np.zeros(grid.shape, dtype=np.uint8)
        pred = gt.copy()
        gt[1, 2, 2] = 1
        pred[2, 2, 2] = 1
        gt[0, 0, 0] = 2
        pred[4, 4, 4] = 3
        metrics = volume_metrics(pred, gt, grid)
        self.assertEqual(metrics['1'], {'dice': 0., 'hd95_mm': 2., 'assd_mm': 2., 'status': 'ok'})
        self.assertEqual(metrics['2']['status'], 'prediction_empty')
        self.assertIsNone(metrics['2']['hd95_mm'])
        self.assertEqual(metrics['3']['status'], 'target_empty')
        self.assertEqual(metrics['4']['dice'], 1.)
        json.dumps(metrics, allow_nan=False)
        shear = np.eye(4)
        shear[0, 1] = .5
        with self.assertRaisesRegex(ValueError, 'shear'):
            volume_metrics(pred, gt, Grid(grid.shape, shear))

    def test_checkpoint_cli_route_runs_tiles_to_original_nifti_and_metrics(self):
        class Tiny(torch.nn.Module):
            def __init__(self, input_channels, classes, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(input_channels, classes, 1)
            def forward(self, image):
                return self.conv(image)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source/train/Patient_01'
            source.mkdir(parents=True)
            ct = np.arange(210, dtype=np.int16).reshape(5, 6, 7)
            for name, data in [('Patient_01.nii.gz', ct), ('GT.nii.gz', (ct%5).astype(np.uint8))]:
                image = nib.Nifti1Image(data, np.diag([1., 1., 3., 1.]))
                image.header.set_xyzt_units('mm')
                nib.save(image, source / name)
            with patch.object(pipeline, 'sanity_ct', return_value=True):
                pipeline.slice_patient('Patient_01', root / 'data/SEGTHOR/val', root / 'source',
                    (256, 256), target_spacing=(.5, 1., 1.5), hu_window=(-400, 400),
                    tiling=True, tile_size=(4, 4), tile_stride=(2, 2))
            model = Tiny(1, 5)
            torch.save(model.state_dict(), root / 'weights.pt')
            args = SimpleNamespace(dataset='SEGTHOR', data_root=root/'data', dest=root/'results',
                                   debug=False, slices=1, gpu=False, evaluate_checkpoint=root/'weights.pt')
            with patch.dict(main.datasets_params['SEGTHOR'], {'net': Tiny}), redirect_stdout(io.StringIO()):
                main.evaluate_tiled_checkpoint(args)
            output = nib.load(root / 'results/Patient_01.nii.gz')
            self.assertEqual(output.shape, (5, 6, 7))
            report = json.loads((root / 'results/metrics.json').read_text())
            self.assertEqual(report['grid'], 'original_CT')
            self.assertEqual(set(report['patients']['Patient_01']), {'1', '2', '3', '4'})
            self.assertTrue(0 <= report['mean_foreground_dice'] <= 1)


if __name__ == '__main__':
    unittest.main()
