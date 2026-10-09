"""Original-grid evaluation and unweighted-CE baseline training."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import main
import slice_segthor as preprocessing
from evaluate_tiled import restore_original_grid
from geometry import Grid
from preprocessing_metadata import build_metadata


class FullOriginalEvaluationTests(unittest.TestCase):
    def test_resize_coordinates_restore_original_grid(self):
        record = build_metadata('Patient_01', 'val', 'unused.nii.gz',
                                Grid((4, 4, 3), np.diag([1., 2., 3., 1.])),
                                (4, 4, 3), None, (2, 2), None)
        labels = np.array([[1, 2], [3, 4]], dtype=np.uint8)[:, :, None].repeat(3, axis=2)
        restored = restore_original_grid(labels, record)
        np.testing.assert_array_equal(restored, labels.repeat(2, axis=0).repeat(2, axis=1))

    def test_baseline_and_25d_train_select_and_reload_original_grid(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for slices, resample, weighting in [(1, 'none', 'none'), (3, 'xyz', 'inverse_frequency')]:
                with self.subTest(slices=slices), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    for pid in ('Patient_01', 'Patient_02'):
                        folder = root/'source/train'/pid
                        folder.mkdir(parents=True)
                        x, y, z = np.indices((33, 35, 3))
                        for name, data in [(f'{pid}.nii.gz', (x*10+y*3+z*20-200).astype(np.int16)),
                                           ('GT.nii.gz', ((x+y+z)%5).astype(np.uint8))]:
                            image = nib.Nifti1Image(data, np.diag([1., 1., 3., 1.]))
                            image.header.set_xyzt_units('mm')
                            nib.save(image, folder/name)
                    prep = SimpleNamespace(source_dir=root/'source', dest_dir=root/'data/SEGTHOR',
                        retains=1, fold=0, process=1, shape=[32, 32], hu_clip=False, resample=resample,
                        target_spacing=None if resample == 'none' else [1., 1., 1.5],
                        tiling=False, save_metadata=True)
                    random.seed(0)
                    with patch.object(preprocessing, 'sanity_ct', return_value=True), redirect_stdout(io.StringIO()):
                        preprocessing.main(prep)
                    args = SimpleNamespace(dataset='SEGTHOR', data_root=root/'data', dest=root/'training',
                        tiling=False, loss='ce', mode='full', epochs=1, workers=0, slices=slices,
                        gpu=False, seed=0, debug=False, optimizer='adam', scheduler='none',
                        class_weighting=weighting, original_grid_validation=True)
                    with patch.dict(main.datasets_params['SEGTHOR'], {'B':2}), \
                            patch.object(main, 'CrossEntropy', wraps=main.CrossEntropy) as ce, \
                            redirect_stdout(io.StringIO()):
                        main.runTraining(args)
                        if weighting == 'none':
                            self.assertIsNone(ce.call_args.kwargs['weights'])
                        else:
                            self.assertIsNotNone(ce.call_args.kwargs['weights'])
                        args.evaluate_checkpoint = args.dest/'bestweights.pt'
                        first = json.loads((args.dest/'original_epoch_000/metrics.json').read_text())
                        args.dest = root/'reloaded'
                        args.split = 'val'
                        report = main.evaluate_tiled_checkpoint(args)
                    self.assertEqual(report, first)
                    self.assertEqual(report['grid'], 'original_CT')
                    for path in args.dest.glob('*.nii.gz'):
                        image = nib.load(path)
                        self.assertEqual(image.shape, (33, 35, 3))
                        np.testing.assert_allclose(image.affine, np.diag([1., 1., 3., 1.]))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
