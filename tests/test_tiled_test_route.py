"""Unlabelled preprocessing and checkpoint inference on original CT grids."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import main
import slice_segthor as pipeline
from geometry import Grid
from preprocessing_metadata import build_metadata, save_metadata, load_metadata
from tiling import TileLayout


class TiledTestRouteTests(unittest.TestCase):
    def test_cli_rejects_conflicting_settings(self):
        base = ['slice_segthor.py', '--source_dir', 'source', '--dest_dir', 'out',
                '--test-from-metadata', 'metadata']
        for flags in [['--hu_clip'], ['--resample', 'xyz'], ['--tile-size', '32', '32']]:
            with patch.object(sys, 'argv', base + flags), redirect_stdout(io.StringIO()), \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                pipeline.get_args()
            self.assertEqual(error.exception.code, 2)
        with patch.object(sys, 'argv', ['main.py', '--dest', 'out', '--split', 'test']), \
                patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            main.main()
        self.assertEqual(error.exception.code, 2)

    def test_unlabelled_inference_reuses_training_settings(self):
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for target, slices in [(None, 1), ([1., 1.], 3), ([1., 1., 1.5], 5)]:
                with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / 'source'
                    (source / 'test').mkdir(parents=True)
                    shape = (33, 35, 3)
                    affine = np.diag([1., 1., 3., 1.])
                    ct = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
                    image = nib.Nifti1Image(ct, affine)
                    image.header.set_xyzt_units('mm')
                    nib.save(image, source / 'test/Patient_41.nii.gz')
                    working = (*shape[:2], 6) if target and len(target) == 3 else shape
                    record = build_metadata('Patient_01', 'train', 'unused.nii.gz',
                        Grid(shape, affine), working, target, working[:2], (-400, 400),
                        tile_layout=TileLayout(working[:2], (32, 32), (24, 24)))
                    metadata = root / 'training_metadata'
                    save_metadata(metadata / 'Patient_01.json', record)
                    args = SimpleNamespace(source_dir=source, dest_dir=root/'data/SEGTHOR',
                                           test_from_metadata=metadata)
                    with patch.object(pipeline, 'sanity_ct', return_value=True), redirect_stdout(io.StringIO()):
                        pipeline.main(args)
                    self.assertFalse((args.dest_dir/'test/tiles/Patient_41/gt').exists())
                    actual = load_metadata(args.dest_dir/'test/metadata/Patient_41.json')
                    self.assertEqual(actual['intensity'], record['intensity'])
                    self.assertEqual(actual['requested_spacing_mm'], target)
                    self.assertEqual(actual['tiling'], record['tiling'])
                    model_class = main.ENet if slices == 1 else main.ENet_2_5d
                    model = model_class(1, 5, kernels=8, factor=2)
                    checkpoint = root/'weights.pt'
                    torch.save(model.state_dict(), checkpoint)
                    prediction_args = SimpleNamespace(dest=root/'predictions', debug=False,
                        dataset='SEGTHOR', data_root=root/'data', split='test', slices=slices,
                        gpu=False, evaluate_checkpoint=checkpoint)
                    with redirect_stdout(io.StringIO()):
                        report = main.evaluate_tiled_checkpoint(prediction_args)
                    self.assertIsNone(report['mean_foreground_dice'])
                    self.assertEqual(report['patients'], {'Patient_41': None})
                    prediction = nib.load(prediction_args.dest/'Patient_41.nii.gz')
                    self.assertEqual(prediction.shape, shape)
                    np.testing.assert_allclose(prediction.affine, affine)
                    self.assertTrue(np.isin(np.asarray(prediction.dataobj), range(5)).all())
                    with self.assertRaises(FileExistsError):
                        pipeline.main(args)
                    record['split'] = 'val'
                    save_metadata(metadata/'Patient_01.json', record)
                    with self.assertRaisesRegex(ValueError, 'training metadata'):
                        pipeline.main(args)
        finally:
            torch.set_num_threads(torch_threads)


if __name__ == '__main__':
    unittest.main()
