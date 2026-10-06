import io
import json
import random
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
import slice_segthor as preprocessing
from train_tiled import run_tiled_training


class TiledTrainingTests(unittest.TestCase):
    def prepare(self, root, mode):
        source = root / 'source'
        for pid in ['Patient_01', 'Patient_02']:
            folder = source / 'train' / pid
            folder.mkdir(parents=True)
            x, y, z = np.indices((33, 35, 3))
            ct = (x*10 + y*3 + z*20 - 200).astype(np.int16)
            gt = ((x+y+z)%5).astype(np.uint8)
            for name, array in [(f'{pid}.nii.gz', ct), ('GT.nii.gz', gt)]:
                image = nib.Nifti1Image(array, np.diag([1., 1., 3., 1.]))
                image.header.set_xyzt_units('mm')
                nib.save(image, folder / name)
        target = {'none': None, 'xy': [.8, .8], 'xyz': [.8, .8, 1.5]}[mode]
        args = SimpleNamespace(source_dir=source, dest_dir=root/'data/SEGTHOR', retains=1,
            fold=0, process=1, shape=[256, 256], hu_clip=False, resample=mode,
            target_spacing=target, tiling=True, tile_size=[32, 32], tile_stride=[24, 24], save_metadata=False)
        random.seed(0)
        with patch.object(preprocessing, 'sanity_ct', return_value=True), redirect_stdout(io.StringIO()):
            preprocessing.main(args)

    def test_real_enet_training_and_best_reload_all_resampling_modes(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for mode, slices, optimizer, scheduler in [('none', 1, 'adam', 'none'),
                    ('xy', 1, 'adamw', 'cosine'), ('xyz', 3, 'adamw', 'cosine')]:
                with self.subTest(mode=mode, slices=slices), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.prepare(root, mode)
                    args = SimpleNamespace(dataset='SEGTHOR', data_root=root/'data', dest=root/'results',
                        tiling=True, loss='ce', mode='partial' if mode == 'xy' else 'full', epochs=2, workers=0, slices=slices,
                        gpu=False, seed=0, debug=False, optimizer=optimizer, scheduler=scheduler)
                    with patch.dict(main.datasets_params['SEGTHOR'], {'B': 2}), redirect_stdout(io.StringIO()):
                        history = main.runTraining(args)
                    self.assertEqual(len(history), 2)
                    self.assertTrue(all(np.isfinite(row['train_ce']) and np.isfinite(row['val_ce']) for row in history))
                    summary = json.loads((args.dest/'summary.json').read_text())
                    self.assertTrue(summary['complete'])
                    self.assertAlmostEqual(summary['best_validation_dice'], summary['reloaded_best_dice'])
                    if scheduler == 'cosine':
                        self.assertLess(history[1]['lr'], history[0]['lr'])
                    state = torch.load(args.dest/'last.pt', weights_only=True, map_location='cpu')
                    self.assertEqual(state['epoch'], 1)
                    self.assertTrue(state['optimizer']['state'])
                    for value in state['state_dict'].values():
                        self.assertTrue(torch.isfinite(value).all())
                    outputs = list((args.dest/'best_predictions').glob('*.nii.gz'))
                    self.assertEqual(len(outputs), 1)
                    prediction = nib.load(outputs[0])
                    self.assertEqual(prediction.shape, (33, 35, 3))
                    np.testing.assert_allclose(prediction.affine, np.diag([1., 1., 3., 1.]))
                    report = json.loads((args.dest/'best_predictions/metrics.json').read_text())
                    self.assertEqual(report['grid'], 'original_CT')
                    with self.assertRaises(FileExistsError):
                        run_tiled_training(args)
        finally:
            torch.set_num_threads(previous_threads)

    def test_unsupported_loss_fails_before_io(self):
        with self.assertRaisesRegex(ValueError, 'only ce'):
            main.runTraining(SimpleNamespace(tiling=True, loss='dice'))

    def test_no_tiling_keeps_legacy_dispatch(self):
        with patch.object(main, 'setup', side_effect=RuntimeError('legacy route')) as setup:
            with self.assertRaisesRegex(RuntimeError, 'legacy route'):
                main.runTraining(SimpleNamespace(tiling=False, dataset='SEGTHOR', mode='full'))
            setup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
