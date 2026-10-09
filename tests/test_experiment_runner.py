"""Screening plans, dataset provenance and isolated run lifecycle."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from geometry import Grid
from preprocessing_metadata import build_metadata, save_metadata
import run_experiments as runner


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_config(runner.ROOT/'experiments/screening.json')

    def args(self, root, **overrides):
        args = dict(config=runner.ROOT/'experiments/screening.json', run_id='M1',
                    run_index=None, data_root=root/'data', source_dir=root/'raw',
                    results_root=root/'results', python=Path('/usr/bin/python3'),
                    workers=4, attempt=1, dry_run=False)
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_all_supplied_plans(self):
        rows = self.config['configurations']
        expected = ['SEGTHOR_hu']*4 + ['full_hu_xy/SEGTHOR', 'full_hu_xyz/SEGTHOR',
                                     'tiled_hu_xy/SEGTHOR', 'full_hu_xy/SEGTHOR']
        self.assertEqual(len(rows), 8)
        for index, (row, suffix) in enumerate(zip(rows, expected)):
            dataset = runner.dataset_path(row, Path('/data'))
            self.assertEqual(dataset, Path('/data')/suffix)
            command = runner.training_command(self.config, row, dataset, Path('/out'), Path('/python'), 4)
            self.assertEqual(command[command.index('--epochs')+1], '3' if index == 6 else '20')
            self.assertEqual(command[command.index('--loss')+1], 'ce' if index == 6 else row['loss'])
            self.assertEqual('--tiling' in command, index == 6)
            self.assertEqual(command[command.index('--dataset')+1], dataset.name)
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(runner.subprocess, 'run') as process:
                root = Path(d)
                runner.run(self.args(root, run_index=index, run_id=None, dry_run=True))
                self.assertFalse((root/'results').exists())
                process.assert_not_called()

    def test_commands_are_accepted_by_training_cli(self):
        import main
        for row in self.config['configurations']:
            command = runner.training_command(self.config, row, runner.dataset_path(row, Path('/data')),
                                              Path('/out'), Path('/python'), 4)
            with patch.object(sys, 'argv', command[2:]), patch.object(main, 'runTraining') as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                main.main()
                train.assert_called_once()
                args = train.call_args.args[0]
                self.assertEqual(args.epochs, row['epochs'])
                self.assertEqual(args.tiling, row['tiling'])

    def test_invalid_configuration_rejected(self):
        for change in [{'loss':'CE_DICE', 'tiling':True}, {'loss':'CE_TILED'},
                       {'slices':2}, {'epochs':0}, {'augmentation':'false'}, {'id':'../bad'}]:
            config = copy.deepcopy(self.config)
            config['configurations'][0].update(change)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'config.json'
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    runner.load_config(path)

    def test_augmentation_blocked_before_launch_or_output(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner.subprocess, 'run') as process:
            root = Path(d)
            with self.assertRaisesRegex(ValueError, 'Augmentation'):
                runner.run(self.args(root, run_id='M7'))
            self.assertFalse((root/'results').exists())
            process.assert_not_called()

    def test_dataset_settings_split_and_inventory(self):
        config = copy.deepcopy(self.config)
        config['validation_patients'] = 1
        row = config['configurations'][0]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, dataset = root/'raw', root/'data'
            for split, pid in [('val', 'Patient_01'), ('train', 'Patient_02')]:
                ct = source/'train'/pid/f'{pid}.nii.gz'
                ct.parent.mkdir(parents=True)
                ct.touch()
                ct.with_name('GT.nii.gz').touch()
                record = build_metadata(pid, split, ct, Grid((4, 4, 2), np.eye(4)),
                                        (4, 4, 2), None, (256, 256), (-400, 400))
                save_metadata(dataset/split/'metadata'/f'{pid}.json', record)
                for kind in ('img', 'gt'):
                    folder = dataset/split/kind
                    folder.mkdir()
                    for z in range(2):
                        (folder/f'{pid}_{z:04d}.png').touch()
            (dataset/'spacing.pkl').touch()
            data = runner.check_dataset(config, row, dataset, source)
            self.assertEqual(data['inventory']['val']['patients'], ['Patient_01'])
            with self.assertRaisesRegex(ValueError, 'HU'):
                runner.check_dataset(config, dict(row, hu_clipping=False), dataset, source)
            with self.assertRaisesRegex(ValueError, 'patients'):
                runner.check_dataset(dict(config, fold=1), row, dataset, source)
            (dataset/'train/gt/Patient_02_0001.png').rename(dataset/'train/gt/wrong.png')
            with self.assertRaisesRegex(ValueError, 'Missing images or labels'):
                runner.check_dataset(config, row, dataset, source)

    def test_completed_run_skips_and_failed_run_requires_new_attempt(self):
        metrics = {'best_epoch':0, 'mean_foreground_dice':0.5, 'evaluation_grid':'prepared_volume'}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'check_dataset', return_value={}), \
                patch.object(runner, 'summarize', return_value=metrics), \
                patch.object(runner.subprocess, 'run') as process:
            root = Path(d)
            args = self.args(root)
            runner.run(args)
            self.assertEqual(process.call_count, 2)
            manifest = root/'results/screening/M1/seed_0/attempt_001/runner.json'
            self.assertEqual(json.loads(manifest.read_text())['status'], 'completed')
            runner.run(args)
            self.assertEqual(process.call_count, 2)
            with self.assertRaises(FileExistsError):
                runner.run(self.args(root, workers=2))
            process.side_effect = [None, subprocess.CalledProcessError(1, ['main.py'])]
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run(self.args(root, attempt=2))
            failed = manifest.parent.parent/'attempt_002/runner.json'
            self.assertEqual(json.loads(failed.read_text())['status'], 'failed')
            with self.assertRaises(FileExistsError):
                runner.run(self.args(root, attempt=2))

    def test_metric_summary_requires_selected_checkpoint(self):
        row = self.config['configurations'][0]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                runner.summarize(root, row)
            (root/'bestweights.pt').touch()
            (root/'best_epoch.txt').write_text('Improved 3D dice at epoch 2: 0.4->0.5 DSC')
            (root/'iter002').mkdir()
            np.save(root/'iter002/dice_3d_val.npy', np.array([[1., .2, .4, .6, .8]]))
            for name in ('loss_tra.npy', 'loss_val.npy', 'lr.npy'):
                np.save(root/name, np.zeros(20))
            summary = runner.summarize(root, row)
            self.assertEqual(summary['best_epoch'], 2)
            self.assertAlmostEqual(summary['mean_foreground_dice'], .5)
            self.assertEqual(summary['evaluation_grid'], 'prepared_volume')


if __name__ == '__main__':
    unittest.main()
