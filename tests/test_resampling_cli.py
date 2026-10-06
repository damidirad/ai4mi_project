"""Regression coverage for the none/xy/xyz CLI and its preprocessing dispatch."""
import io
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import slice_segthor as pipeline


class ResamplingCliTests(unittest.TestCase):
    def parse(self, options):
        argv = ['slice_segthor.py', '--source_dir', 'source', '--dest_dir', 'output']
        with patch.object(sys, 'argv', argv + options), redirect_stdout(io.StringIO()):
            return pipeline.get_args()

    def test_modes_and_manual_targets(self):
        for options, mode, target in [
            ([], 'none', None),
            (['--resample', 'none'], 'none', None),
            (['--resample', 'xy'], 'xy', None),
            (['--resample', 'xyz'], 'xyz', None),
            (['--resample', 'xy', '--target_spacing', '.8', '1.2'], 'xy', [.8, 1.2]),
            (['--resample', 'xyz', '--target_spacing', '1', '1', '2'], 'xyz', [1, 1, 2]),
        ]:
            with self.subTest(options=options):
                args = self.parse(options)
                self.assertEqual(args.resample, mode)
                self.assertEqual(args.target_spacing, target)
                self.assertFalse(hasattr(args, 'spatial_normalize'))

    def test_old_and_invalid_arguments_fail_clearly(self):
        for options, message in [
            (['--resample'], 'expected one argument'),
            (['--spatial_normalize'], 'unrecognized arguments'),
            (['--spatial_nromalize'], 'unrecognized arguments'),
            (['--resample', 'invalid'], 'invalid choice'),
            (['--target_spacing', '1', '1'], 'requires --resample'),
            (['--resample', 'none', '--target_spacing', '1', '1'], 'requires --resample'),
            (['--resample', 'xy', '--target_spacing', '1', '1', '1'], 'requires 2 values'),
            (['--resample', 'xyz', '--target_spacing', '1', '1'], 'requires 3 values'),
        ]:
            with self.subTest(options=options), redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as error:
                    self.parse(options)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, output.getvalue())

    def test_invalid_spacing_values(self):
        for mode, dims in [('xy', 2), ('xyz', 3)]:
            for axis in range(dims):
                for value in ['0', '-1', 'nan', 'inf']:
                    target = ['1'] * dims
                    target[axis] = value
                    with self.subTest(mode=mode, axis=axis, value=value), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            self.parse(['--resample', mode, '--target_spacing', *target])
                        self.assertEqual(error.exception.code, 2)

    def test_main_dispatch_and_training_only_medians(self):
        spacings = {'Train1': (1., 2., 3.), 'Train2': (3., 4., 5.), 'Val': (99., 99., 99.)}
        for mode in ['none', 'xy', 'xyz']:
            for manual in [False, True] if mode != 'none' else [False]:
                with self.subTest(mode=mode, manual=manual), tempfile.TemporaryDirectory() as directory:
                    options = ['--resample', mode]
                    dims = 3 if mode == 'xyz' else 2
                    if manual:
                        options += ['--target_spacing'] + ['1'] * dims
                    args = self.parse(options)
                    args.source_dir = directory
                    args.dest_dir = str(Path(directory) / 'output')
                    expected = None if mode == 'none' else ((1.,) * dims if manual else (2., 3., 4.)[:dims])
                    def load(path):
                        return SimpleNamespace(header=SimpleNamespace(get_zooms=lambda: spacings[Path(path).parent.name]))
                    def slice_patient(pid, *, dest_path, target_spacing, **kwargs):
                        self.assertEqual(target_spacing, expected)
                        dest_path.mkdir(parents=True, exist_ok=True)
                        return spacings[pid]
                    with patch.object(pipeline, 'get_splits', return_value=(['Train1', 'Train2'], ['Val'], [])), patch.object(pipeline.nib, 'load', side_effect=load) as loader, patch.object(pipeline, 'slice_patient', side_effect=slice_patient) as slicer, redirect_stdout(io.StringIO()):
                        pipeline.main(args)
                    self.assertEqual(slicer.call_count, 3)
                    loaded = [Path(call.args[0]).parent.name for call in loader.call_args_list]
                    self.assertEqual(loaded, ['Train1', 'Train2'] if mode != 'none' and not manual else [])
                    with (Path(args.dest_dir) / 'spacing.pkl').open('rb') as file:
                        self.assertEqual(pickle.load(file), spacings)

    def test_xy_preserves_z_and_xyz_interpolates_z(self):
        ct = np.broadcast_to(np.array([0, 10, 20], dtype=np.int16), (2, 2, 3)).copy()
        gt = np.broadcast_to(np.array([0, 1, 2], dtype=np.uint8), ct.shape).copy()
        for target, depth in [((1., 1.), 3), ((1., 1., 1.), 6)]:
            with self.subTest(target=target):
                actual_ct, actual_gt = pipeline.resample_voxels(ct, gt, (1., 1., 2.), target)
                self.assertEqual(actual_ct.shape, (2, 2, depth))
                np.testing.assert_allclose(actual_ct[0, 0], np.linspace(0, 20, depth))
                np.testing.assert_array_equal(actual_gt[0, 0], [0, 1, 2] if depth == 3 else [0, 0, 1, 1, 2, 2])
                self.assertEqual(actual_ct.dtype, np.float32)
                self.assertEqual(actual_gt.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
