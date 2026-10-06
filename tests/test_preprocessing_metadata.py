import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from geometry import Grid
from preprocessing_metadata import (build_metadata, load_metadata, save_metadata,
                                    validate_metadata, resized_slice_grid)
import slice_segthor as pipeline


class MetadataTests(unittest.TestCase):
    def record(self):
        return build_metadata('Patient_01', 'train', 'source.nii.gz',
                              Grid((5, 6, 7), np.diag([1., 2., 3., 1.])),
                              (10, 6, 7), (.5, 2.), (4, 4), (-400, 400))

    def test_roundtrip(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata' / 'patient.json'
            save_metadata(path, record)
            self.assertEqual(load_metadata(path), record)
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_rejects_incomplete_and_inconsistent_records(self):
        for key in self.record():
            value = self.record()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_metadata(value)
        for key, replacement in [('schema_version', 2), ('resample', 'xyz'),
                                 ('requested_spacing_mm', [0, 2]),
                                 ('output_to_original', np.eye(4).tolist())]:
            value = self.record()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_metadata(value)
        value = self.record()
        value['working']['affine_ras_mm'][0][3] += 1
        with self.assertRaises(ValueError):
            validate_metadata(value)

    def test_resize_positions_match_actual_resize(self):
        grid = Grid((8, 6, 3), np.diag([2., 3., 4., 1.]))
        output = resized_slice_grid(grid, (4, 3))
        x, y = np.indices((8, 6))
        plane = (2*x + 3*y).astype(np.float32)
        actual = pipeline.resize_(plane, (4, 3))
        coords = np.stack([*np.meshgrid(np.arange(4), np.arange(3), indexing='ij'), np.zeros((4, 3))], axis=-1)
        world = output.voxel_to_world(coords)
        np.testing.assert_allclose(actual, world[..., 0] + world[..., 1])
        self.assertEqual(output.shape[2], 3)

    def test_real_nifti_and_png_output_unchanged_for_all_modes(self):
        for target, depth in [(None, 7), ((.5, 1.), 7), ((.5, 1., 1.5), 14)]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                patient = root / 'source' / 'train' / 'Patient_01'
                patient.mkdir(parents=True)
                ct = np.arange(5*6*7, dtype=np.int16).reshape(5, 6, 7)
                gt = (ct % 5).astype(np.uint8)
                for filename, array in [('Patient_01.nii.gz', ct), ('GT.nii.gz', gt)]:
                    image = nib.Nifti1Image(array, np.diag([1., 1., 3., 1.]))
                    image.header.set_xyzt_units('mm')
                    nib.save(image, patient / filename)
                spacings = []
                for enabled in (False, True):
                    destination = root / str(enabled) / 'train'
                    # Only bypass the dataset-specific 512x512/135-slice bounds.
                    with patch.object(pipeline, 'sanity_ct', return_value=True):
                        spacings.append(pipeline.slice_patient('Patient_01', destination,
                            root / 'source', (4, 4), target_spacing=target,
                            hu_window=(-400, 400), save_geometry=enabled))
                self.assertEqual(spacings[0], spacings[1])
                self.assertFalse((root / 'False/train/metadata').exists())
                record = load_metadata(root / 'True/train/metadata/Patient_01.json')
                self.assertEqual(record['output']['shape'], [4, 4, depth])
                self.assertEqual(record['resample'], 'none' if target is None else 'xy' if len(target) == 2 else 'xyz')
                for folder in ('img', 'gt'):
                    first = sorted((root / 'False/train' / folder).glob('*.png'))
                    second = sorted((root / 'True/train' / folder).glob('*.png'))
                    self.assertEqual(len(first), depth)
                    self.assertEqual([p.name for p in first], [p.name for p in second])
                    for a, b in zip(first, second):
                        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_rejects_misaligned_labels_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ct = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.int16), np.eye(4))
            affine = np.eye(4)
            affine[0, 3] = 1
            gt = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), affine)
            for image in (ct, gt):
                image.header.set_xyzt_units('mm')
            with patch.object(pipeline.nib, 'load', side_effect=[ct, gt]), patch.object(pipeline, 'sanity_ct', return_value=True):
                with self.assertRaisesRegex(ValueError, 'same physical grid'):
                    pipeline.slice_patient('Patient_01', root / 'out/train', root,
                                           (4, 4), save_geometry=True)
            self.assertFalse((root / 'out').exists())

    def test_hu_window_still_uses_training_foreground(self):
        from types import SimpleNamespace
        ct = SimpleNamespace(dataobj=np.array([0, 10, 20, 1000], dtype=np.int16))
        gt = SimpleNamespace(dataobj=np.array([0, 1, 2, 0], dtype=np.uint8))
        with patch.object(pipeline.nib, 'load', side_effect=[ct, gt]):
            np.testing.assert_allclose(pipeline.hu_window(['Patient_01'], Path('source')),
                                       [10.05, 19.95])

    def test_cli_metadata_is_explicit(self):
        import sys
        base = ['slice_segthor.py', '--source_dir', 'source', '--dest_dir', 'output']
        for options, enabled in [([], False), (['--save_metadata'], True)]:
            with patch.object(sys, 'argv', base + options), redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.get_args().save_metadata, enabled)


if __name__ == '__main__':
    unittest.main()
