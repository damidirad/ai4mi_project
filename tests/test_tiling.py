import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image

from geometry import Grid
from preprocessing_metadata import build_metadata, load_metadata, validate_metadata
from tiling import TileLayout
import slice_segthor as pipeline


class TilingTests(unittest.TestCase):
    def test_coverage_alignment_and_padding(self):
        for shape, size, stride in [((512, 512), (256, 256), (128, 128)),
                ((7, 11), (4, 5), (3, 2)), ((2, 3), (4, 5), (2, 3)),
                ((8, 10), (4, 5), (4, 5)), ((1, 1), (1, 1), (1, 1))]:
            with self.subTest(shape=shape):
                layout = TileLayout(shape, size, stride)
                plane = np.arange(np.prod(shape)).reshape(shape)
                labels = plane % 5
                coverage = np.zeros(shape, dtype=int)
                for index, (x, y) in enumerate(layout.starts):
                    tile, valid = layout.extract(plane, index, fill=-1)
                    label, label_valid = layout.extract(labels, index)
                    nx, ny = min(size[0], shape[0]-x), min(size[1], shape[1]-y)
                    self.assertEqual(tile.shape, size)
                    np.testing.assert_array_equal(valid, label_valid)
                    np.testing.assert_array_equal(tile[:nx, :ny], plane[x:x+nx, y:y+ny])
                    np.testing.assert_array_equal(label[valid], tile[valid] % 5)
                    self.assertTrue(np.all(tile[~valid] == -1))
                    coverage[x:x+nx, y:y+ny] += 1
                    tile[:] = -999
                    self.assertNotIn(-999, plane)
                self.assertTrue(np.all(coverage >= 1))
                self.assertEqual(layout.starts[0], (0, 0))
                self.assertTrue(all(0 <= p < n for start in layout.starts for p, n in zip(start, shape)))
        standard = TileLayout((512, 512))
        self.assertEqual(len(standard.starts), 9)
        self.assertEqual(standard.padding_high, (0, 0))

    def test_invalid_layout_and_index(self):
        for shape, size, stride in [((0, 2), (4, 4), (2, 2)),
                ((3, 3), (0, 4), (2, 2)), ((3, 3), (4, 4), (5, 2)),
                ((3, 3), (4, 4), (0, 2)), ((3, 3), (4., 4), (2, 2))]:
            with self.assertRaises(ValueError):
                TileLayout(shape, size, stride)
        layout = TileLayout((3, 3), (4, 4), (2, 2))
        for index in [-1, 1]:
            with self.assertRaises(ValueError):
                layout.extract(np.zeros((3, 3)), index)
        with self.assertRaises(ValueError):
            layout.extract(np.zeros((2, 3)), 0)

    def test_cli_requires_explicit_tiling_and_valid_stride(self):
        base = ['slice_segthor.py', '--source_dir', 'source', '--dest_dir', 'output']
        for options in [['--tile-size', '4', '4'], ['--tiling', '--tile-stride', '0', '2'],
                        ['--tiling', '--tile-size', '4', '4', '--tile-stride', '5', '2'],
                        ['--tiling', '--shape', '64', '64']]:
            with self.subTest(options=options), patch.object(sys, 'argv', base + options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    pipeline.get_args()
                self.assertEqual(error.exception.code, 2)
        for mode in ('none', 'xy', 'xyz'):
            with patch.object(sys, 'argv', base + ['--tiling', '--resample', mode]), redirect_stdout(io.StringIO()):
                args = pipeline.get_args()
            self.assertTrue(args.tiling)
            self.assertEqual(args.resample, mode)

    def test_metadata_rejects_corrupted_tile_positions(self):
        layout = TileLayout((7, 11), (4, 5), (3, 2))
        value = build_metadata('Patient_01', 'train', 'source.nii.gz',
            Grid((7, 11, 3), np.eye(4)), (7, 11, 3), None, (7, 11), None,
            tile_layout=layout)
        self.assertEqual(value['schema_version'], 2)
        self.assertEqual(value['interpolation']['slice_resize'], 'none')
        for key in ('starts_xy', 'padding_high', 'size', 'stride'):
            bad = copy.deepcopy(value)
            bad['tiling'][key] = []
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_metadata(bad)

    def test_real_tiles_preserve_working_pixels_in_all_modes(self):
        for target in [None, (.5, 1.), (.5, 1., 1.5)]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                patient = root / 'source/train/Patient_01'
                patient.mkdir(parents=True)
                ct = np.arange(5*6*7, dtype=np.int16).reshape(5, 6, 7)
                gt = (ct % 5).astype(np.uint8)
                for filename, array in [('Patient_01.nii.gz', ct), ('GT.nii.gz', gt)]:
                    image = nib.Nifti1Image(array, np.diag([1., 1., 3., 1.]))
                    image.header.set_xyzt_units('mm')
                    nib.save(image, patient / filename)
                destination = root / 'out/train'
                with patch.object(pipeline, 'sanity_ct', return_value=True), patch.object(pipeline, 'resize_', side_effect=AssertionError('Tiles must not resize')):
                    spacing = pipeline.slice_patient('Patient_01', destination, root / 'source',
                        (256, 256), target_spacing=target, hu_window=(-400, 400),
                        tiling=True, tile_size=(4, 5), tile_stride=(3, 3))
                record = load_metadata(destination / 'metadata/Patient_01.json')
                working_ct, working_gt = (ct, gt) if target is None else pipeline.resample_voxels(ct, gt, (1., 1., 3.), target)
                expected = pipeline.hu_clipping(working_ct, -400, 400)
                self.assertEqual(record['working'], record['output'])
                np.testing.assert_allclose(spacing, np.linalg.norm(np.array(record['working']['affine_ras_mm'])[:3, :3], axis=0))
                self.assertFalse((destination / 'img').exists())
                folder = destination / 'tiles/Patient_01'
                starts = record['tiling']['starts_xy']
                self.assertEqual(len(list((folder / 'img').glob('*.png'))), len(starts)*working_ct.shape[2])
                self.assertEqual(len(list((folder / 'valid').glob('*.png'))), len(starts))
                for index, (x, y) in enumerate(starts):
                    nx, ny = min(4, expected.shape[0]-x), min(5, expected.shape[1]-y)
                    with Image.open(folder / 'valid' / f't{index:04d}.png') as file:
                        mask = np.array(file) == 255
                    self.assertEqual(int(mask.sum()), nx*ny)
                    for z in range(expected.shape[2]):
                        for kind, volume in [('img', expected), ('gt', working_gt*63)]:
                            with Image.open(folder / kind / f'z{z:04d}_t{index:04d}.png') as file:
                                actual = np.array(file)
                            np.testing.assert_array_equal(actual[:nx, :ny], volume[x:x+nx, y:y+ny, z])
                            self.assertTrue(np.all(actual[~mask] == 0))


if __name__ == '__main__':
    unittest.main()
