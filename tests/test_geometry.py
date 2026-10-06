"""Physical-coordinate checks against the current production resampler."""
import unittest

import nibabel as nib
import numpy as np

from geometry import Grid, grid_after_resampling, grid_from_nifti, voxel_mapping
from slice_segthor import resample_voxels


class GeometryTests(unittest.TestCase):
    def setUp(self):
        # Rotated, flipped axes and a nonzero origin.
        self.affine = np.array([[0., -2., 0., 17.], [1., 0., 0., -23.],
                                [0., 0., -3., 41.], [0., 0., 0., 1.]])
        self.grid = Grid((5, 6, 7), self.affine)

    def test_known_world_point_and_owned_affine(self):
        np.testing.assert_array_equal(self.grid.voxel_to_world([2, 3, 4]), [11, -21, 29])
        self.affine[0, 3] = 100
        self.assertEqual(self.grid.origin[0], 17)
        self.assertFalse(self.grid.affine.flags.writeable)
        np.testing.assert_array_equal(self.grid.spacing, [1, 2, 3])

    def test_identity_and_singleton_preserve_geometry(self):
        for source in [self.grid, Grid((5, 6, 1), self.affine)]:
            result = grid_after_resampling(source, source.shape)
            np.testing.assert_array_equal(result.affine, source.affine)
            np.testing.assert_allclose(voxel_mapping(source, result), np.eye(4), atol=1e-14)

    def test_xy_and_xyz_match_sampled_world_coordinates(self):
        # The intensity is a linear function of physical position. Correct
        # resampling must reproduce that function at the tracked coordinates.
        indices = np.moveaxis(np.indices(self.grid.shape), 0, -1)
        world = self.grid.voxel_to_world(indices)
        ct = (world @ np.array([1., 2., 3.])).astype(np.float32)
        gt = (indices[..., 2] % 5).astype(np.uint8)
        for target, expected_shape in [((.6, 3.), (8, 4, 7)), ((.6, 3., 2.), (8, 4, 10))]:
            with self.subTest(target=target):
                output, labels = resample_voxels(ct, gt, self.grid.spacing, target)
                result = grid_after_resampling(self.grid, output.shape)
                self.assertEqual(result.shape, expected_shape)
                out_indices = np.moveaxis(np.indices(output.shape), 0, -1)
                expected = result.voxel_to_world(out_indices) @ np.array([1., 2., 3.])
                np.testing.assert_allclose(output, expected, atol=2e-5)
                mapping = voxel_mapping(self.grid, result)
                source_indices = out_indices @ mapping[:3, :3].T + mapping[:3, 3]
                np.testing.assert_allclose(self.grid.voxel_to_world(source_indices), result.voxel_to_world(out_indices), atol=1e-12)
                nearest = np.floor(source_indices + .5).astype(int)
                np.testing.assert_array_equal(labels, gt[tuple(np.moveaxis(nearest, -1, 0))])
                np.testing.assert_allclose(result.voxel_to_world(np.array(result.shape)-1), self.grid.voxel_to_world(np.array(self.grid.shape)-1))
                if len(target) == 2:
                    np.testing.assert_array_equal(result.affine[:, 2], self.grid.affine[:, 2])
                self.assertNotAlmostEqual(result.spacing[0], target[0])

    def test_nifti_units_and_shear(self):
        affine = self.affine.copy()
        affine[0, 2] = .4
        for unit, factor in [('mm', 1.), ('meter', 1000.), ('micron', .001)]:
            with self.subTest(unit=unit):
                image = nib.Nifti1Image(np.zeros((5, 6, 7)), affine)
                image.header.set_xyzt_units(unit)
                result = grid_from_nifti(image)
                np.testing.assert_allclose(result.voxel_to_world([1, 2, 3]), (affine @ [1, 2, 3, 1])[:3] * factor)
        image.header.set_xyzt_units('unknown')
        with self.assertRaisesRegex(ValueError, 'units'):
            grid_from_nifti(image)

    def test_invalid_geometry(self):
        for shape in [(0, 2, 3), (2, 3), (2., 3, 4), (True, 3, 4)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                Grid(shape, self.affine)
        for affine in [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan), np.diag([1, 0, 1, 1])]:
            with self.assertRaises(ValueError):
                Grid((2, 3, 4), affine)
        for shape in [(1, 6, 7), (5, 0, 7)]:
            with self.assertRaises(ValueError):
                grid_after_resampling(self.grid, shape)
        with self.assertRaises(ValueError):
            grid_after_resampling(Grid((1, 6, 7), self.affine), (2, 6, 7))


if __name__ == '__main__':
    unittest.main()
