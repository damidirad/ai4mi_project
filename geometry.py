"""
Geometry building blocks for tiling and optional preprocessing metadata.

Array axes are voxel X/Y/Z, not necessarily anatomical axes. World coordinates
are NIfTI RAS in millimetres. Full affines preserve rotations, flips and shear.
This module describes scipy.ndimage.zoom(grid_mode=False), NOT the later
skimage full-slice resize or the old tiling branch's SimpleITK convention.
"""
from dataclasses import dataclass

import numpy as np


def _shape3(shape):
    values = tuple(shape)
    if len(values) != 3 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
        for n in values
    ):
        raise ValueError('Shape must contain three positive integers')
    return tuple(int(n) for n in values)


@dataclass(frozen=True, eq=False)
class Grid:
    """Voxel-centre indices -> RAS millimetres via a homogeneous affine."""

    shape: tuple[int, int, int]
    affine: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'shape', _shape3(self.shape))
        affine = np.array(self.affine, dtype=np.float64, copy=True)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError('Affine must be a finite 4x4 matrix')
        if not np.array_equal(affine[3], [0, 0, 0, 1]):
            raise ValueError('Affine must have homogeneous last row [0, 0, 0, 1]')
        if np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError('Affine must be invertible')
        affine.setflags(write=False)
        object.__setattr__(self, 'affine', affine)

    @property
    def spacing(self):
        return tuple(np.linalg.norm(self.affine[:3, :3], axis=0))

    @property
    def origin(self):
        return tuple(self.affine[:3, 3])

    @property
    def direction(self):
        """Unit axis columns; may be non-orthogonal for sheared NIfTI grids."""
        return self.affine[:3, :3] / np.asarray(self.spacing)

    def voxel_to_world(self, indices):
        indices = np.asarray(indices, dtype=np.float64)
        if indices.ndim == 0 or indices.shape[-1] != 3 or not np.isfinite(indices).all():
            raise ValueError('Voxel coordinates must be finite with final dimension 3')
        return indices @ self.affine[:3, :3].T + self.affine[:3, 3]


def grid_from_nifti(image):
    """Read geometry without loading pixels; reject unspecified spatial units."""
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(
        image.header.get_xyzt_units()[0]
    )
    if scale is None:
        raise ValueError('NIfTI spatial units must be mm, meter or micron')
    if image.affine is None:
        raise ValueError('NIfTI image must have an affine')
    affine = np.array(image.affine, dtype=np.float64, copy=True)
    affine[:3, :] *= scale
    return Grid(image.shape, affine)


def voxel_mapping(source: Grid, destination: Grid):
    """Map destination voxel indices to source indices (for later reconstruction)."""
    return np.linalg.solve(source.affine, destination.affine)


def grid_after_resampling(source: Grid, output_shape):
    """
    Track the actual output of the existing grid_mode=False resampler.

    SciPy rounds input_size * requested_zoom to obtain the output size, then
    samples using (input_size - 1) / (output_size - 1). Therefore requested
    target spacing is generally not the effective spacing. Pass the actual
    returned array shape; do not infer it from rounded spacing metadata.

    Changing an axis from/to one voxel has no bijective endpoint geometry.
    Reject it here instead of inventing an invertible physical grid. An
    unchanged singleton axis retains its original affine column.
    """
    shape = _shape3(output_shape)
    scale = np.ones(3)
    for axis, (before, after) in enumerate(zip(source.shape, shape)):
        if before == after:
            continue
        if before == 1 or after == 1:
            raise ValueError('Cannot track resampling from/to a singleton axis')
        scale[axis] = (before - 1) / (after - 1)
    mapping = np.diag([*scale, 1.])
    return Grid(shape, source.affine @ mapping)
