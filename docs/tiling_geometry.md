# Stage 1: geometry for the future optional tiling pipeline

`geometry.py` was introduced as a standalone building block in stage 1.
Stage 2 now uses it for optional metadata export; see
[preprocessing_metadata.md](preprocessing_metadata.md). Tile extraction and
training integration remain later stages. Existing `spacing.pkl` behavior is unchanged.

A `Grid` records the array shape and a full affine: a matrix mapping voxel-centre
indices to physical positions in NIfTI RAS millimetres. Axis columns describe
orientation and voxel steps; the translation describes the first voxel centre.
Rotations, flips and shear are preserved. X/Y/Z refer to array axes.

`grid_from_nifti` reads the selected NIfTI affine and converts declared spatial
units to mm. Missing units are rejected rather than guessed. This module does
not resolve conflicting qform/sform values, check CT/label alignment or perform
registration; these concerns remain for integration.

## Match the current interpolation

The current `resample_voxels` uses SciPy `zoom(..., grid_mode=False)`:

- Output length: `round(input_length * source_spacing / requested_spacing)`.
- Output index to input index: `j * (input_length - 1) / (output_length - 1)`.
- The first and last voxel centres stay fixed on each non-singleton axis.

For example, 5 voxels at 1 mm requested at 0.6 mm produce 8 voxels. Their
actual spacing is `4 / 7` mm, not 0.6 mm. The current full-slice resize is a
separate operation with its own geometry and is not described by this helper.

Call `grid_after_resampling(source_grid, actual_output.shape)` after resampling.
It derives the output affine from actual dimensions; it does not change pixels
or adopt the older branch's SimpleITK centre/extent convention. An unchanged Z
axis retains its affine column exactly. Changing from/to a singleton axis is
rejected because it lacks an invertible endpoint mapping; unchanged singleton
axes are supported.

`voxel_mapping(source, destination)` maps destination indices to source indices.
This supports later reconstruction; it does not imply that interpolation is
lossless. Metadata must eventually retain requested spacing separately from
this effective geometric spacing.

## Verification

```bash
ai4mi/bin/python -m unittest discover -s tests -p 'test_geometry.py' -v
```

Tests compare physical-coordinate ramps against the current production
resampler, including rounded dimensions, XY/XYZ modes, discrete labels,
rotations, flips, nonzero origins, unit conversion, shear and invalid geometry.
