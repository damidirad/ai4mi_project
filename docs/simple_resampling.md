# Introduction

This document describes the changes made on the `feature/simple-resampling` branch. It is organized by commit to document what changed, why each change was introduced, and how it was verified.

The goal of this branch is to improve Amanda’s existing optional voxel-resampling step while keeping the rest of the preprocessing pipeline unchanged as much as possible. Resampling is limited to the X/Y dimensions; the Z dimension is preserved. The existing HU-windowing and full-slice resize to 256×256 are intentionally retained.

This means that resampling is evaluated as an additional preprocessing step within the existing pipeline, rather than as part of a larger redesign of the input or training procedure.

The later full-slice resize to 256×256 can reduce some of the physical scale standardization introduced by resampling. This is an accepted limitation of this experiment.

## Commit 1: Baseline regression tests

`tests/test_slice_segthor.py` checks the existing preprocessing behavior
without resampling: HU-windowing, resize to 256×256, slice count and
naming, label encoding, and original spacing metadata in `spacing.pkl`.
It also checks that the default call and explicit `resample=False`
produce the same expected output.

These tests provide a reference for checking that later resampling
changes preserve the baseline.

All three tests passed (OK). This confirms that the current pipeline matches the baseline expectations covered by the tests. These tests can now help detect unintended changes to that behavior when we update resampling.

## Commit 2: Derive and validate X/Y target spacing

With `--resample`, the target is now the median spacing per X/Y axis of the training split. Validation and test data do not contribute. Empty training sets and non-finite or non-positive source spacings are rejected.

`--target_spacing DX DY` remains available as a validated manual override. It now accepts two values instead of three. The log identifies whether the target comes from the training-set median or a manual override. The normal experiment uses the training-set median.

Each patient’s original Z spacing is passed to the existing resampling function as its Z target, keeping the Z zoom factor at 1. Without `--resample`, target-spacing calculation is skipped and the baseline path is retained. Interpolation improvements are reserved for the next step.

Seven new tests cover spacing validation, empty training sets, training-only target estimation, a shared target for training and validation, and manual override handling and logging. All 10 tests passed (`OK`), including the three baseline regression tests. This confirms the target-spacing behavior covered by these tests and preserves the tested baseline expectations.

## Commit 3: Align CT and label resampling in X/Y

This step updates existing `resample_voxels` function. Before interpolation, it checks that CT and labels are both 3D, have identical shapes, and contain no empty spatial dimensions. Source spacings and the two-value X/Y target are validated as finite and positive.

### Shared geometry and Z preservation

Both arrays use the same zoom factors:

```python
zoom_factors = (dx / target_dx, dy / target_dy, 1.0)
```

A smaller target spacing increases the number of samples along that axis; a larger target spacing reduces it. Identical input shapes, zoom factors, and coordinate settings place CT and labels on the same output grid. The Z factor is fixed at 1, preserving the original slice count and order without interpolation between slices. X/Y values within each slice can still change.

`slice_patient` now passes the X/Y target directly. This replaces step 2’s temporary approach of appending each patient’s Z spacing to the target. The slice loop continues to use the original Z count, and `spacing.pkl` retains its original source-spacing meaning.

### Interpolation settings

- **CT:** conversion to `float32` happens before `zoom`, followed by linear interpolation (`order=1`). Previously, interpolating an integer CT array produced integer output, losing fractional interpolated values. The conversion preserves those values until the existing HU-windowing and uint8 conversion.
- **Labels:** nearest-neighbor interpolation (`order=0`) selects existing class values instead of blending them. The label dtype is preserved.
- **Coordinates and boundaries:** both calls explicitly use `grid_mode=False`, retaining the existing voxel-center coordinate convention, and `mode="constant"` with zero fill outside the input. These settings are shared so the interpolation methods sample the same geometry.
- **Prefiltering:** `prefilter=False` is explicit because interpolation orders 0 and 1 do not require a spline prefilter.

Resampling remains before HU-windowing. The existing HU range, full-slice resize to 256×256, and label encoding remain unchanged. The later resize can reduce the physical scale standardization introduced here, as accepted for this experiment.

### Verification

Six new tests use small synthetic volumes with independently calculated expectations. They check fractional CT output and `float32` dtype; shared coordinates while enlarging X and reducing Y; discrete label values; separate, ordered Z slices for different source Z spacings; unchanged values and inputs for identity resampling; and rejection of invalid shapes or spacings.

All 16 tests passed (`OK`), including the baseline and target-spacing tests. This verifies the covered numerical resampling behavior and baseline expectations. 
