# Introduction

This document describes the changes made on the `feature/simple-resampling` branch. It is organized by commit to document what changed, why each change was introduced, and how it was verified.

The goal of this branch is to add an optional voxel-resampling step while keeping the rest of the preprocessing pipeline unchanged as much as possible. Resampling is limited to the X/Y dimensions; the Z dimension is preserved. The existing HU-windowing and full-slice resize to 256×256 are intentionally retained.

This means that resampling is evaluated as an additional preprocessing step within the existing pipeline, rather than as part of a larger redesign of the input or training procedure.

## Commit 1: Baseline regression tests

`tests/test_slice_segthor.py` checks the existing preprocessing behavior
without resampling: HU-windowing, resize to 256×256, slice count and
naming, label encoding, and original spacing metadata in `spacing.pkl`.
It also checks that the default call and explicit `resample=False`
produce the same expected output.

These tests provide a reference for checking that later resampling
changes preserve the baseline.

All three tests passed (OK). This confirms that the current pipeline matches the baseline expectations covered by the tests. These tests can now help detect unintended changes to that behavior when we update resampling.

