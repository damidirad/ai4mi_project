# Stage 2: optional preprocessing metadata

Add `--save_metadata` to the existing preprocessing command:

```bash
python slice_segthor.py --source_dir DATA --dest_dir OUT --retains 5 --resample xyz --save_metadata
```

Without this option, no JSON is written and no new physical-grid checks run.
PNG contents and the existing `spacing.pkl` contract are unchanged.

For each completed patient, the split contains `metadata/Patient_XX.json`:

- Schema version, patient ID, split and absolute source CT path.
- Explicit RAS/mm/XYZ convention.
- Original, working (after voxel resampling) and output (after slice resize)
  shapes and full affines. Affine column lengths give effective spacing.
- Requested target spacing separately from the actual geometric spacing;
  original header spacing used by SciPy is also retained, avoiding rounding
  ambiguity between NIfTI header and affine precision.
- Resampling mode, interpolation convention, intensity method/HU window and
  label encoding (PNG class multiplier 63).
- Matrix mapping output indices back to original CT indices.

The slice resize uses cell-centred sampling: `input_index =
(output_index + 0.5) * input_length / output_length - 0.5` in X/Y. Its origin
shift is recorded; Z is untouched by this resize. Boundary fill and clipping
can affect intensity values but do not change these sampling coordinates.

`preprocessing_metadata.load_metadata` validates required fields, schema,
spacing, shapes, affine relationships and the mapping. Unknown versions and
inconsistent records fail. JSON is published after all patient PNGs have been
written, via a temporary file and rename. This is a per-patient record, not a
whole-dataset completion manifest or checksum audit.

Metadata export currently requires the source CT to declare mm units and its
header spacing to agree with its affine. Labels must have the same physical
grid. No registration or repair is performed. Resampling to/from a singleton
axis remains unsupported by geometry tracking. Existing datasets are not
backfilled. The selected NIfTI affine is stored; original qform/sform headers
are not archived separately, so exact header restoration will still need the
source CT. Training does not consume the JSON yet.

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_preprocessing_metadata test_geometry test_resampling_cli -v
```

Tests cover JSON round trips, missing/inconsistent fields, resize coordinates,
misaligned labels and real small NIfTI-to-PNG processing for none/xy/xyz.

## Stage 3 extension

Full-slice records remain schema 1. Tiled output uses schema 2 with mandatory
tile positions, padding and storage conventions; its output grid equals the
unpadded working grid. `--tiling` always enables metadata, independently of
`--save_metadata`. See [tile extraction](tiling.md).
