# Voxel resampling CLI

`slice_segthor.py` accepts `--resample none|xy|xyz`, defaulting to `none`.

| Mode | Behavior | Optional `--target_spacing` |
| --- | --- | --- |
| `none` | No voxel resampling | Not accepted |
| `xy` | Resample X/Y, preserve Z | DX DY |
| `xyz` | Resample X/Y/Z | DX DY DZ |

Without an override, each selected axis uses its training-set median spacing.
Explicit spacings must be finite and positive. Medians need not be isotropic.
The subsequent full-slice resize (default 256×256) still runs in all modes.
These options configure preprocessing, not the network architecture.

```bash
python slice_segthor.py --source_dir DATA --dest_dir OUT --retains 5 --resample none
python slice_segthor.py --source_dir DATA --dest_dir OUT_XY --retains 5 --resample xy
python slice_segthor.py --source_dir DATA --dest_dir OUT_XYZ --retains 5 --resample xyz --target_spacing 1 1 1
```

Replace bare `--resample` with `--resample xy`, and replace
`--spatial_normalize` with `--resample xyz`. Both old forms now fail with
argparse exit code 2. Calls without either option keep resampling off.
Python callers constructing a Namespace for `main` must use the string mode.
Direct `slice_patient` calls use `target_spacing=None`, a pair, or a triple.

Interpolation and full-slice resizing are unchanged. `stitch.py` still assumes
the original Z slice count; XYZ reconstruction needs separate work.

Run current CLI and dispatch regression tests:

```bash
ai4mi/bin/python -m unittest discover -s tests -p 'test_resampling_cli.py' -v
```

The historical `test_simple_resampling.py` suite already had failures before
this migration (removed helpers and changed normalization/spacing contracts).

## Optional geometry metadata

Add `--save_metadata` to write per-patient JSON alongside the PNGs.
See [metadata format and usage](preprocessing_metadata.md). The option defaults
to off and does not alter PNG values or `spacing.pkl`.
