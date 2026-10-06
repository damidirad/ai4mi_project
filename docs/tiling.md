# Stage 3: optional tile extraction

Tiling cuts fixed rectangles out of each working X/Y slice. It does not resize
pixels or select tiles based on organs. The resampling mode remains independent:
`none`, `xy` and `xyz` all support tiling.

```bash
python slice_segthor.py --source_dir DATA --dest_dir OUT_TILES --retains 5 \
  --resample xyz --tiling --tile-size 256 256 --tile-stride 128 128
```

With `--tiling`, size defaults to 256×256 and stride to 128×128 (50% overlap).
For a 512×512 working slice this gives nine tiles at starts 0, 128 and 256 on
each axis. The existing whole-slice resize is bypassed. `--shape` remains a
full-slice option: non-default values are rejected with tiling.

Tile size and stride must be positive integer pairs, with stride no larger
than size. Smaller strides mean more overlap. Starts begin at zero and retain
a fixed stride; the final tile is never shifted or scaled. Padding is added
only at the high X/Y edges. Slices smaller than a tile produce one padded tile.
There is no Z padding or Z tiling: each processed Z slice uses the same XY layout.
This also provides the positions needed by the future 2.5D dataset loader.

## Output

```text
OUT_TILES/
  spacing.pkl
  train/
    metadata/Patient_01.json
    tiles/Patient_01/
      img/z0000_t0000.png
      gt/z0000_t0000.png
      valid/t0000.png
  val/
    ...
```

CT normalization happens on the working volume before extraction, using the
existing min/max or HU-window choice. Images are uint8, labels use class×63.
Padding is zero in normalized image space and background in labels. It is not
assigned a universal HU value. Valid masks use 255 for real pixels and 0 for
padding; one mask per XY tile is shared across Z. PNG rows correspond to the
first array axis X, matching the existing preprocessing convention.

Tile metadata is mandatory and automatically enabled by `--tiling`, even
without `--save_metadata`. Schema 2 stores tile size, stride, starts, padding
and filename conventions. `working` and `output` describe the same **unpadded
volume grid**, not individual tile shapes. Tile index `(i,j)` at start `(x,y)`
and slice `z` maps to working voxel `(x+i,y+j,z)`. Metadata validation rejects
missing/changed tile positions and inconsistent grids. Full-slice schema 1
continues to load as before.

For tiled output `spacing.pkl` contains effective working-grid spacing,
consistent with the affine; requested spacing is separately retained in JSON.
The full-slice path and its existing spacing behavior are unchanged.

Overlapping pixels are stored in multiple PNGs, so tiling uses more storage.
The writer refuses an existing patient tile directory. Metadata is published
after that patient's tile files finish; interrupted output is not resumable.
The existing CLI also requires a new destination directory for each run.

## Scope of this commit

This step prepares tiles, masks and positions only. Loading tiles into batches is now available in [stage 4](tiled_dataset.md); excluding padding and correcting overlap in the loss
comes in stage 5. Reconstruction is also a later stage. These tiles are stored
separately from the legacy `train/img` layout. The explicit tileloader supports
batch checks; tiled training awaits the loss and reconstruction stages. No model or loss has been changed.

Without `--tiling`, preprocessing continues to produce the existing full-slice
PNGs. Supplying tile settings without `--tiling` is an error.

## Checks

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```

Tests check full coverage, overlap, odd/non-square sizes, small images, masks,
invalid settings, corrupted metadata and actual NIfTI-to-tile PNG output for
all three resampling modes. Existing geometry/metadata/CLI regressions are also
run. Tiny integration scans bypass only the dataset-specific 512×512 and
minimum-depth sanity checks; resampling, normalization and image I/O run normally.
