# Resampling during preprocessing

Use `slice_segthor.py --resample none|xy|xyz` to choose voxel resampling during
preprocessing. The default is `none`.

| Mode | Behavior | Optional `--target_spacing` (mm) |
| --- | --- | --- |
| `none` | No voxel resampling | Not accepted |
| `xy` | Resample X/Y, preserve Z | DX DY |
| `xyz` | Resample X/Y/Z | DX DY DZ |

Without `--target_spacing`, spacing is determined from the training-set median
for each selected axis. Explicit values must be positive and finite.

## Examples

Replace `DATA` with the raw dataset path. Use a new destination for each run.

```bash
python slice_segthor.py --source_dir DATA --dest_dir OUT --retains 5 --resample none
python slice_segthor.py --source_dir DATA --dest_dir OUT_XY --retains 5 --resample xy
python slice_segthor.py --source_dir DATA --dest_dir OUT_XYZ --retains 5 --resample xyz --target_spacing 1 1 1
```

## Output options

- By default, full slices are resized to 256×256 after resampling.
- Add `--save_metadata` to save per-patient geometry JSON alongside the PNGs.
- Add `--tiling` to extract tiles instead of resizing full slices. Tile size
  defaults to `--tile-size 256 256`, with `--tile-stride 128 128`. Metadata is
  always saved for tiles. See [tiled training](tiled_training.md) for the workflow.

`stitch.py` assumes the original Z slice count. For XYZ-resampled tiles, use the
evaluation workflow in the tiled training guide to restore the original CT grid.
