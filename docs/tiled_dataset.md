# Stage 4: loading tiles in 2D and 2.5D

`dataset.build_dataset(..., tiling=True)` selects `TiledDataset` explicitly.
Without tiling it selects the existing `SliceDataset`. A layout mismatch gives
an error rather than silently loading zero samples.

The loader reads stage-3 metadata and checks patient identity, split, tile size,
preprocessing configuration and the full image/label/mask filename inventory.
Unexpected or missing tiles and patient folders without metadata are rejected.
Each loaded PNG must be uint8 with the expected dimensions. Label encoding and
mask contents are also checked when accessed; reading the inventory does not
check every PNG's pixels in advance. Test datasets do not require labels.

A sample contains `images`, central-slice `gts` (except for test), a unique
`stems` string, `patient_id`, `z`, `tile_index`, `tile_start`, and boolean
`valid_mask`. Batched shapes are images `[B,S,X,Y]`, labels `[B,5,X,Y]`, masks
`[B,X,Y]`. This uses the current SegTHOR five-class encoding and transforms.

For 2.5D, an odd number of slices is required. Each neighbour has the same
patient and tile index; only Z changes. Out-of-range neighbours repeat the
nearest end slice. Debug mode limits sample centres to ten but retains all
neighbour data, so it cannot change the input of a retained sample.

## Inspect batches through the existing CLI

For a tiled dataset at `data/SEGTHOR`:

```bash
python main.py --dataset SEGTHOR --data_root data --dest results/check \
  --tiling --check-data --slices 3 --workers 0
```

`--check-data` loads one batch from train and val, prints sizes and compares
spatial/preprocessing configurations. It creates no network, optimizer or
results directory. The existing parser still requires `--dest`. Use `--slices
1` for 2D. No architecture change is introduced.

Tiled training is deliberately not enabled in this stage: `--tiling` without
`--check-data` fails before model/optimizer setup. The existing loss ignores
padding/overlap and the existing metrics assume full slices. Those need the
planned stages 5–7 before training is correct. The dataset already provides
masks and coordinates for those stages. No new sampling/weighting strategy is
introduced here. Ordinary full-slice training retains its existing route.

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_tiled_dataset test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```
