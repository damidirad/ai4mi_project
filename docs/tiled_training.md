# Optional tiled training

Use `--tiling` to train and evaluate ENet on tiles instead of full slices.

## Example workflow

Replace `DATA` with the raw dataset path. Use new output directories and keep
the dataset name `SEGTHOR` under the chosen data root, as shown below.

```bash
# Prepare tiles. Use xy or none for other resampling modes.
python slice_segthor.py --source_dir DATA --dest_dir data/tiled_xyz/SEGTHOR \
  --retains 5 --seed 0 --resample xyz --tiling \
  --tile-size 256 256 --tile-stride 128 128

# Train.
python main.py --dataset SEGTHOR --data_root data/tiled_xyz --dest results/tiled_xyz \
  --tiling --loss ce --mode full --slices 3 --epochs 20 \
  --optimizer adamw --scheduler cosine --seed 0 --workers 0 --gpu

# Evaluate the best checkpoint on validation data.
python main.py --dataset SEGTHOR --data_root data/tiled_xyz --dest results/tiled_xyz_eval \
  --tiling --evaluate-checkpoint results/tiled_xyz/bestweights.pt --slices 3 --gpu
```

## Important options and requirements

- Use the five-class SEGTHOR configuration. Training and validation must use
  matching preprocessing and separate patients.
- Only `--loss ce` is supported. Use `--mode full` for all classes or
  `--mode partial` for supervised class IDs `[0,1,3,4]`.
- `--slices 1` selects 2D; an odd value greater than 1 selects 2.5D.
  Evaluation must use the same slice count and preprocessing as training.
- Tile dimensions must be multiples of 8; use the default 256×256 tiles.
- `--gpu` enables GPU use; omit it to run on CPU.
- Output directories must not already exist. Training does not resume automatically.

See [resampling options](resampling.md) for `none`, `xy` and `xyz`.

## Main outputs

- `bestweights.pt`: checkpoint selected by validation Dice on the original CT grid.
- `history.json`: training loss, validation loss and Dice per epoch.
- `best_predictions/`: predictions and metrics from the best checkpoint.
- `summary.json`: final results and selected checkpoint scores.

## Predict unlabelled test scans

Place scans at `SOURCE/test/Patient_XX.nii.gz`. Reuse the metadata from the
training dataset associated with the checkpoint; do not add preprocessing
overrides. Use new destination directories.

```bash
python slice_segthor.py --source_dir SOURCE --dest_dir data/test_tiled/SEGTHOR \
  --test-from-metadata data/tiled_xyz/SEGTHOR/train/metadata
python main.py --dataset SEGTHOR --data_root data/test_tiled \
  --dest results/test_predictions --tiling --split test --slices 3 \
  --evaluate-checkpoint results/tiled_xyz/bestweights.pt --gpu
```

Use the same `--slices` value as training. Predictions are exported as
`Patient_XX.nii.gz` on the original CT grid. No scores are calculated without
ground-truth labels.
