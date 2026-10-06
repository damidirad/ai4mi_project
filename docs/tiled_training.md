# Stage 8: complete optional tiled training

The eight-stage implementation is connected through `main.py --tiling`.
Without this flag, `runTraining` keeps the historical full-slice path and losses.
With it, `train_tiled.py` runs corrected CE, whole-volume reconstruction,
original-CT validation and checkpoint selection. The ENet architectures are
unchanged. `--slices 1` selects 2D; odd `--slices >1` selects the existing 2.5D ENet.

## Example workflow

Keep the dataset's configured name (e.g. SEGTHOR) under a separate parent so
existing preprocessing is not overwritten. All output directories must be new.
Replace DATA with the raw dataset path.

```bash
# Prepare tiles with XYZ resampling. Use xy or none for the other modes.
python slice_segthor.py --source_dir DATA --dest_dir data/tiled_xyz/SEGTHOR \
  --retains 5 --seed 0 --resample xyz --tiling \
  --tile-size 256 256 --tile-stride 128 128

# Optional diagnostics. The parser requires --dest, but checks do not write it.
python main.py --dataset SEGTHOR --data_root data/tiled_xyz --dest results/check \
  --tiling --check-data --check-loss --loss ce --slices 3 --workers 0

# Explicit training example; this documentation does not execute a real run.
python main.py --dataset SEGTHOR --data_root data/tiled_xyz --dest results/tiled_xyz \
  --tiling --loss ce --mode full --slices 3 --epochs 20 \
  --optimizer adamw --scheduler cosine --seed 0 --workers 0 --gpu

# Evaluate the selected state_dict in a new directory.
python main.py --dataset SEGTHOR --data_root data/tiled_xyz --dest results/tiled_xyz_eval \
  --tiling --evaluate-checkpoint results/tiled_xyz/bestweights.pt --slices 3 --gpu
```

## Training behavior

- Training and validation must have matching preprocessing configurations and
  disjoint patient IDs. The five-class SegTHOR configuration is required.
- Tiles use uniform shuffled sampling. Class counts exclude padding and correct
  overlap. Class weights are derived only from training; validation reuses them.
- Only `--loss ce` is supported for tiles. `--mode partial` retains the existing
  SEGTHOR supervised IDs `[0,1,3,4]`. Other tiled loss choices fail before I/O.
- ENet tile dimensions must be multiples of 8; 256×256 is the default and normal
  choice. Very small tiles or batches can still violate architecture/BatchNorm
  constraints. The integration smoke uses 32×32 tiles and batch size 2.
- Adam/AdamW and optional cosine scheduling follow the existing parameter choices.
  Epoch loss is weighted by batch sample counts, including a short last batch.
- Every epoch reconstructs complete validation patients, restores the original
  grid and evaluates against original GT. It also records tiled validation CE.
- Best weights are selected by mean original-grid Dice over patients and four
  foreground classes; ties keep the earlier checkpoint. The best weights are
  reloaded and evaluated again at the end.
- `--debug` limits training centres to ten but keeps full validation. It is not
  a tiny validation run. Diagnostic subsets are not representative model scores.

One epoch over tiles has more optimizer steps than an epoch over full slices.
Identical epoch counts are not matched experimental training budgets. The
current objective weights working voxels before class weights, not patients
or original physical volume equally. Compare experimental budgets deliberately.

## Outputs

- `run.json`: arguments, device/PyTorch version, corrected training class counts,
  class weights, supervised IDs, sampling definition and both metadata snapshots.
- `history.json`: per-epoch learning rate, training/validation CE and original-grid
  foreground Dice.
- `epoch_XXX/`: original-grid validation NIfTI volumes and class/patient metrics.
- `bestweights.pt`: plain state_dict usable by `--evaluate-checkpoint`.
- `last.pt`: last model/optimizer/scheduler state, epoch and basic configuration.
- `best_predictions/`: inference and metrics after reloading bestweights.pt.
- `summary.json`: completion and selected/reloaded checkpoint scores.

The destination must not exist. Checkpoint files use temporary-file replacement;
this is not transactional whole-run storage. There is no automatic resume.
Per-epoch predictions consume storage; surface metrics and full validation can
be expensive. Checkpoints are not cryptographically bound to a dataset; the
recorded configuration must match any subsequent evaluation choice. Seeds are
set, but bitwise GPU determinism is not promised.

## Verification and limits

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest \
  test_tiled_training test_evaluate_tiled test_reconstruct_tiled test_tiled_losses \
  test_tiled_dataset test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```

The smoke prepares real synthetic NIfTI files, resamples/saves overlapping and
padded tiles, trains the actual 2D/2.5D ENets for two epochs, reconstructs original
volumes, checks Adam/AdamW and cosine scheduling, and reloads the best checkpoint.
It covers none/xy/xyz and full/partial supervision. Only the preprocessing
sanity check for the real dataset's fixed 512×512/minimum-depth bounds is bypassed.
Other tests cover mathematical loss/gradient equivalence, metadata, missing
files, geometry, masks and the unchanged legacy dispatch.

These are CPU synthetic tests, not a real SegTHOR/GPU training experiment or
proof of segmentation improvement. The older test_simple_resampling.py suite
still contains failures predating this work; see the CLI migration notes.
No original data, remote branch or full training job is modified by these tests.
