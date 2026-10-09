> This document records its implementation stage. The full route is now
> connected in [stage 8](tiled_training.md); historical statements below about
> training being blocked applied before that integration.

# Stage 7: original-grid restoration and evaluation

`evaluate_tiled.py` connects working-grid reconstruction to the original scan.
The saved original/working affines define the mapping from original voxel
indices to working indices. Nearest-neighbour sampling restores integer labels
0..4 with the original shape, including Z in XYZ mode. Sampling uses nearest
edge extension to avoid floating-point endpoint roundoff creating background
borders. Valid metadata restricts this to the existing endpoint-preserving
resampling geometry; this is not a general registration/resampling tool.

Restoring a grid does not recover detail lost through downsampling.

## Original data and output geometry

The source CT must still match the recorded shape and affine. Original ground
truth is read from its neighbouring `GT.nii.gz`, checked for the same physical
grid, and validated as integer classes 0..4. Evaluation never substitutes the
resampled tile labels for original GT. The Python helper also accepts explicit
source CT/GT paths for relocated data; the CLI uses the recorded CT path.

NIfTI predictions use the original header, affine, qform/sform matrices and
codes, with uint8 label data and unit scaling. Existing output files/directories
are refused. Original source files are required for exact header restoration;
metadata does not archive every header field or verify source content hashes.

## Metrics

Each patient receives foreground-class 1–4 Dice, HD95 and ASSD. Surface distances
use the existing project's 26-neighbour surface erosion and pooled distances
in both directions, with original XYZ spacing in mm. Rotations/flips are
supported; sheared grids are rejected for distances because axis spacings alone
cannot represent those physical distances correctly.

- Both masks empty: Dice=1, distances=0, status `both_empty`.
- Exactly one empty: Dice=0, distances=null, with `target_empty` or
  `prediction_empty` status.
- Otherwise: status `ok`, numeric Dice/HD95/ASSD.

The summary averages Dice equally over patients and four foreground classes.
HD95/ASSD are retained per class/patient without silently excluding undefined
cases to produce an optimistic aggregate.

## Explicit checkpoint evaluation

```bash
python main.py --dataset SEGTHOR --data_root data --dest results/tiled_eval \
  --tiling --evaluate-checkpoint PATH_TO_STATE_DICT.pt --slices 1
```

Use `--slices 3` for a matching 2.5D model; `--gpu` explicitly selects an
available CUDA/MPS device and errors if neither exists. The checkpoint must be
a plain state_dict such as `bestweights.pt`, loaded with `weights_only=True` and
strict parameter matching. Whole-model pickle files are not accepted. Parameter
compatibility does not verify preprocessing, class order or training provenance;
the supplied checkpoint must match the chosen model and prepared dataset.

This route evaluates all validation patients, writes one `Patient_XX.nii.gz`
per patient and a final `metrics.json`. It does not train or create an optimizer.
It cannot combine with `--check-data`, `--check-loss` or `--debug`. The destination
must be new. If interrupted, partial predictions may remain without the final
report; there is no resume or dataset-wide transaction yet.

The Python `evaluate_model` helper also supports a complete test-split dataset,
exporting predictions without GT metrics. Preprocessing the test split and a
test-split CLI selector are not added here.

## Stage boundary and checks

Original-grid inference/export/evaluation is now connected. Full tiled training
remains blocked while the training loop's loss, class counts, validation and
checkpoint selection are joined together. That final integration and an
end-to-end smoke check belong to the next commit; removing the guard alone
would still invoke the legacy full-slice assumptions.

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_evaluate_tiled test_reconstruct_tiled test_tiled_losses test_tiled_dataset test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```

Tests verify independently calculated nearest-neighbour coordinates for none,
XY/XYZ up/downsampling with rotated/flipped translated geometry; exact original
shapes; preserved NIfTI forms/codes and labels; mismatch rejection; known 2 mm
surface distances; empty-mask statuses; and complete checkpoint-to-volume
execution with a small synthetic model. No real training or GPU accuracy is
claimed by these tests.

## Full-slice checkpoints and baseline comparisons

Full-slice metadata (`slice_segthor.py --save_metadata`) records the resized
output grid as well as the working and original CT grids. `restore_original_grid`
now restores schema-1 full-slice labels from the output grid, including the
cell-centred slice-resize transform, and schema-2 tile labels from the working
grid. Both routes use identical original-label Dice, HD95 and ASSD definitions.

`main.py --original-grid-validation` selects full-slice checkpoints by mean
original-CT foreground Dice. It reuses the validation predictions already
computed during the epoch instead of running an additional model pass. Each
epoch writes `original_epoch_XXX/metrics.json` and NIfTI predictions. Existing
prepared-grid metric arrays remain available, but do not determine checkpoint
selection when this option is enabled. Tiled training already uses this rule.

Checkpoint evaluation accepts full-slice datasets without `--tiling`:

```bash
python main.py --dataset SEGTHOR --data_root data/confirmation/fold_1 \
  --dest results/baseline_original_validation --slices 1 \
  --evaluate-checkpoint results/baseline/bestweights.pt --split val --gpu
```

`--class-weighting none` restores the original unweighted CE objective and avoids
the extra training-loader pass for class counts. The default remains
`inverse_frequency`. The original ENet training recipe uses one slice, CE, Adam
(lr 0.0005, betas 0.9/0.999, no weight decay), no scheduler and no augmentation,
HU clipping, resampling or tiling. The confirmation protocol uses 20 epochs and
original-grid checkpoint selection for all models; it does not claim to reproduce
the original repository's 2D-Dice checkpoint-selection protocol bit for bit.
