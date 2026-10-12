# Initial comparison experiments

`run_experiments.py` executes one configuration from `initial_comparison.json` by calling
`main.py`. It does not implement training or submit jobs itself. Code and run
settings are recorded with each attempt. M1 is the reference within this initial comparison,
not the original unweighted-CE baseline. A faithful original baseline is a
separate experiment that is not included in these configurations.

| ID | HU | Resampling | Tiles | Slices | Loss | Optimizer | Scheduler | Augmentation | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Yes | none | No | 1 | CE_DICE | Adam | none | No | 20 |
| M2 | Yes | none | No | 1 | CE_DICE | Adam | none | Yes | 20 |
| M3 | Yes | none | No | 1 | CE_DICE | AdamW | cosine | Yes | 20 |
| M4 | Yes | none | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M5 | Yes | xy | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M6 | Yes | xyz | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M7 | Yes | xy | Yes | 3 | CE_TILED | AdamW | cosine | No | 3 |
| M8 | Yes | xy | No | 3 | CE_DICE_BOUNDARY | AdamW | cosine | Yes | 20 |
| M9 | Yes | none | Yes | 3 | CE_TILED | AdamW | cosine | No | 3 |

`CE_TILED` maps to `--tiling --loss ce`. M7 and M9's three epochs are an explicit
initial comparison budget, not a claim of equal optimizer updates to the other runs.
All configurations use full supervision, split seed 0, fold 0, five validation
patients and training seed 0.

Full-slice configurations M2–M6 and M8 use `--augmentation mild`, sharing one
random affine transformation across each input stack and label. M1 and the tiled
configurations M7 and M9 use `--augmentation none`. Tiled augmentation is deliberately
excluded to preserve overlap geometry and coverage-weight interpretation; see
[the design decision](../docs/tiled_training.md#design-decision-no-data-augmentation-in-tiled-training).
Validation and checkpoint evaluation are not augmented. Tiled runs therefore differ
from augmented full-slice runs in both tiling and augmentation, so that comparison
does not isolate the effect of tiling alone.

## Prepare the data

The Makefile consumes raw scans; it does not create raw data. Its default source
is `data/segthor_train_full`, extracted from `data/segthor_train_full.zip` if
needed. An alternative source can be specified with `SEGTHOR_SOURCE=...` and
must contain `train/Patient_XX/Patient_XX.nii.gz` and `GT.nii.gz`.

Only five prepared datasets are needed for these configurations:

```bash
make data/SEGTHOR_hu data/full_hu_xy/SEGTHOR \
  data/full_hu_xyz/SEGTHOR data/tiled_hu/SEGTHOR data/tiled_hu_xy/SEGTHOR \
  PYTHON=ai4mi/bin/python SEED=0 FOLD=0 RETAINS=5
```

`make preprocess-all` is also supported but prepares eight additional datasets
not used here. Run preprocessing in an appropriate compute allocation before
submitting training jobs; never prepare the same output concurrently.

The Makefile now requests metadata for full slices as well as tiles. The runner
verifies the selected dataset's patient IDs against the raw source and configured
split, and checks HU, resampling and layout settings. Existing datasets without
metadata are refused. Changing Make variables does not invalidate existing
outputs: regenerate such datasets in a fresh data root (for example with
`slice_segthor.py --save_metadata`) and pass `--data-root` to the runner. Do not
assume a directory name alone proves its preprocessing configuration.

## Inspect and execute

From the repository root, inspect a run without touching data or creating files:

```bash
ai4mi/bin/python run_experiments.py --run-id M1 --dry-run
ai4mi/bin/python run_experiments.py --run-id M9 --dry-run
```

Each run can also be selected by its zero-based `--run-index` (0=M1, 8=M9).
Execution requires a CUDA GPU and prepared data. No CPU fallback is allowed.

```bash
ai4mi/bin/python run_experiments.py --run-id M1
```

## Snellius

The job script requests one H100 in `gpu_h100`, 16 CPU cores, 180 GiB host RAM,
and six hours. These resources match the minimum H100 node allocation described
in the [SURF partition documentation](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions).
Four data-loader workers are used. Six hours is a limit, not a completion-time
guarantee. The job uses your existing `ai4mi/bin/python`; load any environment
modules required by that environment before submitting. Set `EXPERIMENT_PYTHON`
to an alternative compatible interpreter if needed.

```bash
mkdir -p results/slurm
sbatch snellius/initial_comparison.sbatch
```

By default only index 0 (M1) is submitted. Add `--account=...` to `sbatch` if your
allocation requires an explicit account. Paths below are examples of overrides:

```bash
sbatch snellius/initial_comparison.sbatch --source-dir /path/to/raw/scans
```

The same script can schedule independent jobs:

```bash
sbatch --array=0-8%2 snellius/initial_comparison.sbatch
```

The `%2` caps concurrent runs at two. These instructions do not submit jobs themselves.

## Results and retries

Each attempt has its own directory:

```text
results/initial_comparison/M1/seed_0/attempt_001/
  runner.json
  train.log
  training/
    bestweights.pt
    ...
```

`runner.json` records the configuration, exact command, metadata hashes, Python
source hashes, patient IDs, sample counts, status, Slurm job ID and selected
checkpoint's Dice. Live training output is in `train.log`; Slurm logs contain the
plan and final status. A matching completed attempt is skipped only if its
required outputs can still be read and its summary matches. Changed settings or
source code, failed attempts and incomplete directories are never overwritten.
Use `--attempt 2` after inspecting a failed run. This restarts training; it does
not resume checkpoints. A hard kill/time limit can leave status `running`;
such a directory is treated as incomplete, never as a successful run.

Metrics are labeled with their evaluation grid: current full-slice runs use the
prepared volume, whereas tiled runs use the original CT grid. Do not automatically
rank these scores as a fair comparison; common original-grid evaluation is still
needed before final model selection. The same runner now also supports the later
multi-seed/fold phase through `final_comparison.json`.

## Follow-up comparisons

See [FINAL_COMPARISON.md](FINAL_COMPARISON.md) for the original baseline plus two selected
candidates across three patient folds and three training seeds. Full-slice
original-grid checkpoint selection is now available through
`--original-grid-validation`; this is enabled by the final comparison configuration,
not retroactively for the initial comparison runs described above.
