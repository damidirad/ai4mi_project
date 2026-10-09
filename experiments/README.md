# Screening experiments

`run_experiments.py` executes one configuration from `screening.json` by calling
`main.py`. It does not implement training or submit jobs itself. Code and run
settings are recorded with each attempt. M1 is the reference within this screen,
not the original unweighted-CE baseline. A faithful original baseline is a
separate experiment that is not included in the supplied eight configurations.

| ID | HU | Resampling | Tiles | Slices | Loss | Optimizer | Scheduler | Augmentation | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Yes | none | No | 1 | CE_DICE | Adam | none | No | 20 |
| M2 | Yes | none | No | 1 | CE_DICE | Adam | none | Yes | 20 |
| M3 | Yes | none | No | 1 | CE_DICE | AdamW | cosine | Yes | 20 |
| M4 | Yes | none | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M5 | Yes | xy | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M6 | Yes | xyz | No | 3 | CE_DICE | AdamW | cosine | Yes | 20 |
| M7 | Yes | xy | Yes | 3 | CE_TILED | AdamW | cosine | Yes | 3 |
| M8 | Yes | xy | No | 3 | CE_DICE_BOUNDARY | AdamW | cosine | Yes | 20 |

`CE_TILED` maps to `--tiling --loss ce`. M7's three epochs are an explicit
screening budget, not a claim of equal optimizer updates to the other runs.
All configurations use full supervision, split seed 0, fold 0, five validation
patients and training seed 0.

**M2–M8 are blocked until augmentation is implemented and connected to this
runner.** Dry runs display the intended configuration and the blocker. The
training command shown for a blocked run is not an executable augmentation
recipe. Never remove the blocker without also forwarding the augmentation
option to training and checking image/label alignment and validation behavior.

## Prepare the data

The Makefile consumes raw scans; it does not create raw data. Its default source
is `data/segthor_train_full`, extracted from `data/segthor_train_full.zip` if
needed. An alternative source can be specified with `SEGTHOR_SOURCE=...` and
must contain `train/Patient_XX/Patient_XX.nii.gz` and `GT.nii.gz`.

Only four prepared datasets are needed for these eight configurations:

```bash
make data/SEGTHOR_hu data/full_hu_xy/SEGTHOR \
  data/full_hu_xyz/SEGTHOR data/tiled_hu_xy/SEGTHOR \
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
ai4mi/bin/python run_experiments.py --run-id M7 --dry-run
```

Each run can also be selected by its zero-based `--run-index` (0=M1, 7=M8).
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
sbatch snellius/screening.sbatch
```

By default only index 0 (M1) is submitted. Add `--account=...` to `sbatch` if your
allocation requires an explicit account. Paths below are examples of overrides:

```bash
sbatch snellius/screening.sbatch --source-dir /path/to/raw/scans
```

Once augmentation is integrated, the same script can schedule independent jobs:

```bash
sbatch --array=0-7%2 snellius/screening.sbatch
```

The `%2` caps concurrent runs at two. Do not submit this full array while the
augmentation blocker remains. These instructions do not submit jobs themselves.

## Results and retries

Each attempt has its own directory:

```text
results/screening/M1/seed_0/attempt_001/
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
needed before final model selection. This runner does not select winners or
implement the later multi-seed/fold phase or submission export.
