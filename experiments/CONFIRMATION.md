# Confirm promising models across patients and training seeds

The same `run_experiments.py` runner reads `confirmation.json`. This protocol
compares the original ENet training recipe with two explicitly selected screening
configurations. It never chooses winners from incomplete or mixed-grid scores.

## Patient folds

The raw patient IDs are sorted, shuffled once with split seed 0, then divided into
blocks of five. Fold 0 was used for screening. Folds 1, 2 and 3 hold out the next
three blocks for validation. All other patients train each model. These are
patient-level holdouts with non-overlapping validation groups, **not three equal
K-fold partitions of the entire dataset**. At least 20 patients are required for
the configured blocks. Every model and training seed uses the same patients in
a given fold. The raw source inventory must remain unchanged between runs.

Training sets overlap between folds. The expanded runs are not 27 independent
patient cohorts. Candidate selection already used the screening data; this is a
robustness comparison, not an untouched final test. Do not use test-set outcomes
to select candidates.

## Model selection and baseline

Set `top_candidates.TOP1` and `TOP2` to distinct IDs from `screening.json` only
when the candidates are chosen. Their initial null values are intentional;
placeholder jobs print a blocker and cannot train. The baseline can run while
these placeholders remain unset. Selected augmented candidates stay blocked
until augmentation is actually integrated. Preprocessing can be prepared without
augmentation because augmentation belongs to training.

The baseline is **not M1**. It uses the original ENet with 1 slice, unweighted CE,
Adam (lr 0.0005, betas 0.9/0.999), no scheduler and no augmentation, HU clipping,
resampling or tiling. Full slices are resized to 256x256 as in the initial repo.
The runner requests `--class-weighting none` to disable the current class weights.
All runs use full supervision. Full-slice models train for 20 epochs and tiled
models for 3, as requested. These are not equal optimizer-update budgets. There
is no early stopping or checkpoint resume.

All models select their best checkpoint by mean foreground 3D Dice on the
**original CT grid**. Both layouts use the same Dice, HD95, ASSD, interpolation
and empty-mask conventions in `evaluate_tiled.py`. This shared evaluation
protocol deliberately differs from the initial repo's 2D-Dice checkpoint
selection. The selected checkpoint is reloaded and evaluated again before a
confirmation run is marked completed.

## Run index mapping

Three roles × three folds × three seeds gives 27 jobs. Training seeds are 0, 1, 2.
Within each row below, indices correspond to seeds 0, 1, 2 in order.

| Role | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| BASELINE | 0, 1, 2 | 3, 4, 5 | 6, 7, 8 |
| TOP1 | 9, 10, 11 | 12, 13, 14 | 15, 16, 17 |
| TOP2 | 18, 19, 20 | 21, 22, 23 | 24, 25, 26 |

```bash
ai4mi/bin/python run_experiments.py --config experiments/confirmation.json \
  --run-index 0 --dry-run
```

The plan shows the fold, seed, preprocessing command, training command,
checkpoint evaluation command and any blocker. No files are created in dry runs.
Freeze candidate choices and commit the configuration before launching jobs.

## Prepare each fold once

Use `--prepare-only` in a CPU allocation before submitting GPU jobs. Settings
such as the HU window and automatic target spacing are derived separately from
each fold's training patients. The preprocessing command passes `--save_metadata`.
There is no need to rerun preprocessing for each training seed.

```bash
# Baseline datasets for all three folds.
for index in 0 3 6; do
  ai4mi/bin/python run_experiments.py --config experiments/confirmation.json \
    --run-index "$index" --prepare-only --processes 4
done

# After selecting TOP1 and TOP2, prepare their datasets as well.
for index in 9 12 15 18 21 24; do
  ai4mi/bin/python run_experiments.py --config experiments/confirmation.json \
    --run-index "$index" --prepare-only --processes 4
done
```

Data lives under `data/confirmation/fold_N/`, using the same layout names as the
Makefile. Candidates with identical preprocessing reuse the same verified data.
Existing partial or mismatched output is refused; no data is deleted. A file lock
prevents two processes from preparing the same dataset concurrently. Use
`--source-dir` and `--data-root` if your data is elsewhere. Training never starts
preprocessing automatically, so GPU jobs do not race or spend allocation time
preparing data. The original Makefile targets still describe the screening split;
changing their FOLD variable does not produce the confirmation directory layout.

## Submit to Snellius

From the repository root with the Python environment's required modules loaded:

```bash
mkdir -p results/slurm
# Run the nine baseline jobs, up to four concurrently.
sbatch snellius/confirmation.sbatch

# Once candidates, augmentation and datasets are ready, run all 27 jobs.
sbatch --array=0-26%4 snellius/confirmation.sbatch
```

Each task requests one H100, 16 CPUs, 180 GiB host RAM and six hours, including
final evaluation. The `%4` limit applies to this array, not unrelated jobs from
other arrays; do not overlap submissions if four is your total slot limit.
Twenty-seven six-hour jobs represent at most 162 allocated GPU-hours; with four
slots they need about seven scheduling waves, excluding queue delays. Actual
completion time depends on the workloads and scheduler. No jobs are submitted by
the Python runner itself.

Results are isolated by role, fold, seed and attempt:

```text
results/confirmation/BASELINE/fold_1/seed_0/attempt_001/
  runner.json
  train.log
  evaluation.log
  training/
  original_validation/
    Patient_XX.nii.gz
    metrics.json
```

The manifest records the resolved candidate configuration (including its source
M ID), dataset metadata hashes, source hashes and original-grid Dice. The report
contains per-patient, per-organ Dice, HD95 and ASSD. Compare corresponding
fold/seed pairs against the baseline; do not treat tiles or repeated predictions
of the same patient as independent observations. A missing run or metric is not
a zero score. Submission packaging remains a separate task.
