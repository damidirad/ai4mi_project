#!/usr/bin/env python3
"""Build the AI4MI SEGTHOR submission folder from a completed model run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


PATIENT_RE = re.compile(r"Patient_\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the README submission structure for a trained SEGTHOR model."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing training/ and optionally original_validation/.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Matching prepared SEGTHOR dataset directory with val/metadata.")
    parser.add_argument("--test-pred-dir", type=Path, required=True,
                        help="Directory containing original-grid test Patient_XX.nii.gz predictions.")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Destination folder to create, e.g. submission/group-26.")
    parser.add_argument("--group", default="26",
                        help="Group identifier used in filenames. Default: 26.")
    parser.add_argument("--val-pred-dir", type=Path,
                        help="Override validation prediction directory. Defaults to run-dir/original_validation, "
                             "then the best training/original_epoch_XXX directory.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Repository root for creating the git bundle. Default: current directory.")
    parser.add_argument("--git-ref", default="HEAD",
                        help="Git ref to bundle. Use master/main if required by the course. Default: HEAD.")
    parser.add_argument("--skip-git-bundle", action="store_true",
                        help="Do not create group-XX.bundle.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove an existing destination folder before packaging.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        fail(f"missing file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def patient_id(path: Path) -> str:
    match = PATIENT_RE.search(path.name)
    if not match:
        fail(f"could not infer patient id from {path}")
    return match.group(0)


def nii_predictions(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        fail(f"prediction directory does not exist: {folder}")
    predictions = {patient_id(path): path for path in sorted(folder.glob("Patient_*.nii.gz"))}
    if not predictions:
        fail(f"no Patient_XX.nii.gz predictions found in {folder}")
    return predictions


def best_epoch_from_text(training_dir: Path) -> int | None:
    path = training_dir / "best_epoch.txt"
    if not path.is_file():
        return None
    text = path.read_text()
    match = re.search(r"epoch\s+(\d+)", text)
    return int(match.group(1)) if match else None


def resolve_val_pred_dir(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    original_validation = run_dir / "original_validation"
    if list(original_validation.glob("Patient_*.nii.gz")):
        return original_validation

    training_dir = run_dir / "training"
    best_epoch = best_epoch_from_text(training_dir)
    if best_epoch is not None:
        candidate = training_dir / f"original_epoch_{best_epoch:03d}"
        if list(candidate.glob("Patient_*.nii.gz")):
            return candidate

    candidates = sorted(training_dir.glob("original_epoch_*"))
    candidates = [path for path in candidates if list(path.glob("Patient_*.nii.gz"))]
    if candidates:
        return candidates[-1]
    fail(f"could not find validation NIfTI predictions under {run_dir}")


def resolve_metrics_json(run_dir: Path, val_pred_dir: Path) -> Path:
    candidates = [
        val_pred_dir / "metrics.json",
        run_dir / "original_validation" / "metrics.json",
        run_dir / "training" / "best_epoch" / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fail("could not find validation metrics.json")


def write_metric_npz(metrics_json: Path, destination: Path) -> list[Path]:
    report = json.loads(metrics_json.read_text())
    patients = report.get("patients")
    if not isinstance(patients, dict) or not patients:
        fail(f"{metrics_json} does not contain a non-empty patients mapping")

    metric_specs = {
        "dice_3d.npz": "dice",
        "hd95.npz": "hd95_mm",
        "assd.npz": "assd_mm",
    }
    written = []
    for filename, metric_key in metric_specs.items():
        arrays = {}
        for patient, class_metrics in sorted(patients.items()):
            values = []
            for class_id in sorted(class_metrics, key=lambda item: int(item)):
                value = class_metrics[class_id].get(metric_key)
                values.append(np.nan if value is None else float(value))
            arrays[patient] = np.asarray(values, dtype=np.float32)
        out = destination / filename
        np.savez_compressed(out, **arrays)
        written.append(out)
    return written


def copy_validation_gt(metadata_dir: Path, patient_ids: set[str], destination: Path) -> None:
    if not metadata_dir.is_dir():
        fail(f"validation metadata directory does not exist: {metadata_dir}")
    for patient in sorted(patient_ids):
        metadata_path = metadata_dir / f"{patient}.json"
        if not metadata_path.is_file():
            fail(f"missing validation metadata for {patient}: {metadata_path}")
        record = json.loads(metadata_path.read_text())
        source_ct = Path(record["source_ct"])
        gt_path = source_ct.parent / "GT.nii.gz"
        copy_file(gt_path, destination / f"{patient}.nii.gz")


def copy_predictions(predictions: dict[str, Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for patient, source in sorted(predictions.items()):
        copy_file(source, destination / f"{patient}.nii.gz")


def create_git_bundle(repo_root: Path, git_ref: str, destination: Path) -> None:
    command = ["git", "bundle", "create", str(destination), git_ref]
    subprocess.run(command, cwd=repo_root, check=True)


def git_status(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return ["Could not read git status."]
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    test_pred_dir = args.test_pred_dir.resolve()
    dest = args.dest.resolve()

    if dest.exists():
        if not args.overwrite:
            fail(f"{dest} already exists; pass --overwrite to replace it")
        shutil.rmtree(dest)

    training_dir = run_dir / "training"
    best_model = training_dir / "bestmodel.pkl"
    best_weights = training_dir / "bestweights.pt"
    if not best_model.is_file():
        fail(f"missing best model: {best_model}")
    if not best_weights.is_file():
        fail(f"missing best weights: {best_weights}")

    val_pred_dir = resolve_val_pred_dir(run_dir, args.val_pred_dir.resolve() if args.val_pred_dir else None)
    metrics_json = resolve_metrics_json(run_dir, val_pred_dir)
    val_predictions = nii_predictions(val_pred_dir)
    test_predictions = nii_predictions(test_pred_dir)

    val_pred_out = dest / "val" / "pred"
    val_gt_out = dest / "val" / "gt"
    test_pred_out = dest / "test" / "pred"
    dest.mkdir(parents=True, exist_ok=False)

    copy_predictions(test_predictions, test_pred_out)
    copy_predictions(val_predictions, val_pred_out)
    copy_validation_gt(dataset_dir / "val" / "metadata", set(val_predictions), val_gt_out)

    metric_files = write_metric_npz(metrics_json, dest / "val")
    copy_file(metrics_json, dest / "val" / "metrics.json")
    copy_file(best_model, dest / f"bestmodel-group-{args.group}.pkl")
    copy_file(best_weights, dest / f"bestweights-group-{args.group}.pt")

    bundle_path = None
    if not args.skip_git_bundle:
        bundle_path = dest / f"group-{args.group}.bundle"
        create_git_bundle(args.repo_root.resolve(), args.git_ref, bundle_path)

    status = git_status(args.repo_root.resolve())
    manifest = {
        "group": args.group,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "val_pred_dir": str(val_pred_dir),
        "test_pred_dir": str(test_pred_dir),
        "metrics_json": str(metrics_json),
        "validation_patients": sorted(val_predictions),
        "test_patients": sorted(test_predictions),
        "metric_npz": [path.name for path in metric_files],
        "best_model": f"bestmodel-group-{args.group}.pkl",
        "best_weights": f"bestweights-group-{args.group}.pt",
        "git_bundle": bundle_path.name if bundle_path else None,
        "git_ref": args.git_ref if bundle_path else None,
        "git_status_warnings": status,
    }
    (dest / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Created submission folder: {dest}")
    print(f"Validation predictions: {len(val_predictions)}")
    print(f"Test predictions: {len(test_predictions)}")
    if status:
        print("WARNING: git has uncommitted/untracked changes; the bundle contains only committed git data.")


if __name__ == "__main__":
    main()
