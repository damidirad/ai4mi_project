#!/usr/bin/env python3

"""Compare SEGTHOR training with and without isotropic spatial resampling."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from slice_segthor import slice_patient


def prepare_variant(source_dir: Path, variant_dir: Path, train_patients: list[str],
                    val_patients: list[str], shape: tuple[int, int], resample: bool,
                    target_spacing: tuple[float, float, float]) -> None:
    if variant_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {variant_dir}; use --force to recreate the experiment."
        )

    for split, patients in (("train", train_patients), ("val", val_patients)):
        for patient_id in patients:
            patient_dir = source_dir / "train" / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")

            slice_patient(
                dest_path=variant_dir / split,
                source_path=source_dir,
                shape=shape,
                test_mode=False,
                resample=resample,
                target_spacing=target_spacing,
                id_=patient_id,
            )

    metadata = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "resample": resample,
        "target_spacing_mm": target_spacing,
        "slice_shape": shape,
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_variant(variant_dir: Path, result_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("main.py")),
        "--dataset", "SEGTHOR",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--data_root", str(variant_dir.parent),
        "--dest", str(result_dir),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    if args.gpu:
        command.append("--gpu")
    subprocess.run(command, check=True)


def summarize(result_dir: Path) -> dict[str, float | int | str]:
    dice_val = np.load(result_dir / "dice_val.npy")
    foreground_dice = dice_val[:, :, 1:].mean(axis=(1, 2))
    best_epoch = int(np.argmax(foreground_dice))
    return {
        "variant": result_dir.name,
        "best_epoch": best_epoch,
        "best_val_foreground_dice": float(foreground_dice[best_epoch]),
        "final_val_foreground_dice": float(foreground_dice[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=Path("data/segthor_part1"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/spatial_normalization"))
    parser.add_argument("--train_patients", nargs="+", default=["Patient_01", "Patient_02"])
    parser.add_argument("--val_patients", nargs="+", default=["Patient_03"])
    parser.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("DX", "DY", "DZ"),
                        help="Common voxel spacing in mm for the normalized variant.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["full", "partial"], default="full")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Delete this experiment output before recreating it.")
    args = parser.parse_args()

    if set(args.train_patients) & set(args.val_patients):
        parser.error("--train_patients and --val_patients must be disjoint")
    if args.output_dir.exists():
        if not args.force:
            parser.error(f"{args.output_dir} exists; use --force to recreate it")
        shutil.rmtree(args.output_dir)

    target_spacing = tuple(args.target_spacing)
    variants = {
        "without_normalization": False,
        "with_spatial_normalization": True,
    }
    summaries = []
    for name, resample in variants.items():
        variant_dir = args.output_dir / "data" / name / "SEGTHOR"
        result_dir = args.output_dir / "results" / name
        print(f"Preparing {name} in {variant_dir}")
        prepare_variant(args.source_dir, variant_dir, args.train_patients, args.val_patients,
                        tuple(args.shape), resample, target_spacing)
        print(f"Training {name}")
        train_variant(variant_dir, result_dir, args)
        summaries.append(summarize(result_dir))

    summary_path = args.output_dir / "comparison.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Comparison written to {summary_path}")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()