"""Plot best-epoch SEGTHOR results for all preprocessing configurations."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from adjustText import adjust_text

RUNS = [
    "SEGTHOR", "SEGTHOR_hu",
    "full_xy", "full_hu_xy",
    "full_xyz", "full_hu_xyz",
    "tiled", "tiled_hu",
    "tiled_xy", "tiled_hu_xy",
    "tiled_xyz", "tiled_hu_xyz",
]

CLASSES = (1, 2, 3, 4)

# Color represents resampling.
RESAMPLING_COLORS = {
    "none": "#3974B8",
    "xy": "#D9822B",
    "xyz": "#27845A",
}

# Marker shape represents tiling.
MARKERS = {
    False: "o",  # full slices
    True: "s",   # tiled
}


def configuration(name):
    """Infer preprocessing factors from a run name."""
    tiled = name.startswith("tiled")
    hu = "_hu" in name

    if name in ("SEGTHOR", "SEGTHOR_hu", "tiled", "tiled_hu"):
        resampling = "none"
    elif name in (
        "full_xy", "full_hu_xy", "tiled_xy", "tiled_hu_xy"
    ):
        resampling = "xy"
    else:
        resampling = "xyz"

    return {
        "name": name,
        "tiled": tiled,
        "hu": hu,
        "resampling": resampling,
    }


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def summarize_npy(path):
    """Return macro-average of the four foreground-class patient means."""
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.asarray(np.load(path), dtype=float)

    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(
            f"{path}: expected shape (patients, 5), got {data.shape}"
        )

    class_means = [finite_mean(data[:, c]) for c in CLASSES]
    valid = [v for v in class_means if np.isfinite(v)]

    return float(np.mean(valid)) if valid else np.nan


def summarize_tiled_json(path):
    """Average patient values within each foreground class, then across classes."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    patients = payload.get("patients", {})
    if not patients:
        raise ValueError(f"No patient metrics found in {path}")

    key_map = {
        "dice": "dice",
        "assd": "assd_mm",
        "hd95": "hd95_mm",
    }
    result = {}

    for metric, json_key in key_map.items():
        class_means = []

        for class_id in CLASSES:
            values = []

            for patient in patients.values():
                class_data = patient.get(str(class_id), {})

                if not isinstance(class_data, dict):
                    continue
                if class_data.get("status", "ok") != "ok":
                    continue

                try:
                    value = float(class_data[json_key])
                except (KeyError, TypeError, ValueError):
                    continue

                if np.isfinite(value):
                    values.append(value)

            class_means.append(finite_mean(values))

        valid = [v for v in class_means if np.isfinite(v)]
        result[metric] = (
            float(np.mean(valid)) if valid else np.nan
        )

    return result


def load_run(results_root, name):
    run_dir = results_root / name / "ce"
    config = configuration(name)

    if config["tiled"]:
        metrics = summarize_tiled_json(
            run_dir / "best_predictions" / "metrics.json"
        )
    else:
        best = run_dir / "best_epoch"
        metrics = {
            "dice": summarize_npy(best / "dice_3d_val.npy"),
            "assd": summarize_npy(best / "assd_val.npy"),
            "hd95": summarize_npy(best / "hd95_val.npy"),
        }

    return {**config, **metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Compare best-epoch SEGTHOR preprocessing configurations."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/scur0147/ai4mi_project/results/segthor_runs_2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/scur0147/ai4mi_project/plots/segthor_run_2_best_epoch_comparison.png"),
    )
    parser.add_argument(
        "--assd-weight",
        type=float,
        default=0.5,
        help="ASSD weight between 0 and 1; HD95 weight is 1 minus this.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide point labels.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for exporting summary metrics.",
    )
    args = parser.parse_args()

    if not 0 <= args.assd_weight <= 1:
        parser.error("--assd-weight must be between 0 and 1.")

    root = args.results_dir.expanduser()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")

    records = []

    for name in RUNS:
        if not (root / name).is_dir():
            warnings.warn(f"Missing run directory: {root / name}")
            continue

        try:
            record = load_run(root, name)
            record["boundary_error"] = (
                args.assd_weight * record["assd"]
                + (1 - args.assd_weight) * record["hd95"]
            )

            if not (
                np.isfinite(record["dice"])
                and np.isfinite(record["boundary_error"])
            ):
                warnings.warn(f"Skipping {name}: invalid aggregate metrics.")
                continue

            records.append(record)

        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f"Skipping {name}: {exc}")

    if not records:
        parser.error("No valid runs found. Check the results directory and file layout.")

    fig, ax = plt.subplots(figsize=(14, 8))


    # Plot all runs: color = resampling, shape = tiling,
    # outline thickness = HU clipping.
    for r in records:
        color = RESAMPLING_COLORS[r["resampling"]]
        edge_width = 2.8 if r["hu"] else 0.8

        ax.scatter(
            r["dice"],
            r["boundary_error"],
            marker=MARKERS[r["tiled"]],
            s=155,
            color=color,
            edgecolors="black",
            linewidths=edge_width,
            alpha=0.95,
            zorder=3,
        )

    # Create labels ONCE, after all points have been plotted.
    if not args.no_labels:
        texts = []

        for r in records:
            texts.append(
                ax.text(
                    r["dice"],
                    r["boundary_error"],
                    r["name"],
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )
            )

        adjust_text(
            texts,
            ax=ax,
            expand=(1.5, 1.8),
            force_text=(1.2, 1.5),
            force_static=(0.85, 1.1),
            iter_lim=500,
        )
    # Optional paired lines: connect configurations differing only in HU.
    """
    by_name = {r["name"]: r for r in records}
    hu_pairs = [
        ("SEGTHOR", "SEGTHOR_hu"),
        ("full_xy", "full_hu_xy"),
        ("full_xyz", "full_hu_xyz"),
        ("tiled", "tiled_hu"),
        ("tiled_xy", "tiled_hu_xy"),
        ("tiled_xyz", "tiled_hu_xyz"),
    ]

    for plain, clipped in hu_pairs:
        if plain in by_name and clipped in by_name:
            a, b = by_name[plain], by_name[clipped]
            ax.plot(
                [a["dice"], b["dice"]],
                [a["boundary_error"], b["boundary_error"]],
                color="gray",
                linewidth=0.9,
                alpha=0.35,
                zorder=1,
            )
    """

    ax.set_xlabel("Mean foreground 3D Dice (higher is better)", fontsize=12)
    ax.set_ylabel(
        f"Weighted boundary error (mm): "
        f"{args.assd_weight:.2f} × ASSD + "
        f"{1 - args.assd_weight:.2f} × HD95 (lower is better)",
        fontsize=12,
    )
    ax.set_title("SEGTHOR Best-Epoch Comparison", fontsize=15, pad=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Separate legends make the encodings easy to interpret.
    resampling_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor="black",
            label=label,
        )
        for label, color in RESAMPLING_COLORS.items()
    ]

    tiling_handles = [
        Line2D(
            [0], [0],
            marker=MARKERS[tiled],
            linestyle="None",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            label=label,
        )
        for tiled, label in [
            (False, "Full slices"),
            (True, "Tiled"),
        ]
    ]

    hu_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=width,
            label=label,
        )
        for width, label in [
            (0.8, "No HU clipping"),
            (2.8, "HU clipping"),
        ]
    ]

    legend1 = ax.legend(
        handles=resampling_handles,
        title="Resampling",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=tiling_handles + hu_handles,
        title="Other preprocessing factors",
        loc="best",
        frameon=True,
        ncol=2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {args.output}")

    print("\nSummary of plotted values")
    print(
        f"{'Run':<16} {'Resample':<9} {'Tiled':<7} {'HU':<5} "
        f"{'Dice':>8} {'ASSD':>9} {'HD95':>9} {'Weighted':>10}"
    )

    for r in sorted(records, key=lambda item: item["name"]):
        print(
            f"{r['name']:<16} {r['resampling']:<9} "
            f"{str(r['tiled']):<7} {str(r['hu']):<5} "
            f"{r['dice']:>8.4f} {r['assd']:>9.4f} "
            f"{r['hd95']:>9.4f} {r['boundary_error']:>10.4f}"
        )

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "name", "resampling", "tiled", "hu",
                "dice", "assd_mm", "hd95_mm",
                "weighted_boundary_error_mm",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for r in records:
                writer.writerow({
                    "name": r["name"],
                    "resampling": r["resampling"],
                    "tiled": r["tiled"],
                    "hu": r["hu"],
                    "dice": r["dice"],
                    "assd_mm": r["assd"],
                    "hd95_mm": r["hd95"],
                    "weighted_boundary_error_mm": r["boundary_error"],
                })

        print(f"Saved CSV to: {args.csv}")

    plt.show()


if __name__ == "__main__":
    main()