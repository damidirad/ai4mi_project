"""Plot best-epoch SEGTHOR results for all preprocessing configurations."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

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

LABELS = {
    "SEGTHOR": "SEGTHOR",
    "SEGTHOR_hu": "SEGTHOR HU",
    "full_xy": "full XY",
    "full_hu_xy": "full HU+XY",
    "full_xyz": "full XYZ",
    "full_hu_xyz": "full HU+XYZ",
    "tiled": "tiled",
    "tiled_hu": "tiled HU",
    "tiled_xy": "tiled XY",
    "tiled_hu_xy": "tiled HU+XY",
    "tiled_xyz": "tiled XYZ",
    "tiled_hu_xyz": "tiled HU+XYZ",
    "baseline": "baseline",
}

LABEL_OFFSETS = {
    "SEGTHOR": (0, -18, "center", "top"),
    "SEGTHOR_hu": (0, 18, "center", "bottom"),
    "full_xy": (-8, 10, "right", "bottom"),
    "full_hu_xy": (8, -12, "left", "top"),
    "full_xyz": (-8, 16, "right", "bottom"),
    "full_hu_xyz": (0, 36, "center", "bottom"),
    "tiled": (0, 8, "center", "bottom"),
    "tiled_hu": (8, 8, "left", "bottom"),
    "tiled_xy": (0, 8, "center", "bottom"),
    "tiled_hu_xy": (0, 8, "center", "bottom"),
    "tiled_xyz": (0, 8, "center", "bottom"),
    "tiled_hu_xyz": (8, 8, "left", "bottom"),
    "baseline": (0, 10, "center", "bottom"),
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


def resolve_array_path(path):
    if path.is_file():
        return path
    alternatives = []
    if path.suffix == ".npy":
        alternatives.append(path.with_suffix(".npz"))
    elif path.suffix == ".npz":
        alternatives.append(path.with_suffix(".npy"))
    for alternative in alternatives:
        if alternative.is_file():
            return alternative
    raise FileNotFoundError(path)


def load_array(path):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            return np.asarray(data[data.files[0]], dtype=float)
        finally:
            data.close()
    return np.asarray(data, dtype=float)


def summarize_array(path):
    """Return macro-average of the four foreground-class patient means."""
    data = load_array(resolve_array_path(path))

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


def load_full_ce_run(run_dir):
    best = run_dir / "best_epoch"
    return {
        "dice": summarize_array(best / "dice_3d_val.npy"),
        "assd": summarize_array(best / "assd_val.npy"),
        "hd95": summarize_array(best / "hd95_val.npy"),
    }


def load_run(results_root, name):
    run_dir = results_root / name / "ce"
    config = configuration(name)

    if config["tiled"]:
        metrics = summarize_tiled_json(
            run_dir / "best_predictions" / "metrics.json"
        )
    else:
        metrics = load_full_ce_run(run_dir)

    return {**config, **metrics}


def load_baseline(baseline_root):
    metrics = load_full_ce_run(baseline_root / "SEGTHOR" / "ce")
    return {
        "name": "baseline",
        "resampling": "baseline",
        "tiled": False,
        "hu": False,
        **metrics,
    }


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
        "--baseline-dir",
        type=Path,
        default=None,
        help=(
            "Optional baseline results root. Defaults to a sibling "
            "'baseline' directory next to --results-dir."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not add the baseline reference point.",
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
    baseline_record = None

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

    if not args.no_baseline:
        baseline_root = (
            args.baseline_dir.expanduser()
            if args.baseline_dir is not None
            else root.parent / "baseline"
        )

        if baseline_root.is_dir():
            try:
                baseline_record = load_baseline(baseline_root)
                baseline_record["boundary_error"] = (
                    args.assd_weight * baseline_record["assd"]
                    + (1 - args.assd_weight) * baseline_record["hd95"]
                )

                if not (
                    np.isfinite(baseline_record["dice"])
                    and np.isfinite(baseline_record["boundary_error"])
                ):
                    warnings.warn("Skipping baseline: invalid aggregate metrics.")
                    baseline_record = None

            except (OSError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(f"Skipping baseline: {exc}")
        else:
            warnings.warn(f"Missing baseline directory: {baseline_root}")

    if not records and baseline_record is None:
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

    if baseline_record is not None:
        ax.scatter(
            baseline_record["dice"],
            baseline_record["boundary_error"],
            marker="*",
            s=360,
            color="#D62728",
            edgecolors="black",
            linewidths=1.6,
            alpha=1.0,
            zorder=6,
        )

    # Create labels ONCE, after all points have been plotted.
    if not args.no_labels:
        texts = []

        labeled_records = records + ([baseline_record] if baseline_record else [])
        for r in labeled_records:
            dx, dy, ha, va = LABEL_OFFSETS.get(
                r["name"], (8, 8, "left", "bottom")
            )
            text = ax.annotate(
                LABELS.get(r["name"], r["name"]),
                xy=(r["dice"], r["boundary_error"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                ha=ha,
                va=va,
                zorder=7,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
            texts.append(text)

        if adjust_text is not None:
            adjust_text(
                texts,
                ax=ax,
                expand=(1.5, 1.8),
                force_text=(1.2, 1.5),
                force_static=(0.85, 1.1),
                iter_lim=500,
            )
        else:
            warnings.warn(
                "adjustText is not installed; labels will not be auto-adjusted."
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
    ax.margins(x=0.08, y=0.08)

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

    baseline_handles = []
    if baseline_record is not None:
        baseline_handles.append(
            Line2D(
                [0], [0],
                marker="*",
                linestyle="None",
                markersize=15,
                markerfacecolor="#D62728",
                markeredgecolor="black",
                label="Baseline",
            )
        )

    legend1 = ax.legend(
        handles=resampling_handles,
        title="Resampling",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=tiling_handles,
        title="Tiling",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
    )
    ax.add_artist(legend2)

    legend3 = ax.legend(
        handles=hu_handles,
        title="HU clipping",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.80),
        frameon=True,
    )
    ax.add_artist(legend3)

    if baseline_handles:
        ax.legend(
            handles=baseline_handles,
            title="Reference",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.58),
            frameon=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {args.output}")

    print("\nSummary of plotted values")
    print(
        f"{'Run':<16} {'Resample':<9} {'Tiled':<7} {'HU':<5} "
        f"{'Dice':>8} {'ASSD':>9} {'HD95':>9} {'Weighted':>10}"
    )

    summary_records = records + ([baseline_record] if baseline_record else [])

    for r in sorted(summary_records, key=lambda item: item["name"]):
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

            for r in summary_records:
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
