#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CLASS_ORDER = {
    "Background": 0,
    "Heart": 1,
    "Esophagus": 2,
    "Trachea": 3,
    "Aorta": 4,
}

# Background is excluded from evaluation
EVAL_CLASSES = {
    name: idx
    for name, idx in CLASS_ORDER.items()
    if idx != 0
}


def load_metric(path):
    arr = np.load(path)

    # (E, N) -> (E, N, 1)
    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(
            f"{path} has shape {arr.shape}; "
            "expected (epochs, samples) or (epochs, samples, classes)."
        )

    return arr


def save_legend(handles, labels, path, title="Preprocessing"):
    """Save a standalone legend with a white background."""

    fig = plt.figure(
        figsize=(6, 3),
        facecolor="white",
    )

    fig.legend(
        handles,
        labels,
        title=title,
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )

    plt.close(fig)

    print(f"Saved legend: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric",
        required=True,
        help="Metric filename, e.g. dice_val.npy",
    )

    parser.add_argument(
        "--result_dirs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--dest",
        default=None,
    )

    args = parser.parse_args()

    if len(args.result_dirs) != len(args.labels):
        raise ValueError(
            "Number of result directories must equal number of labels."
        )

    rows = []

    for result_dir, experiment in zip(
        args.result_dirs,
        args.labels,
    ):
        metric_path = Path(result_dir) / args.metric

        if not metric_path.exists():
            raise FileNotFoundError(
                f"Metric not found: {metric_path}"
            )

        metrics = load_metric(metric_path)

        print(
            f"{experiment}: "
            f"{args.metric} shape = {metrics.shape}"
        )

        if metrics.shape[2] < 5:
            raise ValueError(
                f"{metric_path} contains {metrics.shape[2]} classes, "
                "but 5 classes are expected."
            )

        eval_ids = list(EVAL_CLASSES.values())

        for epoch in range(metrics.shape[0]):

            # Individual classes
            for class_name, class_id in EVAL_CLASSES.items():

                values = metrics[
                    epoch,
                    :,
                    class_id,
                ]

                for value in values:
                    rows.append({
                        "Epoch": epoch + 1,
                        "Experiment": experiment,
                        "Class": class_name,
                        "Dice": float(value),
                    })

            # Foreground mean
            foreground = metrics[
                epoch,
                :,
                eval_ids,
            ]

            foreground_mean = foreground.mean(axis=1)

            for value in foreground_mean:
                rows.append({
                    "Epoch": epoch + 1,
                    "Experiment": experiment,
                    "Class": "Mean Dice",
                    "Dice": float(value),
                })

    df = pd.DataFrame(rows)

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    # ========================================================
    # Output paths
    # ========================================================

    if args.dest:
        dest = Path(args.dest)
        dest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mean_dest = dest.with_name(
            dest.stem + "_mean" + dest.suffix
        )

        mean_legend_dest = dest.with_name(
            dest.stem + "_mean_legend" + dest.suffix
        )

        class_dest = dest.with_name(
            dest.stem + "_per_class" + dest.suffix
        )

        class_legend_dest = dest.with_name(
            dest.stem + "_per_class_legend" + dest.suffix
        )

    # ========================================================
    # Plot 1: foreground mean
    # ========================================================

    mean_df = df[
        df["Class"] == "Mean Dice"
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.lineplot(
        data=mean_df,
        x="Epoch",
        y="Dice",
        hue="Experiment",
        errorbar="sd",
        linewidth=2.5,
        ax=ax,
    )

    # Extract legend before removing it
    handles, labels = ax.get_legend_handles_labels()

    # Remove legend from main figure
    legend = ax.get_legend()

    if legend is not None:
        legend.remove()

    ax.set_title(
        "Mean Dice"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Dice"
    )

    ax.set_ylim(
        0,
        1
    )

    fig.tight_layout()

    if args.dest:
        fig.savefig(
            mean_dest,
            dpi=200,
            bbox_inches="tight",
        )

        print(f"Saved: {mean_dest}")

        save_legend(
            handles,
            labels,
            mean_legend_dest,
        )

    plt.show()
    plt.close(fig)

    # ========================================================
    # Plot 2: individual classes
    # ========================================================

    class_df = df[
        df["Class"] != "Mean Dice"
    ]

    g = sns.relplot(
        data=class_df,
        x="Epoch",
        y="Dice",
        hue="Experiment",
        col="Class",
        col_order=list(EVAL_CLASSES.keys()),
        kind="line",
        errorbar="sd",
        linewidth=2,
        height=5,
        aspect=0.85,
    )

    # Get legend handles before removing legend
    legend = g._legend

    if legend is not None:
        handles = legend.legend_handles
        labels = [
            text.get_text()
            for text in legend.texts
        ]

        legend.remove()

    g.set_axis_labels(
        "Epoch",
        "Dice",
    )

    g.set_titles(
        "{col_name}"
    )

    g.set(
        ylim=(0, 1)
    )

    g.figure.suptitle(
        "Dice by Anatomical Structure",
        y=1.05,
    )

    g.figure.tight_layout()

    if args.dest:
        g.figure.savefig(
            class_dest,
            dpi=200,
            bbox_inches="tight",
        )

        print(f"Saved: {class_dest}")

        save_legend(
            handles,
            labels,
            class_legend_dest,
        )

    plt.show()
    plt.close(g.figure)


if __name__ == "__main__":
    main()