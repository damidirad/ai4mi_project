import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def run(args: argparse.Namespace) -> None:

    metric_files = sorted(args.metric_dir.glob(f"iter*/{args.metric_name}.npy"))

    means = []

    for file in metric_files:

        metrics = np.load(file)  # (patients, classes)

        # Exclude background
        mean_metric = metrics[:, 1:].mean()

        means.append(mean_metric)

    epcs = np.arange(len(means))

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(epcs, means, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)

    fig.tight_layout()

    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Plot 3D validation metrics")

    parser.add_argument(
        "--metric_dir",
        type=Path,
        required=True,
        help="Result directory containing iterXXX folders."
    )

    parser.add_argument(
        "--metric_name",
        type=str,
        required=True,
        choices=["dice_3d_val", "hd95_val", "assd_val"],
        help="Metric to plot."
    )

    parser.add_argument(
        "--ylabel",
        type=str,
        required=True
    )

    parser.add_argument(
        "--title",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dest",
        type=Path
    )

    parser.add_argument(
        "--headless",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())