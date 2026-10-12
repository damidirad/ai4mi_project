import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "Background",
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}


def load_metrics(metric_dir: Path, metric_name: str):
    """
    Load per-epoch metric files.

    Each file is expected to contain:
        (patients, classes)

    Returns:
        epochs:       (epochs,)
        all_metrics:  (epochs, patients, classes)
    """

    metric_files = sorted(
        list(metric_dir.glob(f"iter*/{metric_name}.npy"))
        + list(metric_dir.glob(f"iter*/{metric_name}.npz"))
    )

    if not metric_files:
        raise FileNotFoundError(
            f"No {metric_name}.npy or {metric_name}.npz files found in {metric_dir}"
        )

    epochs = []
    all_metrics = []

    for file in metric_files:
        epoch = int(file.parent.name.replace("iter", ""))

        data = np.load(file, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                metrics = data[data.files[0]]
            finally:
                data.close()
        else:
            metrics = data

        if metrics.ndim != 2:
            raise ValueError(
                f"{file} has shape {metrics.shape}; "
                "expected (patients, classes)."
            )

        if metrics.shape[1] != 5:
            raise ValueError(
                f"{file} contains {metrics.shape[1]} classes; "
                "expected 5."
            )

        epochs.append(epoch)
        all_metrics.append(metrics)

    order = np.argsort(epochs)

    epochs = np.asarray(epochs)[order]
    all_metrics = np.stack(all_metrics, axis=0)[order]

    return epochs, all_metrics


def calculate_statistics(metrics):
    """
    Calculate mean and standard deviation across patients.

    Input:
        metrics: (epochs, patients, classes)

    Returns:
        means: (epochs, classes)
        stds:  (epochs, classes)

    Any NaN/inf makes that epoch/class mean and SD undefined (NaN).
    Zero distances, including correctly absent organs, remain valid values.
    """

    n_epochs = metrics.shape[0]
    n_classes = metrics.shape[2]

    means = np.full(
        (n_epochs, n_classes),
        np.nan,
        dtype=np.float64,
    )

    stds = np.full(
        (n_epochs, n_classes),
        np.nan,
        dtype=np.float64,
    )

    for epoch_idx in range(n_epochs):
        for class_idx in range(n_classes):

            values = metrics[epoch_idx, :, class_idx]

            if values.size == 0 or not np.isfinite(values).all():
                continue
            means[epoch_idx, class_idx] = values.mean()
            stds[epoch_idx, class_idx] = values.std(ddof=1) if values.size > 1 else 0.0

    return means, stds


def plot_all_classes(
    epochs,
    means,
    stds,
    ylabel,
    title,
    dest=None,
):
    """
    Plot all foreground classes together.

    Each line is the mean across patients.
    The shaded region is mean ± 1 standard deviation.
    """

    fig, ax = plt.subplots(figsize=(11, 7))

    for class_idx in range(1, 5):

        mean = means[:, class_idx]
        std = stds[:, class_idx]

        ax.plot(
            epochs,
            mean,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Class {class_idx}: {CLASS_NAMES[class_idx]}",
        )

        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend()

    if not np.isfinite(means[:, 1:]).all():
        fig.text(0.5, 0.01, 'Gaps indicate undefined means; no patients are excluded.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()

    if dest is not None:
        fig.savefig(
            dest,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def plot_separate_classes(
    epochs,
    means,
    stds,
    ylabel,
    title,
    dest=None,
):
    """
    Create one subplot per foreground class.

    Each subplot shows:
        mean across patients
        ±1 standard deviation
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
    )

    axes = axes.ravel()

    for plot_idx, class_idx in enumerate(range(1, 5)):

        ax = axes[plot_idx]

        mean = means[:, class_idx]
        std = stds[:, class_idx]

        ax.plot(
            epochs,
            mean,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.2,
        )

        ax.set_title(
            f"Class {class_idx}: {CLASS_NAMES[class_idx]}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{title} — Mean ± 1 SD Across Patients",
        fontsize=14,
    )

    if not np.isfinite(means[:, 1:]).all():
        fig.text(0.5, 0.01, 'Gaps indicate undefined means; no patients are excluded.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()

    if dest is not None:
        separate_dest = dest.with_name(
            f"{dest.stem}_separate{dest.suffix}"
        )

        fig.savefig(
            separate_dest,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def run(args: argparse.Namespace) -> None:

    epochs, metrics = load_metrics(
        args.metric_dir,
        args.metric_name,
    )

    means, stds = calculate_statistics(metrics)

    # ---------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------

    print("\nMean ± SD across patients:")
    print(
        f"{'Epoch':>8} "
        f"{'Esophagus':>22} "
        f"{'Heart':>22} "
        f"{'Trachea':>22} "
        f"{'Aorta':>22}"
    )

    for epoch_idx, epoch in enumerate(epochs):

        values = means[epoch_idx]
        deviations = stds[epoch_idx]

        cells = []
        for class_idx in range(1, 5):
            raw = metrics[epoch_idx, :, class_idx]
            undefined = int((~np.isfinite(raw)).sum())
            cells.append(f"undefined ({undefined}/{raw.size})" if undefined else
                         f"{values[class_idx]:.4f} ± {deviations[class_idx]:.4f}")
        print(f"{epoch:>8} " + '   '.join(cells))

    # ---------------------------------------------------------
    # Combined plot
    # ---------------------------------------------------------

    plot_all_classes(
        epochs=epochs,
        means=means,
        stds=stds,
        ylabel=args.ylabel,
        title=args.title,
        dest=args.dest,
    )

    # ---------------------------------------------------------
    # Separate class plots
    # ---------------------------------------------------------

    plot_separate_classes(
        epochs=epochs,
        means=means,
        stds=stds,
        ylabel=args.ylabel,
        title=args.title,
        dest=args.dest,
    )

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Plot per-class 3D validation metrics."
    )

    parser.add_argument(
        "--metric_dir",
        type=Path,
        required=True,
        help="Result directory containing iterXXX folders.",
    )

    parser.add_argument(
        "--metric_name",
        type=str,
        required=True,
        choices=[
            "dice_3d_val",
            "hd95_val",
            "assd_val",
        ],
        help="Metric to plot.",
    )

    parser.add_argument(
        "--ylabel",
        type=str,
        required=True,
        help="Y-axis label.",
    )

    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Plot title.",
    )

    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output path for the combined plot.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not display plots.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
