import argparse
import sys
from pathlib import Path
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Direct script execution adds scripts/ to sys.path, so expose the project modules.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import SliceDataset
from utils import class2one_hot


datasets_params: dict[str, dict[str, Any]] = {}

datasets_params["TOY2"] = {'K': 2, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}

datasets_params["SEGTHOR_hu"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_resampled"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_spatial"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_hu_spatial"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
    img = img.convert('L')
    img = np.array(img)[np.newaxis, ...]
    img = img / 255
    return torch.tensor(img, dtype=torch.float32)


def gt_transform(K, img):
    img = np.array(img)[...]

    if K != 5:
        img = img / (255 / (K - 1))
    else:
        img = img / 63

    img = torch.tensor(img, dtype=torch.int64)[None, ...]
    img = class2one_hot(img, K=K)

    return img[0]


def plot_class_imbalance(args):

    K = datasets_params[args.dataset]['K']
    B = datasets_params[args.dataset]['B']
    root_dir = args.data_root / args.dataset

    train_set = SliceDataset(
        'train',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        slices=args.slices,
        debug=args.debug
    )

    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=args.workers,
        shuffle=False
    )

    # Count pixels for each class
    class_counts = torch.zeros(K, dtype=torch.float64)

    for data in train_loader:
        gt = data['gts']
        class_counts += gt.sum(dim=(0, 2, 3)).double()

    # Convert to percentages
    class_freq = class_counts / class_counts.sum() * 100

    classes = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]

    print("Class counts:")
    for name, count, freq in zip(classes, class_counts, class_freq):
        print(f"  {name:10s}: {count.item():,.0f} ({freq.item():.4f}%)")

    # Plot
    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=classes,
        y=class_freq.numpy()
    )

    plt.yscale("log")
    plt.ylabel("Pixel frequency (%)")
    plt.xlabel("Class")
    plt.title("SegTHOR Class Imbalance")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        default='TOY2',
        choices=datasets_params.keys()
    )

    parser.add_argument(
        '--data_root',
        type=Path,
        default=Path('data')
    )

    parser.add_argument(
        '--dest',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--slices',
        default=1,
        type=int
    )

    parser.add_argument(
        '--workers',
        default=4,
        type=int
    )

    parser.add_argument(
        '--debug',
        action='store_true'
    )

    args = parser.parse_args()

    plot_class_imbalance(args)


if __name__ == '__main__':
    main()