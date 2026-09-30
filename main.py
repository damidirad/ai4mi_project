#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import random
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from ENet_2_5d import ENet_2_5d
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   dice_3d,
                   hausdorff95,
                   assd)

from losses import (CrossEntropy)
import pickle

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

datasets_params["SEGTHOR_hu"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_resampled"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_spatial"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_hu_spatial"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    use_gpu = args.gpu

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net_cls = ENet_2_5d if args.slices > 1 else datasets_params[args.dataset]['net']
    net = net_cls(1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    slices: int = args.slices
    root_dir = args.data_root / args.dataset



    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             slices=slices,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           slices=slices,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)
    with open(args.data_path / "spacing.pkl", "rb") as f:
        spacing_dict = pickle.load(f)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

                    if val_dest.exists(): 
                        rmtree(val_dest)

                    # Collect validation slices for 3D Dice
                    patient_preds = {}
                    patient_gts = {}

                    # Collecting hd95 and assd scores
                    hd95_scores = []
                    assd_scores = []

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                if m == 'val':
                    val_dest = args.dest / f"iter{e:03d}" / m
                    if val_dest.exists():
                        rmtree(val_dest)

                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    if args.debug:
                        print(f">> {m} batch {i}: {img.shape=}, {data['stems']=}, {data['gts'].shape=}")

                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    if m == 'val':
                        for b, stem in enumerate(data['stems']):
                            patient_id, slice_id = str(stem).rsplit("_", 1)
                            slice_id = int(slice_id)

                            if patient_id not in patient_preds:
                                patient_preds[patient_id] = {}
                                patient_gts[patient_id] = {}

                            patient_preds[patient_id][slice_id] = pred_seg[b].cpu()
                            patient_gts[patient_id][slice_id] = gt[b].cpu()


                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        if m == 'val':
            dice_3d_scores = []

            for patient_id in sorted(patient_preds):
                slice_ids = sorted(patient_preds[patient_id])
                dx, dy, dz = spacing_dict[patient_id]
                spacing = (dz, dx, dy)

                pred_volume = torch.stack(
                    [patient_preds[patient_id][sid] for sid in slice_ids],
                    dim=0
                )

                gt_volume = torch.stack(
                    [patient_gts[patient_id][sid] for sid in slice_ids],
                    dim=0
                )

                # [D, K, H, W] -> [1, K, D, H, W]
                pred_volume = pred_volume.permute(1, 0, 2, 3).unsqueeze(0)
                gt_volume = gt_volume.permute(1, 0, 2, 3).unsqueeze(0)

                patient_dice = dice_3d(gt_volume, pred_volume)
                patient_hd95 = hausdorff95(gt_volume, pred_volume, spacing=spacing)
                patient_assd = assd(gt_volume, pred_volume, spacing=spacing)

                dice_3d_scores.append(patient_dice.squeeze(0))
                hd95_scores.append(patient_hd95.squeeze(0))
                assd_scores.append(patient_assd.squeeze(0))

            dice_3d_scores = torch.stack(dice_3d_scores)
            hd95_scores = torch.stack(hd95_scores)
            assd_scores = torch.stack(assd_scores)

            # Save [N_patients, K] for this epoch
            np.save(
                args.dest / f"iter{e:03d}" / "dice_3d_val.npy",
                dice_3d_scores.numpy()
            )
            np.save(
                args.dest / f"iter{e:03d}" / "hd95_val.npy",
                hd95_scores.numpy()
            )
            np.save(
                args.dest / f"iter{e:03d}" / "assd_val.npy",
                assd_scores.numpy()
            )

            # Mean foreground 3D Dice, ignoring background
            current_dice = dice_3d_scores[:, 1:].mean().item()

            finite_hd95 = hd95_scores[:, 1:][torch.isfinite(hd95_scores[:, 1:])]
            finite_assd = assd_scores[:, 1:][torch.isfinite(assd_scores[:, 1:])]

            mean_hd95 = finite_hd95.mean().item()
            mean_assd = finite_assd.mean().item()

            print(
                f">>> 3D metrics at epoch {e}: "
                f"3D Dice={current_dice:05.3f}, "
                f"HD95={mean_hd95:05.2f} mm, "
                f"ASSD={mean_assd:05.2f} mm"
            )

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        if current_dice > best_dice:
            message = f">>> Improved 3D dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--data_root', type=Path, default=Path('data'),
                        help="Parent directory containing the selected dataset directory.")
    parser.add_argument('--workers', default=5, type=int,
                        help="Number of DataLoader worker processes.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed for reproducible model initialization and shuffling.")
    parser.add_argument('--slices', default=1, type=int,
                        help="Neighbouring slices per sample; >1 uses ENet_2_5d.")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
