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
import json
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

from dataset import build_dataset
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

from losses import (CrossEntropy, DiceLoss, CEDiceLoss, BoundaryLoss, CEDiceBoundaryLoss)
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
    if getattr(args, 'tiling', False):
        raise ValueError('Use runTraining for the dedicated tiled training route')
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
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    else:
        raise ValueError(args.optimizer)

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    slices: int = args.slices
    root_dir = args.data_root / args.dataset



    train_set = build_dataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             tiling=getattr(args, 'tiling', False),
                             slices=slices,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.workers,
                              shuffle=True)


    weights = None
    if getattr(args, 'class_weighting', 'inverse_frequency') != 'none':
        # Calculate class frequencies from the training ground truth
        class_counts = torch.zeros(K, dtype=torch.float64)

        for data in train_loader:
            gt = data['gts']  # [B, K, W, H]
            class_counts += gt.sum(dim=(0, 2, 3)).double()

        class_freq = class_counts / class_counts.sum()

        # Inverse square-root frequency
        weights = 1.0 / torch.sqrt(class_freq + 1e-10)

        # Normalize so the average weight is 1
        weights = weights / weights.mean()

        print(">> Training class counts:", class_counts.tolist())
        print(">> Training class frequencies:", class_freq.tolist())
        print(">> Class weights:", weights.tolist())

    val_set = build_dataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           tiling=getattr(args, 'tiling', False),
                           slices=slices,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K, None if weights is None else weights.float())


def _write_training_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _save_training_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _setup_tiled_training(args):
    """Validate tiled inputs and prepare components for the shared training loop."""
    from evaluate_tiled import reference_image
    from losses import validate_tiled_loss, inverse_frequency_weights, make_tiled_loss

    validate_tiled_loss(args.loss)
    if args.epochs <= 0 or args.workers < 0:
        raise ValueError('Require positive epochs and nonnegative workers')
    destination = Path(args.dest)
    if destination.exists():
        raise FileExistsError('Tiled training requires a new destination')
    params = datasets_params[args.dataset]
    if params['K'] != 5:
        raise ValueError('Tiled training requires the five-class SegTHOR configuration')
    if args.mode == 'full':
        supervised = list(range(5))
    elif args.mode == 'partial' and args.dataset == 'SEGTHOR':
        supervised = [0, 1, 3, 4]
    else:
        raise ValueError('Unsupported tiled supervision mode')
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            raise ValueError('--gpu requested but no GPU is available')
    else:
        device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data_root) / args.dataset
    common = dict(tiling=True, img_transform=img_transform, gt_transform=partial(gt_transform, 5), slices=args.slices)
    train = build_dataset('train', root, debug=args.debug, **common)
    validation = build_dataset('val', root, debug=False, **common)
    if train.preprocessing_signature != validation.preprocessing_signature:
        raise ValueError('Training and validation preprocessing configurations differ')
    if set(train.records) & set(validation.records):
        raise ValueError('Training and validation patients must be disjoint')
    if any(size % 8 for size in train.tile_size):
        raise ValueError('ENet tile dimensions must be multiples of 8')
    # Fail before training if original CT references needed by validation are unavailable.
    for record in validation.records.values():
        reference_image(record)
        if not Path(record['source_ct']).with_name('GT.nii.gz').is_file():
            raise ValueError('Original validation GT.nii.gz is required')
    counts = train.class_counts()
    weights = (torch.ones(5) if getattr(args, 'class_weighting', 'inverse_frequency') == 'none'
               else inverse_frequency_weights(counts))
    criterion = make_tiled_loss(train, args.loss, supervised, weights, counts)
    val_counts = validation.class_counts()
    val_criterion = make_tiled_loss(validation, args.loss, supervised, weights, val_counts)
    loader = DataLoader(train, batch_size=params['B'], num_workers=args.workers, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation, batch_size=params['B'], num_workers=args.workers, shuffle=False)
    net_class = ENet_2_5d if args.slices > 1 else params['net']
    model = net_class(1, 5, kernels=params.get('kernels', 8), factor=params.get('factor', 2))
    model.init_weights()
    model.to(device)
    optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[args.optimizer]
    optimizer = optimizer_class(model.parameters(), lr=.0005, betas=(.9, .999),
                                weight_decay=.01 if args.optimizer == 'adamw' else 0)
    if args.scheduler not in ('none', 'cosine'):
        raise ValueError('Unknown scheduler')
    destination.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'device': str(device), 'torch_version': str(torch.__version__),
                   'class_counts': counts.tolist(), 'class_weights': weights.tolist(),
                   'supervised_classes': supervised, 'sampling': 'uniform_tile_entries',
                   'checkpoint_metric': 'original_CT_mean_foreground_dice',
                   'preprocessing_signature': train.preprocessing_signature,
                   'train_metadata': train.records, 'val_metadata': validation.records})
    _write_training_json(destination / 'run.json', config)
    return model, optimizer, device, loader, val_loader, 5, weights, criterion, val_criterion


def runTraining(args):
    tiling = getattr(args, 'tiling', False)
    if tiling:
        from evaluate_tiled import evaluate_model
        net, optimizer, device, train_loader, val_loader, K, weights, loss_fn, val_loss_fn = _setup_tiled_training(args)
        destination = Path(args.dest)
        history, best_epoch = [], None
    else:
        print(f">>> Setting up to train on {args.dataset} with {args.mode}")
        net, optimizer, device, train_loader, val_loader, K, weights = setup(args)

    original_records = None
    if not tiling and getattr(args, 'original_grid_validation', False):
        from evaluate_tiled import load_full_metadata, evaluate_full_predictions, reference_image
        if args.debug or K != 5:
            raise ValueError('Original-grid validation requires complete five-class patient volumes')
        original_records = load_full_metadata(args.data_root / args.dataset / 'val' / 'metadata', 'val')
        for record in original_records.values():
            reference_image(record)
            if not Path(record['source_ct']).with_name('GT.nii.gz').is_file():
                raise ValueError('Original validation labels are required')

    # Decays the LR from its initial value to 0 over all epochs; stepped once per epoch
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f">>> Train dataset size: {len(train_loader.dataset)}")
    print(f">>> Val dataset size: {len(val_loader.dataset)}")
    print(f">>> Train batches: {len(train_loader)}")
    print(f">>> Val batches: {len(val_loader)}")

    if not tiling:
        with open(args.data_root / args.dataset / "spacing.pkl", "rb") as f:
            spacing_dict = pickle.load(f)

        if args.mode == "full":
            idk = list(range(K))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
            idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)

        match args.loss:
            case 'ce':
                loss_fn = CrossEntropy(idk=idk, weights=weights)
            case 'dice':
                loss_fn = DiceLoss(idk=idk)
            case 'ce_dice':
                loss_fn = CEDiceLoss(idk=idk, weights=weights)
            case 'boundary':
                loss_fn = BoundaryLoss(idk=idk)
            case 'ce_dice_boundary':
                loss_fn = CEDiceBoundaryLoss(idk=idk, weights=weights)

        # Notice one has the length of the _loader_, and the other one of the _dataset_
        log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
        log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
        log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
        log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
        log_h95_val: Tensor = torch.zeros((args.epochs,))
        log_assd_val: Tensor = torch.zeros((args.epochs,))
        log_lr: Tensor = torch.zeros((args.epochs,))  # LR used during each epoch

    else:
        log_lr = torch.zeros(args.epochs)

    best_dice: float = -1

    for e in range(args.epochs):
        log_lr[e] = optimizer.param_groups[0]['lr']
        print(f">> Learning rate for epoch {e}: {log_lr[e]:.2e}")

        if args.loss == 'ce_dice_boundary':
            # grows by a fixed step per epoch, independent of the total number of epochs
            loss_fn.boundary_weight = min(args.boundary_weight_step * (e + 1), args.boundary_weight_max)
            print(f">> Boundary weight for epoch {e}: {loss_fn.boundary_weight:.3f}")


        tile_losses = {}
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    if not tiling:
                        log_loss = log_loss_tra
                        log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.inference_mode if tiling else torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    if not tiling:
                        log_loss = log_loss_val
                        log_dice = log_dice_val

                    # Collect validation slices for 3D Dice
                    patient_preds = {}
                    patient_gts = {}

                    # Collecting hd95 and assd scores
                    hd95_scores = []
                    assd_scores = []

            tile_total, tile_samples = 0., 0
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                if not tiling and m == 'val':
                    val_dest = args.dest / f"iter{e:03d}" / m
                    if val_dest.exists():
                        rmtree(val_dest)

                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    if tiling:
                        criterion = loss_fn if m == 'train' else val_loss_fn
                        loss = criterion(pred_logits, gt, data['pixel_weights'].to(device))
                        if not torch.isfinite(loss):
                            raise ValueError(f'Nonfinite tiled {m} loss')
                    else:
                        pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                        # Metrics computation, not used for training
                        pred_seg = probs2one_hot(pred_probs)

                        if args.debug and m == 'val' and i == 0:
                            print(
                                f">>> DEBUG epoch {e}: "
                                f"pred pixel counts = "
                                f"{pred_seg.sum(dim=(0, 2, 3)).detach().cpu().tolist()}"
                            )
                            print(
                                f">>> DEBUG epoch {e}: "
                                f"gt pixel counts = "
                                f"{gt.sum(dim=(0, 2, 3)).detach().cpu().tolist()}"
                            )
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
                        if tiling and any(p.grad is not None and not torch.isfinite(p.grad).all()
                                          for p in net.parameters()):
                            raise ValueError('Nonfinite tiled gradients')
                        opt.step()

                    if tiling:
                        tile_total += loss.item() * B
                        tile_samples += B
                        tq_iter.set_postfix({'Loss': tile_total / tile_samples})
                        continue

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

            if tiling:
                tile_losses[m] = tile_total / tile_samples

        if tiling:
            report = evaluate_model(net, val_loader.dataset, destination / f'epoch_{e:03d}',
                                    train_loader.batch_size, device)
            score = report['mean_foreground_dice']
            history.append({'epoch': e, 'lr': optimizer.param_groups[0]['lr'],
                            'train_ce': tile_losses['train'], 'val_ce': tile_losses['val'],
                            'mean_foreground_dice': score})
            if score > best_dice:
                best_dice, best_epoch = score, e
                _save_training_checkpoint(destination / 'bestweights.pt', net.state_dict())
            if scheduler is not None:
                scheduler.step()
            _save_training_checkpoint(destination / 'last.pt', {
                'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': None if scheduler is None else scheduler.state_dict(),
                'epoch': e, 'slices': args.slices,
                'preprocessing_signature': train_loader.dataset.preprocessing_signature})
            _write_training_json(destination / 'history.json', history)
            print(f"Epoch {e+1}/{args.epochs}: CE={tile_losses['train']:.6f}, original-grid Dice={score:.6f}", flush=True)
            continue

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
                   
                # DEBUG: recompute 2D Dice for this patient's slices
                if args.debug and patient_id == sorted(patient_preds)[0]:
                    patient_slice_dice = []

                    for sid in slice_ids:
                        d = dice_coef(
                            patient_preds[patient_id][sid].unsqueeze(0),
                            patient_gts[patient_id][sid].unsqueeze(0)
                        )
                        patient_slice_dice.append(d.squeeze(0))

                    patient_slice_dice = torch.stack(patient_slice_dice)

                    print(
                        f">>> DEBUG {patient_id} 2D Dice mean: "
                        f"{patient_slice_dice.mean(dim=0).tolist()}"
                    )
                    print(
                        f">>> DEBUG {patient_id} 2D Dice foreground: "
                        f"{patient_slice_dice[:, 1:].mean(dim=0).tolist()}"
                    )

                if args.debug and patient_id == sorted(patient_preds)[0]:
                    stored_pred_counts = torch.zeros(5, dtype=torch.long)

                    for sid in slice_ids:
                        stored_pred = patient_preds[patient_id][sid]
                        stored_pred_counts += stored_pred.sum(dim=(1, 2)).long()

                    print(
                        f">>> DEBUG {patient_id} stored pred voxels: "
                        f"{stored_pred_counts.tolist()}"
                    )

                if args.debug and patient_id == sorted(patient_preds)[0]:
                    print(f">>> DEBUG reconstructed {patient_id}")
                    print(f"    slices: {slice_ids[0]} -> {slice_ids[-1]} ({len(slice_ids)} slices)")
                    print(f"    pred shape: {pred_volume.shape}")
                    print(f"    gt shape:   {gt_volume.shape}")
                    print(f"    pred voxels: {pred_volume.sum(dim=(0, 2, 3, 4)).tolist()}")
                    print(f"    gt voxels:   {gt_volume.sum(dim=(0, 2, 3, 4)).tolist()}")

                patient_dice = dice_3d(gt_volume, pred_volume)
                patient_hd95 = hausdorff95(gt_volume, pred_volume, spacing=spacing)
                patient_assd = assd(gt_volume, pred_volume, spacing=spacing)
                if args.debug and patient_id == sorted(patient_preds)[0]:
                    print(
                        f">>> DEBUG metrics {patient_id}: "
                        f"Dice={patient_dice.squeeze(0).tolist()}"
                    )
                    print(
                        f">>> DEBUG metrics {patient_id}: "
                        f"HD95={patient_hd95.squeeze(0).tolist()}"
                    )
                    print(
                        f">>> DEBUG metrics {patient_id}: "
                        f"ASSD={patient_assd.squeeze(0).tolist()}"
                    )

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

            if args.debug:
                print(
                    f">>> DEBUG epoch {e}: "
                    f"HD95 finite={torch.isfinite(hd95_scores[:, 1:]).sum().item()} / "
                    f"{hd95_scores[:, 1:].numel()}"
                )
                print(
                    f">>> DEBUG epoch {e}: "
                    f"ASSD finite={torch.isfinite(assd_scores[:, 1:]).sum().item()} / "
                    f"{assd_scores[:, 1:].numel()}"
                )

            finite_hd95 = hd95_scores[:, 1:][torch.isfinite(hd95_scores[:, 1:])]
            finite_assd = assd_scores[:, 1:][torch.isfinite(assd_scores[:, 1:])]

            mean_hd95 = finite_hd95.mean().item()
            mean_assd = finite_assd.mean().item()

            log_h95_val[e] = mean_hd95
            log_assd_val[e] = mean_assd

            if args.debug:
                print(f"\n>>> DEBUG SUMMARY epoch {e}")
                print(f"    3D Dice shape:  {dice_3d_scores.shape}")
                print(f"    3D Dice values:\n{dice_3d_scores}")
                print(
                    f"    HD95 finite: "
                    f"{torch.isfinite(hd95_scores).sum().item()} / "
                    f"{hd95_scores.numel()}"
                )
                print(
                    f"    ASSD finite: "
                    f"{torch.isfinite(assd_scores).sum().item()} / "
                    f"{assd_scores.numel()}"
                )

            print(
                f">>> 3D metrics at epoch {e}: "
                f"3D Dice={current_dice:05.3f}, "
                f"HD95={mean_hd95:05.2f} mm, "
                f"ASSD={mean_assd:05.2f} mm"
            )

        if scheduler is not None:
            scheduler.step()

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "lr.npy", log_lr)
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "h95_val.npy", log_h95_val)
        np.save(args.dest / "assd_val.npy", log_assd_val)

        if original_records is not None:
            def original_predictions():
                if set(patient_preds) != set(original_records):
                    raise ValueError('Validation patients do not match metadata')
                for pid, record in original_records.items():
                    depth = record['output']['shape'][2]
                    if set(patient_preds[pid]) != set(range(depth)):
                        raise ValueError('Missing validation slices')
                    yield pid, np.stack([patient_preds[pid][z].argmax(0).numpy().astype(np.uint8)
                                         for z in range(depth)], axis=2)
            report = evaluate_full_predictions(original_records, original_predictions(),
                                               args.dest / f'original_epoch_{e:03d}')
            current_dice = report['mean_foreground_dice']
            print(f'Original-grid validation Dice: {current_dice:.6f}', flush=True)

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

    if tiling:
        net.load_state_dict(torch.load(destination / 'bestweights.pt', map_location=device, weights_only=True))
        final = evaluate_model(net, val_loader.dataset, destination / 'best_predictions',
                               train_loader.batch_size, device)
        _write_training_json(destination / 'summary.json', {
            'complete': True, 'best_epoch': best_epoch, 'best_validation_dice': best_dice,
            'reloaded_best_dice': final['mean_foreground_dice'],
            'epochs': args.epochs, 'checkpoint': 'bestweights.pt'})
        return history


def check_data(args):
    """Load a batch per split without creating a model, optimizer or outputs."""
    K = datasets_params[args.dataset]['K']
    if args.tiling and K != 5:
        raise ValueError('Tiled data currently uses the five-class SegTHOR encoding')
    check_loss = getattr(args, 'check_loss', False)
    if check_loss and not args.tiling:
        raise ValueError('--check-loss requires --tiling')
    if check_loss:
        from losses import validate_tiled_loss, make_tiled_loss, inverse_frequency_weights
        validate_tiled_loss(args.loss)
        if args.mode == 'full':
            idk = list(range(K))
        elif args.mode == 'partial' and args.dataset == 'SEGTHOR':
            idk = [0, 1, 3, 4]
        else:
            raise ValueError('Unsupported tiled supervision mode')
    sizes = []
    signatures = []
    for split in ('train', 'val'):
        dataset = build_dataset(split, args.data_root / args.dataset,
            tiling=args.tiling, img_transform=img_transform,
            gt_transform=partial(gt_transform, K), slices=args.slices, debug=args.debug)
        if len(dataset) == 0:
            raise ValueError(f'Empty {split} dataset')
        batch = next(iter(DataLoader(dataset, batch_size=datasets_params[args.dataset]['B'],
                                    num_workers=args.workers, shuffle=False)))
        sizes.append(tuple(batch['images'].shape[-2:]))
        if args.tiling:
            signatures.append(dataset.preprocessing_signature)
        print(f"{split}: {len(dataset)} samples; images={tuple(batch['images'].shape)}, "
              f"labels={tuple(batch['gts'].shape)}")
        if check_loss:
            counts = dataset.class_counts()
            if split == 'train':
                class_weights = inverse_frequency_weights(counts)
            criterion = make_tiled_loss(dataset, args.loss, idk, class_weights, counts=counts)
            # Diagnostic logits only: no model, optimizer or training is run.
            logits = torch.zeros_like(batch['gts'], dtype=torch.float32, requires_grad=True)
            loss = criterion(logits, batch['gts'], batch['pixel_weights'])
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():
                raise ValueError('Nonfinite tiled loss or gradients')
            padded = (~batch['valid_mask']).unsqueeze(1).expand_as(logits)
            if torch.any(logits.grad[padded] != 0):
                raise ValueError('Padding contributed to tiled gradients')
            print(f'{split}: tiled CE={loss.item():.6f}; padding gradients zero')
    if args.tiling and signatures[0] != signatures[1]:
        raise ValueError('Training and validation preprocessing configurations differ')
    if sizes[0] != sizes[1]:
        raise ValueError('Training and validation spatial sizes must match')


def evaluate_tiled_checkpoint(args):
    from evaluate_tiled import evaluate_model
    if args.dest.exists():
        raise FileExistsError('Use a new evaluation destination')
    if args.debug:
        raise ValueError('Volume evaluation requires complete patients; disable --debug')
    params = datasets_params[args.dataset]
    if params['K'] != 5:
        raise ValueError('Tiled evaluation requires the five-class SegTHOR model')
    dataset = build_dataset(getattr(args, 'split', None) or 'val', args.data_root / args.dataset, tiling=getattr(args, 'tiling', True),
        img_transform=img_transform, gt_transform=partial(gt_transform, 5), slices=args.slices)
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            raise ValueError('--gpu requested but no GPU is available')
    else:
        device = torch.device('cpu')
    model_class = ENet_2_5d if args.slices > 1 else params['net']
    model = model_class(1, 5, kernels=params.get('kernels', 8), factor=params.get('factor', 2)).to(device)
    state = torch.load(args.evaluate_checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if getattr(args, 'tiling', True):
        report = evaluate_model(model, dataset, args.dest, batch_size=params['B'], device=device)
    else:
        from evaluate_tiled import load_full_metadata, evaluate_full_model
        split = getattr(args, 'split', None) or 'val'
        records = load_full_metadata(args.data_root / args.dataset / split / 'metadata', split)
        report = evaluate_full_model(model, dataset, records, args.dest, batch_size=params['B'], device=device)
    if dataset.test_mode:
        print(f"Saved predictions for {len(report['patients'])} test patients to {args.dest}")
    else:
        print(f"Original-grid mean foreground Dice: {report['mean_foreground_dice']:.6f}")
    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='ce', choices=['ce', 'dice', 'ce_dice', 'boundary', 'ce_dice_boundary'],
                        help="Loss function: cross-entropy (baseline), Dice, CE+Dice, "
                             "boundary (distance-map based), or CE+Dice+boundary.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--data_root', type=Path, default=Path('data'),
                        help="Parent directory containing the selected dataset directory.")
    parser.add_argument('--workers', default=5, type=int,
                        help="Number of DataLoader worker processes.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed for reproducible model initialization and shuffling.")
    parser.add_argument('--slices', default=1, type=int,
                        help="Positive odd number of slices per sample: 1 uses 2D; >1 uses ENet_2_5d.")

    parser.add_argument('--class-weighting', choices=('none', 'inverse_frequency'), default='inverse_frequency',
                        help='Use none to reproduce the original unweighted CE objective')
    parser.add_argument('--boundary-weight-step', default=0.01, type=float,
                        help='ce_dice_boundary: boundary weight added per epoch (weight = step * (epoch + 1))')
    parser.add_argument('--boundary-weight-max', default=1.0, type=float,
                        help='ce_dice_boundary: maximum boundary weight')
    parser.add_argument('--original-grid-validation', action='store_true',
                        help='Select full-slice checkpoints using original-CT volume Dice (requires metadata)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw'],
                        help="adam: original setup; adamw: Adam with decoupled weight decay (1e-2).")
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine'],
                        help="none: fixed LR; cosine: CosineAnnealingLR from the initial LR to 0 over all epochs.")

    parser.add_argument('--tiling', action='store_true', help='Use tiles for training and reconstruct full volumes for validation')
    parser.add_argument('--check-data', action='store_true',
                        help='Check train/val batches without training or writing results')
    parser.add_argument('--evaluate-checkpoint', type=Path,
                        help='Predict on original CT grids from a state_dict; requires preprocessing metadata')
    parser.add_argument('--split', choices=('val', 'test'),
                        help='Checkpoint prediction split (default: val); test requires no labels')
    parser.add_argument('--check-loss', action='store_true',
                        help='With --tiling --check-data: check CE and gradients on diagnostic logits')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()
    if args.split is not None and not args.evaluate_checkpoint:
        parser.error('--split requires --evaluate-checkpoint')
    if args.slices <= 0 or args.slices % 2 == 0:
        parser.error('--slices must be a positive odd integer (1 for 2D; 3, 5, ... for 2.5D)')
    if args.boundary_weight_step <= 0 or args.boundary_weight_max <= 0:
        parser.error('--boundary-weight-step and --boundary-weight-max must be positive')
    if args.evaluate_checkpoint and (args.check_data or args.check_loss):
        parser.error('--evaluate-checkpoint cannot combine with check modes')
    if args.check_loss and not (args.tiling and args.check_data):
        parser.error('--check-loss requires --tiling --check-data')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pprint(args)

    if args.evaluate_checkpoint:
        evaluate_tiled_checkpoint(args)
    elif args.check_data:
        check_data(args)
    else:
        runTraining(args)


if __name__ == '__main__':
    main()
