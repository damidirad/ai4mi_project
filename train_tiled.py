"""Optional tiled training: corrected CE and original-CT volume validation."""
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_dataset
from evaluate_tiled import evaluate_model, reference_image
from tiled_losses import validate_tiled_loss, inverse_frequency_weights, make_tiled_loss


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def run_tiled_training(args):
    # Runtime import keeps the entry point's existing model/transform choices.
    from main import datasets_params, img_transform, gt_transform, ENet_2_5d
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
    weights = inverse_frequency_weights(counts)
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
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler != 'none':
        raise ValueError('Unknown scheduler')
    destination.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'device': str(device), 'torch_version': str(torch.__version__),
                   'class_counts': counts.tolist(), 'class_weights': weights.tolist(),
                   'supervised_classes': supervised, 'sampling': 'uniform_tile_entries',
                   'checkpoint_metric': 'original_CT_mean_foreground_dice',
                   'preprocessing_signature': train.preprocessing_signature,
                   'train_metadata': train.records, 'val_metadata': validation.records})
    _json(destination / 'run.json', config)
    history = []
    best, best_epoch = -1., None
    for epoch in range(args.epochs):
        model.train()
        total, samples = 0., 0
        lr = optimizer.param_groups[0]['lr']
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch['images'].to(device))
            loss = criterion(logits, batch['gts'].to(device), batch['pixel_weights'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite tiled training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite tiled gradients')
            optimizer.step()
            size = logits.shape[0]
            total += loss.item() * size
            samples += size
        model.eval()
        val_total, val_samples = 0., 0
        with torch.inference_mode():
            for batch in val_loader:
                logits = model(batch['images'].to(device))
                loss = val_criterion(logits, batch['gts'].to(device), batch['pixel_weights'].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite tiled validation loss')
                val_total += loss.item() * logits.shape[0]
                val_samples += logits.shape[0]
        report = evaluate_model(model, validation, destination / f'epoch_{epoch:03d}', params['B'], device)
        score = report['mean_foreground_dice']
        history.append({'epoch': epoch, 'lr': lr, 'train_ce': total/samples,
                        'val_ce': val_total/val_samples, 'mean_foreground_dice': score})
        if score > best:
            best, best_epoch = score, epoch
            _save(destination / 'bestweights.pt', model.state_dict())
        if scheduler is not None:
            scheduler.step()
        _save(destination / 'last.pt', {'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(), 'scheduler': None if scheduler is None else scheduler.state_dict(),
              'epoch': epoch, 'slices': args.slices, 'preprocessing_signature': train.preprocessing_signature})
        _json(destination / 'history.json', history)
        print(f'Epoch {epoch+1}/{args.epochs}: CE={total/samples:.6f}, original-grid Dice={score:.6f}', flush=True)
    model.load_state_dict(torch.load(destination / 'bestweights.pt', map_location=device, weights_only=True))
    final = evaluate_model(model, validation, destination / 'best_predictions', params['B'], device)
    _json(destination / 'summary.json', {'complete': True, 'best_epoch': best_epoch,
          'best_validation_dice': best, 'reloaded_best_dice': final['mean_foreground_dice'],
          'epochs': args.epochs, 'checkpoint': 'bestweights.pt'})
    return history
