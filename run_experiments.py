#!/usr/bin/env python3
"""Run one screening configuration against datasets prepared by the Makefile."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIELDS = {'id', 'hu_clipping', 'resampling', 'tiling', 'slices', 'loss',
          'optimizer', 'scheduler', 'augmentation', 'epochs'}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {'name', 'training_seed', 'split_seed', 'fold',
                       'validation_patients', 'configurations'}:
        raise ValueError('Unexpected or missing experiment configuration fields')
    if not isinstance(config['name'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', config['name']):
        raise ValueError('Experiment name must be a safe directory name')
    for key in ('training_seed', 'split_seed', 'fold', 'validation_patients'):
        if type(config[key]) is not int or config[key] < (1 if key == 'validation_patients' else 0):
            raise ValueError(f'Invalid {key}')
    if not config['configurations']:
        raise ValueError('No configurations supplied')
    ids = set()
    for row in config['configurations']:
        if set(row) != FIELDS or not isinstance(row['id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
            raise ValueError('Invalid configuration fields or ID')
        if row['id'] in ids:
            raise ValueError('Duplicate configuration ID')
        ids.add(row['id'])
        for key in ('hu_clipping', 'tiling', 'augmentation'):
            if type(row[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(row['slices']) is not int or row['slices'] <= 0 or row['slices'] % 2 == 0:
            raise ValueError('slices must be a positive odd integer')
        if type(row['epochs']) is not int or row['epochs'] <= 0:
            raise ValueError('epochs must be a positive integer')
        if row['resampling'] not in ('none', 'xy', 'xyz'):
            raise ValueError('Invalid resampling mode')
        if any(not isinstance(row[key], str) for key in ('loss', 'optimizer', 'scheduler')):
            raise ValueError('loss, optimizer and scheduler must be strings')
        row['loss'] = row['loss'].lower()
        row['optimizer'] = row['optimizer'].lower()
        row['scheduler'] = row['scheduler'].lower()
        if row['loss'] not in ('ce', 'dice', 'ce_dice', 'boundary', 'ce_dice_boundary', 'ce_tiled'):
            raise ValueError('Unknown loss')
        if row['optimizer'] not in ('adam', 'adamw') or row['scheduler'] not in ('none', 'cosine'):
            raise ValueError('Unknown optimizer or scheduler')
        if row['tiling'] and row['loss'] not in ('ce', 'ce_tiled'):
            raise ValueError('Tiling supports CE only')
        if not row['tiling'] and row['loss'] == 'ce_tiled':
            raise ValueError('CE_TILED requires tiling')
    return config


def dataset_path(row, root):
    """Match the twelve Makefile targets, including the legacy HU path."""
    if not row['tiling'] and row['resampling'] == 'none':
        return root / ('SEGTHOR_hu' if row['hu_clipping'] else 'SEGTHOR')
    name = 'tiled' if row['tiling'] else 'full'
    if row['hu_clipping']:
        name += '_hu'
    if row['resampling'] != 'none':
        name += '_' + row['resampling']
    return root / name / 'SEGTHOR'


def training_command(config, row, dataset, destination, python, workers):
    command = [str(python), '-u', str(ROOT/'main.py'), '--dataset', dataset.name,
               '--data_root', str(dataset.parent), '--dest', str(destination),
               '--mode', 'full', '--loss', 'ce' if row['loss'] == 'ce_tiled' else row['loss'],
               '--optimizer', row['optimizer'], '--scheduler', row['scheduler'],
               '--slices', str(row['slices']), '--epochs', str(row['epochs']),
               '--seed', str(config['training_seed']), '--workers', str(workers), '--gpu']
    if row['tiling']:
        command.append('--tiling')
    # Augmented runs are deliberately blocked until an augmentation adapter is implemented.
    return command


def check_dataset(config, row, dataset, source):
    """Verify the recorded preprocessing and patient split, not just a folder name."""
    from preprocessing_metadata import load_metadata
    original_ids = sorted(p.name for p in (source/'train').glob('Patient_*') if p.is_dir())
    count = config['validation_patients']
    if len(original_ids) <= count:
        raise ValueError('Raw training scans are required to verify the patient split')
    random.Random(config['split_seed']).shuffle(original_ids)
    start = config['fold'] * count
    validation = set(original_ids[start:start + count])
    if len(validation) != count:
        raise ValueError('Fold extends beyond the available patients')
    expected = {'train': set(original_ids) - validation, 'val': validation}
    settings, digests, inventory = set(), {}, {}
    for split in ('train', 'val'):
        paths = sorted((dataset/split/'metadata').glob('*.json'))
        if not paths:
            raise ValueError(f'Missing {split} metadata in {dataset}; prepare a fresh dataset with --save_metadata')
        records = [load_metadata(path) for path in paths]
        expected_files = set()
        if {r['patient_id'] for r in records} != expected[split] or len(records) != len(expected[split]):
            raise ValueError(f'{split} patients do not match the configured split seed/fold')
        for path, r in zip(paths, records):
            if r['patient_id'] != path.stem or r['split'] != split:
                raise ValueError('Metadata filenames or split identifiers do not match')
            if (r['schema_version'] == 2) != row['tiling'] or r['resample'] != row['resampling']:
                raise ValueError('Dataset layout or resampling differs from the experiment')
            if (r['intensity']['method'] == 'hu_window') != row['hu_clipping']:
                raise ValueError('Dataset HU clipping differs from the experiment')
            settings.add(json.dumps({k:r[k] for k in ('resample', 'requested_spacing_mm', 'intensity', 'interpolation')} |
                                    {'layout': r.get('tiling', {}).get('size', r['output']['shape'][:2]),
                                     'stride': r.get('tiling', {}).get('stride')}, sort_keys=True))
            ct = source/'train'/r['patient_id']/f"{r['patient_id']}.nii.gz"
            if not ct.is_file() or not ct.with_name('GT.nii.gz').is_file():
                raise ValueError(f'Missing raw CT or labels for {r["patient_id"]}')
            if Path(r['source_ct']).resolve() != ct.resolve():
                raise ValueError('Metadata refers to a different raw source directory')
            if row['tiling']:
                expected_files.update(f"{r['patient_id']}/img/z{z:04d}_t{t:04d}.png"
                    for z in range(r['working']['shape'][2]) for t in range(len(r['tiling']['starts_xy'])))
            else:
                expected_files.update(f"{r['patient_id']}_{z:04d}.png" for z in range(r['output']['shape'][2]))
            digests[f'{split}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        folder = dataset/split
        if row['tiling']:
            images = list((folder/'tiles').glob('*/img/*.png'))
            labels = list((folder/'tiles').glob('*/gt/*.png'))
        else:
            images = list((folder/'img').glob('*.png'))
            labels = list((folder/'gt').glob('*.png'))
        actual_images = {str(p.relative_to(folder/'tiles')) if row['tiling'] else p.name for p in images}
        actual_labels = {str(p.relative_to(folder/'tiles')).replace('/gt/', '/img/') if row['tiling'] else p.name for p in labels}
        if not images or actual_images != expected_files or actual_labels != expected_files:
            raise ValueError(f'Missing images or labels in {folder}')
        inventory[split] = {'patients': sorted(expected[split]), 'samples':len(images)}
    if len(settings) != 1:
        raise ValueError('Patients or splits have inconsistent preprocessing settings')
    if not row['tiling'] and not (dataset/'spacing.pkl').is_file():
        raise ValueError('Full-slice training requires spacing.pkl')
    return {'metadata_sha256': digests, 'inventory': inventory}


def summarize(destination, row):
    """Read the selected checkpoint's metrics; keep different evaluation grids explicit."""
    import numpy as np
    if not (destination/'bestweights.pt').is_file():
        raise ValueError('Training produced no bestweights.pt')
    if row['tiling']:
        summary = json.loads((destination/'summary.json').read_text())
        history = json.loads((destination/'history.json').read_text())
        if not summary['complete'] or summary['epochs'] != row['epochs'] or len(history) != row['epochs']:
            raise ValueError('Incomplete tiled training output')
        if not np.isfinite(summary['reloaded_best_dice']):
            raise ValueError('Nonfinite tiled validation Dice')
        return {'best_epoch':summary['best_epoch'], 'mean_foreground_dice':summary['reloaded_best_dice'],
                'evaluation_grid':'original_CT'}
    match = re.search(r'epoch (\d+):', (destination/'best_epoch.txt').read_text())
    if not match:
        raise ValueError('Cannot identify the best epoch')
    epoch = int(match.group(1))
    if not 0 <= epoch < row['epochs']:
        raise ValueError('Best epoch is outside the configured training budget')
    for name in ('loss_tra.npy', 'loss_val.npy', 'lr.npy'):
        if np.load(destination/name, allow_pickle=False).shape[0] != row['epochs']:
            raise ValueError('Unexpected training history length')
    dice = np.load(destination/f'iter{epoch:03d}'/'dice_3d_val.npy', allow_pickle=False)
    if dice.ndim != 2 or dice.shape[1] != 5 or not np.isfinite(dice).all():
        raise ValueError('Invalid per-patient Dice output')
    return {'best_epoch':epoch, 'mean_foreground_dice':float(dice[:,1:].mean()),
            'evaluation_grid':'prepared_volume'}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    config = load_config(args.config)
    rows = config['configurations']
    if args.run_index is not None:
        if not 0 <= args.run_index < len(rows):
            raise ValueError('Run index is outside the configuration list')
        row = rows[args.run_index]
    else:
        row = next((r for r in rows if r['id'] == args.run_id), None)
        if row is None:
            raise ValueError(f'Unknown run ID: {args.run_id}')
    dataset = dataset_path(row, args.data_root.absolute())
    run_dir = args.results_root.absolute()/config['name']/row['id']/f"seed_{config['training_seed']}"/f'attempt_{args.attempt:03d}'
    destination = run_dir/'training'
    command = training_command(config, row, dataset, destination, args.python.absolute(), args.workers)
    blocker = 'Augmentation is not integrated; this run must not execute without it' if row['augmentation'] else None
    plan = {'configuration':row, 'dataset':str(dataset), 'run_directory':str(run_dir),
            'command':command, 'blocked':blocker}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if blocker:
        raise ValueError(blocker)
    data = check_dataset(config, row, dataset, args.source_dir.absolute())
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    identity = {'experiment':{k:v for k,v in config.items() if k != 'configurations'},
                'configuration':row, 'command':command, 'dataset':data, 'source_sha256':sources}
    manifest_path = run_dir/'runner.json'
    if run_dir.exists():
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            if previous.get('status') == 'completed' and previous.get('identity') == identity:
                if summarize(destination, row) == previous['metrics']:
                    print('Matching completed run already exists; skipping.')
                    return
        raise FileExistsError(f'{run_dir} already exists; inspect it and use a new --attempt number')
    # The legacy training route otherwise silently falls back to CPU.
    subprocess.run([str(args.python.absolute()), '-c',
                    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "A CUDA GPU is required")'], check=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'running', 'identity':identity,
                'started_at':datetime.now(timezone.utc).isoformat(),
                'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    write_json(manifest_path, manifest)
    print('Running: ' + shlex.join(command), flush=True)
    try:
        with (run_dir/'train.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['metrics'] = summarize(destination, row)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    print(f'Completed {row["id"]}: {manifest_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/screening.json')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-id', help='Configuration ID, e.g. M1')
    selection.add_argument('--run-index', type=int, help='Zero-based index for Slurm arrays')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/segthor_train_full')
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--attempt', type=int, default=1, help='Use a new number to retry a failed run')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan and blockers without creating files')
    args = parser.parse_args()
    if args.workers < 0 or args.attempt < 1:
        parser.error('workers must be nonnegative and attempt must be positive')
    try:
        run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
