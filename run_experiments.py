#!/usr/bin/env python3
"""Prepare or run one initial or final comparison experiment."""
import argparse
import copy
import fcntl
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
OPTIONAL_FIELDS = {'early_stopping_patience', 'early_stopping_min_delta'}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if 'folds' in config:
        return load_final_comparison(config, Path(path))
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
        if not FIELDS <= set(row) or not set(row) <= FIELDS | OPTIONAL_FIELDS or not isinstance(row['id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
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
        if 'early_stopping_patience' in row and (
                type(row['early_stopping_patience']) is not int or row['early_stopping_patience'] <= 0):
            raise ValueError('early_stopping_patience must be a positive integer')
        if 'early_stopping_min_delta' in row and (
                not isinstance(row['early_stopping_min_delta'], (int, float)) or row['early_stopping_min_delta'] < 0):
            raise ValueError('early_stopping_min_delta must be nonnegative')
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
        if row['tiling'] and row['augmentation']:
            raise ValueError('Tiled training does not support augmentation')
        if row['tiling'] and row['loss'] not in ('ce', 'ce_tiled'):
            raise ValueError('Tiling supports CE only')
        if not row['tiling'] and row['loss'] == 'ce_tiled':
            raise ValueError('CE_TILED requires tiling')
    return config


def load_final_comparison(config, path):
    """Expand three roles, patient folds and training seeds into independent jobs."""
    required = {'name', 'initial_comparison_config', 'baseline', 'top_candidates', 'folds',
                'training_seeds', 'split_seed', 'validation_patients'}
    optional = {'max_epochs', 'early_stopping_patience', 'early_stopping_min_delta', 'run_order'}
    if not required <= set(config) or not set(config) <= required | optional or config['baseline'] != 'original_enet':
        raise ValueError('Invalid final comparison configuration or baseline')
    if not isinstance(config['name'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', config['name']):
        raise ValueError('Invalid experiment name')
    for key in ('folds', 'training_seeds'):
        values = config[key]
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain distinct nonnegative integers')
    for key in ('split_seed', 'validation_patients'):
        if type(config[key]) is not int or config[key] < (1 if key == 'validation_patients' else 0):
            raise ValueError(f'Invalid {key}')
    max_epochs = config.get('max_epochs', 20)
    patience = config.get('early_stopping_patience')
    min_delta = config.get('early_stopping_min_delta', 0.0)
    if type(max_epochs) is not int or max_epochs <= 0:
        raise ValueError('max_epochs must be a positive integer')
    if patience is not None and (type(patience) is not int or patience <= 0):
        raise ValueError('early_stopping_patience must be a positive integer')
    if not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise ValueError('early_stopping_min_delta must be nonnegative')
    run_order = config.get('run_order', 'balanced_prefix')
    if run_order not in ('balanced_prefix', 'role_fold_seed'):
        raise ValueError('run_order must be balanced_prefix or role_fold_seed')
    candidates = config['top_candidates']
    if set(candidates) != {'TOP1', 'TOP2'}:
        raise ValueError('Specify TOP1 and TOP2, using null until selected')
    selected = [value for value in candidates.values() if value is not None]
    if any(not isinstance(v, str) for v in selected) or len(set(selected)) != len(selected):
        raise ValueError('Top candidates must be distinct configuration IDs')
    initial_comparison_path = path.parent/config['initial_comparison_config']
    # Read only an initial comparison file, avoiding recursive final comparison references.
    if 'folds' in json.loads(initial_comparison_path.read_text()):
        raise ValueError('initial_comparison_config must reference an initial comparison configuration')
    initial_comparison = load_config(initial_comparison_path)
    available = {row['id']:row for row in initial_comparison['configurations']}
    if any(value not in available for value in selected):
        raise ValueError('Unknown top candidate')
    baseline = {'id':'BASELINE', 'hu_clipping':False, 'resampling':'none', 'tiling':False,
                'slices':1, 'loss':'ce', 'optimizer':'adam', 'scheduler':'none',
                'augmentation':False, 'epochs':max_epochs, 'class_weighting':'none',
                'original_grid_validation':True}
    if patience is not None:
        baseline['early_stopping_patience'] = patience
        baseline['early_stopping_min_delta'] = min_delta
    roles = [baseline]
    for role in ('TOP1', 'TOP2'):
        candidate = candidates[role]
        if candidate is None:
            roles.append({'id':role, 'placeholder':True})
        else:
            row = copy.deepcopy(available[candidate])
            row.update(id=role, source_id=candidate, class_weighting='inverse_frequency',
                       original_grid_validation=True, epochs=max_epochs)
            if patience is not None:
                row['early_stopping_patience'] = patience
                row['early_stopping_min_delta'] = min_delta
            roles.append(row)
    if run_order == 'balanced_prefix':
        folds = config['folds']
        seeds = config['training_seeds']
        primary_folds, deferred_folds = folds[:2], folds[2:]
        primary_seeds, deferred_seeds = seeds[:2], seeds[2:]
        fold_seed_pairs = (
            [(fold, seed) for seed in primary_seeds for fold in primary_folds]
            + [(fold, seed) for seed in primary_seeds for fold in deferred_folds]
            + [(fold, seed) for seed in deferred_seeds for fold in folds]
        )
        config['runs'] = [{'configuration':row, 'fold':fold, 'training_seed':seed}
                          for fold, seed in fold_seed_pairs for row in roles]
    else:
        config['runs'] = [{'configuration':row, 'fold':fold, 'training_seed':seed}
                          for row in roles for fold in config['folds'] for seed in config['training_seeds']]
    return config


def preparation_command(config, row, dataset, source, python, processes):
    command = [str(python), '-u', str(ROOT/'slice_segthor.py'), '--source_dir', str(source),
               '--dest_dir', str(dataset), '--retains', str(config['validation_patients']),
               '--seed', str(config['split_seed']), '--fold', str(config['fold']),
               '--process', str(processes), '--save_metadata', '--resample', row['resampling']]
    if row['hu_clipping']:
        command.append('--hu_clip')
    if row['tiling']:
        command += ['--tiling', '--tile-size', '256', '256', '--tile-stride', '128', '128']
    return command


def prepare_dataset(config, row, dataset, source, command):
    """Prepare once per layout/fold; never reuse a partial or mismatched dataset."""
    dataset.parent.mkdir(parents=True, exist_ok=True)
    with (dataset.parent/(dataset.name + '.prepare.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This dataset is being prepared by another process') from None
        if not dataset.exists():
            subprocess.run(command, cwd=ROOT, check=True)
        check_dataset(config, row, dataset, source)
    print(f'Prepared and verified: {dataset}')


def evaluation_command(row, dataset, destination, python):
    command = [str(python), '-u', str(ROOT/'main.py'), '--dataset', dataset.name,
               '--data_root', str(dataset.parent), '--dest', str(destination/'original_validation'),
               '--evaluate-checkpoint', str(destination/'training/bestweights.pt'),
               '--split', 'val', '--slices', str(row['slices']), '--gpu']
    if row['tiling']:
        command.append('--tiling')
    return command


def read_run_metrics(destination, row):
    metrics = summarize(destination/'training', row)
    if row.get('original_grid_validation'):
        report = json.loads((destination/'original_validation/metrics.json').read_text())
        if report['grid'] != 'original_CT' or report['mean_foreground_dice'] is None:
            raise ValueError('Original-grid validation was not completed')
        metrics.update(mean_foreground_dice=report['mean_foreground_dice'], evaluation_grid='original_CT')
    return metrics


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
    if row['tiling'] and row['augmentation']:
        raise ValueError('Tiled training does not support augmentation')
    command = [str(python), '-u', str(ROOT/'main.py'), '--dataset', dataset.name,
               '--data_root', str(dataset.parent), '--dest', str(destination),
               '--mode', 'full', '--loss', 'ce' if row['loss'] == 'ce_tiled' else row['loss'],
               '--optimizer', row['optimizer'], '--scheduler', row['scheduler'],
               '--slices', str(row['slices']), '--epochs', str(row['epochs']),
               '--seed', str(config['training_seed']), '--workers', str(workers), '--gpu']
    if row['tiling']:
        command.append('--tiling')
    if row.get('class_weighting'):
        command += ['--class-weighting', row['class_weighting']]
    if row.get('original_grid_validation'):
        command.append('--original-grid-validation')
    if row.get('early_stopping_patience') is not None:
        command += ['--early-stopping-patience', str(row['early_stopping_patience']),
                    '--early-stopping-min-delta', str(row.get('early_stopping_min_delta', 0.0))]
    command += ['--augmentation', 'mild' if row['augmentation'] else 'none']
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
    def load_array(base):
        base = Path(base)
        candidates = [base.with_suffix('.npy'), base.with_suffix('.npz')]
        for path in candidates:
            if path.is_file():
                value = np.load(path, allow_pickle=False)
                if isinstance(value, np.lib.npyio.NpzFile):
                    return value[value.files[0]]
                return value
        raise ValueError(f'Missing metric file: {base}.npy/.npz')

    if not (destination/'bestweights.pt').is_file():
        raise ValueError('Training produced no bestweights.pt')
    if row['tiling']:
        summary = json.loads((destination/'summary.json').read_text())
        history = json.loads((destination/'history.json').read_text())
        expected = summary.get('epochs_ran', summary['epochs'])
        if not summary['complete'] or summary['epochs'] != row['epochs'] or len(history) != expected:
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
    for name in ('loss_tra', 'loss_val', 'lr'):
        history = load_array(destination/name)
        if history.shape[0] > row['epochs'] or history.shape[0] <= epoch:
            raise ValueError('Unexpected training history length')
    dice = load_array(destination/f'iter{epoch:03d}'/'dice_3d_val')
    if dice.ndim != 2 or dice.shape[1] != 5 or not np.isfinite(dice).all():
        raise ValueError('Invalid per-patient Dice output')
    if row.get('original_grid_validation'):
        report = json.loads((destination/f'original_epoch_{epoch:03d}'/'metrics.json').read_text())
        return {'best_epoch':epoch, 'mean_foreground_dice':report['mean_foreground_dice'],
                'evaluation_grid':'original_CT'}
    return {'best_epoch':epoch, 'mean_foreground_dice':float(dice[:,1:].mean()),
            'evaluation_grid':'prepared_volume'}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    config = load_config(args.config)
    final_comparison = 'runs' in config
    if final_comparison:
        if args.run_index is None or not 0 <= args.run_index < len(config['runs']):
            raise ValueError('Final comparison requires --run-index in the expanded run list')
        selected = config['runs'][args.run_index]
        row = selected['configuration']
        config = {key:value for key,value in config.items() if key != 'runs'}
        config.update(fold=selected['fold'], training_seed=selected['training_seed'])
        if row.get('placeholder'):
            message = f"Select {row['id']} in top_candidates before running this job"
            print(json.dumps({'run_index':args.run_index, 'fold':config['fold'],
                              'training_seed':config['training_seed'], 'blocked':message}, indent=2))
            if args.dry_run:
                return
            raise ValueError(message)
    else:
        rows = config['configurations']
        if args.run_index is not None:
            if not 0 <= args.run_index < len(rows):
                raise ValueError('Run index is outside the configuration list')
            row = rows[args.run_index]
        else:
            row = next((r for r in rows if r['id'] == args.run_id), None)
            if row is None:
                raise ValueError(f'Unknown run ID: {args.run_id}')
    data_root = args.data_root.absolute()
    output_root = args.results_root.absolute()/config['name']/row['id']
    if final_comparison:
        data_root = data_root/config['name']/f"fold_{config['fold']}"
        output_root = output_root/f"fold_{config['fold']}"
    dataset = dataset_path(row, data_root)
    run_dir = output_root/f"seed_{config['training_seed']}"/f'attempt_{args.attempt:03d}'
    destination = run_dir/'training'
    command = training_command(config, row, dataset, destination, args.python.absolute(), args.workers)
    preparation = preparation_command(config, row, dataset, args.source_dir.absolute(),
                                      args.python.absolute(), getattr(args, 'processes', 4))
    evaluation = evaluation_command(row, dataset, run_dir, args.python.absolute()) if final_comparison else None
    blocker = None
    plan = {'configuration':row, 'dataset':str(dataset), 'run_directory':str(run_dir),
            'command':command, 'preparation_command':preparation, 'evaluation_command':evaluation,
            'fold':config['fold'], 'training_seed':config['training_seed'], 'blocked':blocker}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if getattr(args, 'prepare_only', False):
        prepare_dataset(config, row, dataset, args.source_dir.absolute(), preparation)
        return
    if blocker:
        raise ValueError(blocker)
    data = check_dataset(config, row, dataset, args.source_dir.absolute())
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    identity = {'experiment':{key:config[key] for key in ('name', 'training_seed', 'split_seed', 'fold', 'validation_patients')},
                'configuration':row, 'command':command, 'dataset':data, 'source_sha256':sources,
                'evaluation_command':evaluation}
    manifest_path = run_dir/'runner.json'
    if run_dir.exists():
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            if previous.get('status') == 'completed' and previous.get('identity') == identity:
                if read_run_metrics(run_dir, row) == previous['metrics']:
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
        if evaluation is not None:
            with (run_dir/'evaluation.log').open('x') as log:
                subprocess.run(evaluation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['metrics'] = read_run_metrics(run_dir, row)
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
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/initial_comparison.json')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-id', help='Configuration ID, e.g. M1')
    selection.add_argument('--run-index', type=int, help='Zero-based index for Slurm arrays')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/segthor_train_full')
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--attempt', type=int, default=1, help='Use a new number to retry a failed run')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Prepare and verify the selected dataset without training; use a CPU allocation')
    parser.add_argument('--processes', type=int, default=4, help='Preprocessing workers')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan and blockers without creating files')
    args = parser.parse_args()
    if args.workers < 0 or args.attempt < 1 or args.processes < 1:
        parser.error('workers must be nonnegative; attempt and processes must be positive')
    try:
        run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
