"""Small real-data preprocessing check; run from the project directory."""

import argparse
from functools import partial
import pickle
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Support direct execution without installing the project as a package.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dataset import SliceDataset
from main import img_transform, gt_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--work_dir', type=Path, default=Path('results/smoke_simple_resampling'),
        help='Directory for the subset, output, and log; must not exist.'
    )
    parser.add_argument('--process', type=int, default=1)
    args = parser.parse_args()
    if args.process < 1:
        parser.error('--process must be positive')

    patients = sorted(p for p in (args.source_dir / 'train').iterdir() if p.is_dir())[:3]
    if len(patients) != 3:
        parser.error('At least three training patients are required')
    for patient in patients:
        if not (patient / f'{patient.name}.nii.gz').is_file():
            parser.error(f'Missing CT for {patient.name}')
        if len(list(patient.glob('*.seg.nrrd'))) != 1:
            parser.error(f'Expected one .seg.nrrd for {patient.name}')

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = work / 'source'
    (source / 'train').mkdir(parents=True)
    (source / 'test').mkdir()
    for patient in patients:
        (source / 'train' / patient.name).symlink_to(patient.resolve(), target_is_directory=True)
    output = work / 'output'
    command = [
        sys.executable, '-u', str(PROJECT / 'slice_segthor.py'),
        '--source_dir', str(source), '--dest_dir', str(output),
        '--shape', '256', '256', '--retains', '1', '--seed', '0',
        '--resample', '--process', str(args.process),
    ]
    print('Running:', ' '.join(command), flush=True)
    with (work / 'preprocessing.log').open('w') as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)

    with (output / 'spacing.pkl').open('rb') as handle:
        spacings = pickle.load(handle)
    assert set(spacings) == {p.name for p in patients}
    split_patients = {}
    for split in ('train', 'val'):
        images = sorted((output / split / 'img').glob('*.png'))
        labels = sorted((output / split / 'gt').glob('*.png'))
        assert images and [p.name for p in images] == [p.name for p in labels]
        ids = sorted({p.stem.rsplit('_', 1)[0] for p in images})
        split_patients[split] = ids
        for id_ in ids:
            ct = nib.load(str(source / 'train' / id_ / f'{id_}.nii.gz'))
            np.testing.assert_array_equal(spacings[id_], ct.header.get_zooms())
            assert [p.name for p in images if p.stem.rsplit('_', 1)[0] == id_] == [
                f'{id_}_{z:04d}.png' for z in range(ct.shape[2])
            ]
        for image_path, label_path in zip(images, labels):
            with Image.open(image_path) as image, Image.open(label_path) as label:
                a, b = np.array(image), np.array(label)
            assert a.shape == b.shape == (256, 256)
            assert a.dtype == b.dtype == np.uint8
            assert set(np.unique(b)) <= {0, 63, 126, 189, 252}
        dataset = SliceDataset(split, output, img_transform=img_transform,
                               gt_transform=partial(gt_transform, 5))
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
        assert batch['images'].shape == (2, 1, 256, 256)
        assert batch['gts'].shape == (2, 5, 256, 256)
        assert torch.isfinite(batch['images']).all()
        assert 0 <= batch['images'].min() <= batch['images'].max() <= 1
        assert torch.all(batch['gts'].sum(dim=1) == 1)
        print(f'{split}: {len(ids)} patients, {len(images)} slice pairs; batch OK')

    assert len(split_patients['train']) == 2 and len(split_patients['val']) == 1
    assert set(split_patients['train']).isdisjoint(split_patients['val'])
    target = tuple(np.median([spacings[p][:2] for p in split_patients['train']], axis=0).tolist())
    log_text = (work / 'preprocessing.log').read_text()
    assert f'Resampling X/Y to {target} mm (training-set median)' in log_text
    print(f'PASS: training-only target {target}; PNGs, Z counts, source spacings, and batches checked.')
    print(f'Output and preprocessing log retained in {work}')


if __name__ == '__main__':
    main()
