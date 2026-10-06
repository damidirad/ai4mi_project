#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_
from geometry import grid_from_nifti
from preprocessing_metadata import build_metadata, save_metadata


HU_PERCENTILES: tuple[float, float] = (0.5, 99.5)


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def hu_clipping(img: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(img.astype(np.float32), hu_min, hu_max)
    res = 255 * (clipped - hu_min) / (hu_max - hu_min)

    return res.astype(np.uint8)


def resample_voxels(ct: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
                    target_spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    factors = [s / t for s, t in zip(spacing, target_spacing)]
    if len(target_spacing) == 2:  # Keep Z
        factors.append(1.0)

    kw = dict(mode="constant", prefilter=False, grid_mode=False)
    res_ct = zoom(ct.astype(np.float32), factors, order=1, **kw)
    res_gt = zoom(gt, factors, order=0, **kw)

    return res_ct, res_gt


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, target_spacing: tuple[float, ...] | None = None,
                  hu_window: tuple[float, float] | None = None,
                  save_geometry: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    original_grid = None
    if save_geometry:
        if nib_obj.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Metadata export requires CT units in mm for the current resampler')
        original_grid = grid_from_nifti(nib_obj)
        if not np.allclose(original_grid.spacing, (dx, dy, dz), rtol=1e-6, atol=1e-8):
            raise ValueError('CT affine and header spacing disagree')

    gt: np.ndarray
    if not test_mode:
        gt_obj = nib.load(str(id_path / "GT.nii.gz"))
        gt = np.asarray(gt_obj.dataobj)
        if save_geometry:
            gt_grid = grid_from_nifti(gt_obj)
            if gt_grid.shape != original_grid.shape or not np.allclose(
                    gt_grid.affine, original_grid.affine, rtol=0, atol=1e-5):
                raise ValueError('CT and labels must share the same physical grid')
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    if target_spacing is not None:
        ct, gt = resample_voxels(ct, gt, (dx, dy, dz), target_spacing)

        dx = target_spacing[0]
        dy = target_spacing[1]

        if len(target_spacing) == 3:
            dz = target_spacing[2]

    metadata = None
    if save_geometry:
        metadata = build_metadata(id_, dest_path.name, ct_path, original_grid,
                                  ct.shape, target_spacing, shape, hu_window,
                                  source_spacing=nib_obj.header.get_zooms())

    orig_x, orig_y = ct.shape[:2]

    dx = dx * orig_x / shape[0]
    dy = dy * orig_y / shape[1]

    norm_ct: np.ndarray = norm_arr(ct) if hu_window is None else hu_clipping(ct, *hu_window)

    for idz in range(norm_ct.shape[2]):
        img_slice = resize_(norm_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        for save_subfolder, data in zip(["img", "gt"], [img_slice, gt_slice]):
            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / f"{id_}_{idz:04d}.png"), data)

    if metadata is not None:
        save_metadata(dest_path / 'metadata' / f'{id_}.json', metadata)

    return dx, dy, dz


def median_spacing(ids: list[str], src_path: Path, dims: int) -> tuple[float, ...]:
    spacings = [nib.load(str(src_path / "train" / id_ / f"{id_}.nii.gz")).header.get_zooms()[:dims]
                for id_ in ids]

    return tuple(float(s) for s in np.median(spacings, axis=0))


def hu_window(ids: list[str], src_path: Path) -> tuple[float, float]:
    values = []
    for id_ in ids:
        id_path = src_path / "train" / id_
        ct = np.asarray(nib.load(str(id_path / f"{id_}.nii.gz")).dataobj)
        gt = np.asarray(nib.load(str(id_path / "GT.nii.gz")).dataobj)
        values.append(ct[gt > 0])

    hu_min, hu_max = np.percentile(np.concatenate(values), HU_PERCENTILES)

    return float(hu_min), float(hu_max)


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('Patient_*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*.nii.gz')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    window = hu_window(training_ids, src_path) if args.hu_clip else None
    print(f"HU window: {window}")

    target_spacing = None
    if args.resample != "none":
        dims = 3 if args.resample == "xyz" else 2
        target_spacing = tuple(args.target_spacing) if args.target_spacing else median_spacing(training_ids, src_path, dims)
        assert len(target_spacing) == dims, target_spacing
    print(f"Target spacing: {target_spacing}")

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test',
                                 target_spacing=target_spacing,
                                 hu_window=window,
                                 save_geometry=getattr(args, 'save_metadata', False))
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--save_metadata', action='store_true',
                        help='Save validated per-patient geometry and preprocessing JSON')
    parser.add_argument('--hu_clip', action='store_true', help="Clip HU to training-set organ percentiles")
    parser.add_argument('--resample', choices=('none', 'xy', 'xyz'), default='none',
                        help="Voxel resampling: none (default), xy (preserve Z), or xyz")
    parser.add_argument('--target_spacing', type=float, nargs="+", default=None,
                        help="Spacing in mm: DX DY for xy, DX DY DZ for xyz; defaults to training-set medians")
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    if args.target_spacing is not None:
        if args.resample == 'none':
            parser.error('--target_spacing requires --resample xy or --resample xyz')
        dims = 3 if args.resample == 'xyz' else 2
        if len(args.target_spacing) != dims:
            parser.error(f'--resample {args.resample} requires {dims} values for --target_spacing')
        if not all(np.isfinite(value) and value > 0 for value in args.target_spacing):
            parser.error('--target_spacing values must be finite and positive')
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())