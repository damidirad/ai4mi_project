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

import nrrd  # pip install pynrrd
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


# Fixed mapping from segment name -> output label index
CLASS_ORDER: dict[str, int] = {
    "Background": 0,
    "Heart": 1,
    "Esophagus": 2,
    "Trachea": 3,
    "Aorta": 4,
}
NUM_CLASSES: int = len(CLASS_ORDER)
LABEL_STEP: int = 255 // (NUM_CLASSES - 1)  # 63, matches the original encoding


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


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
    # assert set(np.unique(gt)) == set(range(NUM_CLASSES))

    return True


def parse_segment_info(header: dict) -> dict[int, dict]:
    """
    Parse a .seg.nrrd header and return, for each SegmentN found:
        {"name": str, "label_value": int, "layer": int}
    Segments can appear in any order in the file — this just reads out
    whatever metadata Slicer (or another tool) wrote for each one.
    """
    segments: dict[int, dict] = {}
    i = 0
    while f"Segment{i}_Name" in header or f"Segment{i}_ID" in header:
        name = header.get(f"Segment{i}_Name", header.get(f"Segment{i}_ID", f"Segment_{i}"))
        label_value = int(header.get(f"Segment{i}_LabelValue", 1))
        layer = int(header.get(f"Segment{i}_Layer", 0))
        segments[i] = {"name": name, "label_value": label_value, "layer": layer}
        i += 1

    assert segments, "No SegmentN_* metadata found in .seg.nrrd header"
    return segments


def build_gt_from_segnrrd(seg_path: Path, ct_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Load a .seg.nrrd file and remap it to a single (X, Y, Z) uint8 volume
    with values in range(NUM_CLASSES), using CLASS_ORDER for the mapping —
    regardless of what order/label values the segments were saved with.

    Handles two things Slicer commonly does that will silently corrupt
    the labels if ignored:
      1. Multiple segments can be packed into different binary "layers"
         (an extra leading axis) sharing overlapping LabelValues.
      2. The segmentation is often cropped to the bounding box of the
         segmented structures, with an offset into the full CT volume
         recorded in Segmentation_ReferenceImageExtentOffset.
    """
    data, header = nrrd.read(str(seg_path))
    segments = parse_segment_info(header)

    kinds = header.get("kinds", [])
    if data.ndim == 4:
        layer_axis = kinds.index("list") if "list" in kinds else 0
        spatial_shape = tuple(s for i, s in enumerate(data.shape) if i != layer_axis)
    else:
        layer_axis = None
        spatial_shape = data.shape

    cropped_gt = np.zeros(spatial_shape, dtype=np.uint8)

    seen_names = set()
    for seg in segments.values():
        name = seg["name"]
        if name not in CLASS_ORDER:
            raise ValueError(
                f"Unknown segment name '{name}' in {seg_path}; "
                f"expected one of {list(CLASS_ORDER)}"
            )
        seen_names.add(name)
        out_val = CLASS_ORDER[name]

        if layer_axis is not None:
            layer_data = np.take(data, seg["layer"], axis=layer_axis)
        else:
            layer_data = data

        mask = layer_data == seg["label_value"]
        cropped_gt[mask] = out_val

    # Sanity: warn (loudly) if a class from CLASS_ORDER other than Background
    # is missing from this particular file.
    missing = set(CLASS_ORDER) - seen_names - {"Background"}
    if missing:
        warnings.warn(f"{seg_path}: missing expected segment(s) {missing}")

    # Undo the bounding-box crop, if any, so the GT lines up with the full CT.
    offset_str = header.get("Segmentation_ReferenceImageExtentOffset", "0 0 0")
    offset = tuple(int(v) for v in offset_str.split())

    if cropped_gt.shape == ct_shape and offset == (0, 0, 0):
        return cropped_gt

    full_gt = np.zeros(ct_shape, dtype=np.uint8)
    slices = tuple(slice(o, o + s) for o, s in zip(offset, cropped_gt.shape))
    full_gt[slices] = cropped_gt
    return full_gt


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def resample_voxels(ct: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
                    target_spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    zoom_factors = tuple(s / t for s, t in zip(spacing, target_spacing))

    res_ct = zoom(ct, zoom_factors, order=1)
    res_gt = zoom(gt, zoom_factors, order=0)

    return res_ct, res_gt


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, resample: bool = False,
                  target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        seg_candidates = list(id_path.glob("*.seg.nrrd"))
        assert len(seg_candidates) == 1, f"Expected exactly one .seg.nrrd in {id_path}, found {seg_candidates}"
        gt_path: Path = seg_candidates[0]
        gt = build_gt_from_segnrrd(gt_path, ct.shape)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    if resample:
        ct, gt = resample_voxels(ct, gt, (dx, dy, dz), target_spacing)
        z = ct.shape[2]

    norm_ct: np.ndarray = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= LABEL_STEP
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(NUM_CLASSES))
        assert set(np.unique(gt_slice)) <= set(range(0, 256, LABEL_STEP)), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


def compute_median_spacing(ids: list[str], src_path: Path) -> tuple[float, float, float]:
    spacings = [nib.load(str(src_path / "train" / id_ / f"{id_}.nii.gz")).header.get_zooms()
               for id_ in ids]

    return tuple(np.median(np.asarray(spacings), axis=0).tolist())


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    if args.resample:
        if args.target_spacing:
            target_spacing = tuple(args.target_spacing)
        else:
            target_spacing = compute_median_spacing(training_ids, src_path)
        print(f"Resampling to target spacing {target_spacing} "
             f"({'Median of training set' if not args.target_spacing else 'user-provided'})")

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
                                 resample=args.resample,
                                 target_spacing=target_spacing)
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
    parser.add_argument('--resample', action='store_true',
                        help="Resample the CT/GT volumes to a common voxel spacing before slicing.")
    parser.add_argument('--target_spacing', type=float, nargs=3, default=None,
                        help="Target (dx, dy, dz) spacing in mm, used when --resample is set. "
                             "If omitted, uses the median training-set spacing per axis")
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())