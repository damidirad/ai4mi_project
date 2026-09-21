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


# Clip HU values to a soft-tissue/mediastinum window before rescaling to uint8
HU_CLIP_MIN: float = -400.0
HU_CLIP_MAX: float = 400.0


# Intensity Normalization Old
def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm
    return res.astype(np.uint8)


# Intensity Normalization New (HU-clipping)
def hu_clipping(img: np.ndarray) -> np.ndarray:
    clipped = np.clip(img.astype(np.float32), HU_CLIP_MIN, HU_CLIP_MAX)
    res = 255 * (clipped - HU_CLIP_MIN) / (HU_CLIP_MAX - HU_CLIP_MIN)

    return res.astype(np.uint8)

###


# Spatial Normalization
def prepare_variant(source_dir: Path, variant_dir: Path, train_patients: list[str],
                    val_patients: list[str], shape: tuple[int, int], resample: bool,
                    target_spacing: tuple[float, float, float]) -> None:
    if variant_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {variant_dir}; use --force to recreate the experiment."
        )

    for split, patients in (("train", train_patients), ("val", val_patients)):
        for patient_id in patients:
            patient_dir = source_dir / "train" / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")

            slice_patient(
                dest_path=variant_dir / split,
                source_path=source_dir,
                shape=shape,
                test_mode=False,
                resample=False,
                spatial_normalize=resample,
                target_spacing=target_spacing,
                id_=patient_id,
            )

    metadata = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "resample": resample,
        "target_spacing_mm": target_spacing,
        "slice_shape": shape,
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_variant(variant_dir: Path, result_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("main.py")),
        "--dataset", "SEGTHOR",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--data_root", str(variant_dir.parent),
        "--dest", str(result_dir),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    if args.gpu:
        command.append("--gpu")
    subprocess.run(command, check=True)


def summarize(result_dir: Path) -> dict[str, float | int | str]:
    dice_val = np.load(result_dir / "dice_val.npy")
    foreground_dice = dice_val[:, :, 1:].mean(axis=(1, 2))
    best_epoch = int(np.argmax(foreground_dice))
    return {
        "variant": result_dir.name,
        "best_epoch": best_epoch,
        "best_val_foreground_dice": float(foreground_dice[best_epoch]),
        "final_val_foreground_dice": float(foreground_dice[-1]),
    }

###


# Voxel Resampling
def resample_voxels(ct: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
    target_spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    if ct.ndim != 3 or gt.ndim != 3:
        raise ValueError("CT and labels must both be 3D arrays")

    if ct.shape != gt.shape:
        raise ValueError(
            f"CT and labels must have the same shape: {ct.shape} != {gt.shape}"
        )

    if any(size == 0 for size in ct.shape):
        raise ValueError("CT and labels must have non-empty spatial dimensions")

    dx, dy, dz = validate_spacing(
        spacing,
        dimensions=3,
        name="Source spacing",
    )
    target_spacing = validate_spacing(
        target_spacing,
        dimensions=len(target_spacing),
        name="Target spacing",
    )
    if len(target_spacing) not in [2, 3]:
        raise ValueError("Target spacing must contain either X/Y or X/Y/Z values")

    if len(target_spacing) == 2:
        target_dx, target_dy = target_spacing
        zoom_factors = (dx / target_dx, dy / target_dy, 1.0)
    else:
        target_dx, target_dy, target_dz = target_spacing
        zoom_factors = (dx / target_dx, dy / target_dy, dz / target_dz)

    # Float32 preserves fractional CT values during linear interpolation.
    # Both calls use the same boundary and coordinate settings:
    # constant: fill outside the input with zero.
    # grid_mode=False: measure coordinates between voxel centers.
    # Orders 0 and 1 do not need spline prefiltering.
    res_ct = zoom(
        ct.astype(np.float32, copy=False),
        zoom_factors,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
        grid_mode=False,
    )

    res_gt = zoom(
        gt,
        zoom_factors,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
        grid_mode=False,
    )

    return res_ct, res_gt

###


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


def normalize_ct(ct: np.ndarray, mode: str) -> np.ndarray:
    match mode:
        case "old":
            return norm_arr(ct)
        case "new":
            return hu_clipping(ct)
        case "none":
            return ct.astype(np.uint8)
        case _:
            raise ValueError(f"Unknown intensity normalization mode: {mode}")


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, resample: bool = False,
                  target_spacing: tuple[float, ...] | None = None,
                  intensity_normalization: str = "old",
                  spatial_normalize: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    should_resample = resample or spatial_normalize
    spacing_dimensions = 3 if spatial_normalize else 2

    if should_resample:
        validate_spacing(
            (dx, dy, dz),
            dimensions=3,
            name=f"Source spacing for {id_}",
        )

        if target_spacing is None:
            raise ValueError("Target spacing is required when resampling")

        target_spacing = validate_spacing(
            target_spacing,
            dimensions=spacing_dimensions,
            name="Target spacing",
        )

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

    if should_resample:
        ct, gt = resample_voxels(ct, gt, (dx, dy, dz), target_spacing)

    norm_ct: np.ndarray = normalize_ct(ct, intensity_normalization)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(to_slice_ct.shape[2]):
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


def validate_spacing(spacing, *, dimensions: int, name: str) -> tuple[float, ...]:
    values = np.asarray(spacing, dtype=np.float64)

    if values.shape != (dimensions,):
        raise ValueError(f"{name} must contain {dimensions} values")

    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain only finite, positive values")

    return tuple(float(value) for value in values)


def compute_median_spacing(ids: list[str], src_path: Path,
                           dimensions: int = 2) -> tuple[float, ...]:
    if not ids:
        raise ValueError("Cannot compute target spacing from an empty training set")
    if dimensions not in [2, 3]:
        raise ValueError("Median spacing can only be computed for 2 or 3 dimensions")

    spacings = []
    for id_ in ids:
        ct_path = src_path / "train" / id_ / f"{id_}.nii.gz"
        spacing = validate_spacing(
            nib.load(str(ct_path)).header.get_zooms(),
            dimensions=3,
            name=f"Source spacing for {id_}",
        )
        spacings.append(spacing[:dimensions])

    median_spacing = np.median(np.asarray(spacings), axis=0)
    target_spacing = validate_spacing(
        median_spacing,
        dimensions=dimensions,
        name="Training-set target spacing",
    )
    return target_spacing


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids)
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    print(f"Founds {len(validation_ids)} validation ids")
    print(f"Validation ids: {validation_ids}")
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)
    print(f"Founds {len(training_ids)} train ids")
    print(f"Train ids: {training_ids}")

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(f"Test ids: {test_ids}")

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

    intensity_normalization = getattr(args, "intensity_normalization", "old")
    if getattr(args, "hu_clip", False):
        intensity_normalization = "new"

    voxel_resample = getattr(args, "voxel_resample", getattr(args, "resample", False))
    spatial_normalize = getattr(args, "spatial_normalize", False)

    if spatial_normalize and voxel_resample:
        raise ValueError("Use either --spatial_normalize or --voxel_resample/--resample, not both")

    target_spacing: tuple[float, ...] | None = None
    should_resample = voxel_resample or spatial_normalize
    spacing_dimensions = 3 if spatial_normalize else 2

    print(f"Intensity normalization: {intensity_normalization}")

    if should_resample:
        if args.target_spacing is not None:
            target_spacing = validate_spacing(
                args.target_spacing,
                dimensions=spacing_dimensions,
                name="Manual target spacing",
            )
            spacing_source = "manual override"
        else:
            target_spacing = compute_median_spacing(training_ids, src_path, spacing_dimensions)
            spacing_source = "training-set median"

        mode_name = "spatial normalization X/Y/Z" if spatial_normalize else "voxel resampling X/Y"
        print(
            f"{mode_name} to {target_spacing} mm "
            f"({spacing_source})"
        )

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
                                 resample=voxel_resample,
                                 spatial_normalize=spatial_normalize,
                                 target_spacing=target_spacing,
                                 intensity_normalization=intensity_normalization)
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
        print(f"Saved spacing dictionnary to {f}\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--intensity_normalization', choices=['old', 'new', 'none'], default='old',
                        help="CT intensity preprocessing: old min/max normalization, new HU clipping, or none.")
    parser.add_argument('--hu_clip', action='store_true',
                        help="Shortcut for --intensity_normalization new.")
    parser.add_argument('--resample', '--voxel_resample', dest='voxel_resample',
                        action='store_true',
                        help="Resample CT/GT in X/Y before slicing; preserve Z.")
    parser.add_argument('--spatial_normalize', action='store_true',
                        help="Resample CT/GT in X/Y/Z before slicing.")
    parser.add_argument('--target_spacing', type=float, nargs="+", metavar='D', default=None,
        help=(
            "Manual target spacing in mm. Use DX DY with --voxel_resample/--resample "
            "or DX DY DZ with --spatial_normalize. "
            "Values must be finite and positive. "
            "Defaults to the median training-set spacing for the selected dimensions."
        ),
    )
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    if args.spatial_normalize and args.voxel_resample:
        parser.error("Use either --spatial_normalize or --voxel_resample/--resample, not both")
    if args.target_spacing is not None:
        if args.voxel_resample and len(args.target_spacing) != 2:
            parser.error("--voxel_resample/--resample requires --target_spacing DX DY")
        if args.spatial_normalize and len(args.target_spacing) != 3:
            parser.error("--spatial_normalize requires --target_spacing DX DY DZ")
        if not args.voxel_resample and not args.spatial_normalize:
            parser.error("--target_spacing requires --voxel_resample/--resample or --spatial_normalize")
    args.resample = args.voxel_resample
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
