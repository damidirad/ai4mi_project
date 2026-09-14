#!/usr/bin/env python3.10
"""Convert 3D Slicer .seg.nrrd annotations to the GT.nii.gz format expected by slice_segthor.py.

Segment label values inside a .seg.nrrd are assigned by creation order in Slicer, so they
differ between files (e.g. Esophagus can be labelled 1 in one patient and 5 in another).
This script remaps voxels by segment *name* to the project's fixed class scheme, so class
indices are consistent across all patients regardless of how they were drawn in Slicer.
"""

import glob
import argparse
from pathlib import Path

import nrrd
import numpy as np
import nibabel as nib

NAME_TO_CLASS = {
    "esophagus": 1,
    "heart": 2,
    "trachea": 3,
    "aorta": 4,
}


def convert_patient(patient_dir: Path) -> None:
    nrrd_files = list(patient_dir.glob("*.nrrd"))
    assert len(nrrd_files) == 1, f"{patient_dir}: expected exactly 1 .nrrd file, found {nrrd_files}"
    nrrd_path = nrrd_files[0]

    header = nrrd.read_header(open(nrrd_path, "rb"))
    data, _ = nrrd.read(str(nrrd_path))

    segments: dict[str, tuple[int, int]] = {}  # name -> (raw_label, layer)
    i = 0
    while f"Segment{i}_Name" in header:
        name = header[f"Segment{i}_Name"].strip().lower()
        raw_label = int(header[f"Segment{i}_LabelValue"])
        layer = int(header[f"Segment{i}_Layer"])
        segments[name] = (raw_label, layer)
        i += 1

    assert set(segments) == set(NAME_TO_CLASS), f"{patient_dir}: unexpected segment names {set(segments)}"
    assert all(layer == 0 for _, layer in segments.values()), \
        f"{patient_dir}: multi-layer segmentation not supported {segments}"

    known_raw_labels = {raw for raw, _ in segments.values()}
    unexpected = set(np.unique(data).tolist()) - known_raw_labels - {0}
    assert not unexpected, f"{patient_dir}: unexpected voxel values {unexpected} not covered by any segment"

    remapped = np.zeros_like(data, dtype=np.uint8)
    for name, (raw_label, _layer) in segments.items():
        remapped[data == raw_label] = NAME_TO_CLASS[name]

    patient_id = patient_dir.name
    ct_path = patient_dir / f"{patient_id}.nii.gz"
    ct_nib = nib.load(str(ct_path))
    assert ct_nib.shape == remapped.shape, (patient_dir, ct_nib.shape, remapped.shape)

    gt_path = patient_dir / "GT.nii.gz"
    if gt_path.exists():
        backup_path = patient_dir / "GT.nii.gz.bak"
        gt_path.rename(backup_path)
        print(f"  backed up existing {gt_path.name} -> {backup_path.name}")

    new_nib = nib.nifti1.Nifti1Image(remapped, affine=ct_nib.affine)
    nib.save(new_nib, str(gt_path))
    print(f"{patient_dir}: wrote {gt_path} from {nrrd_path.name} "
          f"(classes present: {sorted(np.unique(remapped).tolist())})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=str, default="data/segthor_part1/train",
                        help="Folder containing Patient_XX subfolders")
    args = parser.parse_args()

    patient_dirs = sorted(Path(args.source_dir).glob("Patient_*"))
    assert patient_dirs, f"No Patient_* folders found under {args.source_dir}"

    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        convert_patient(patient_dir)


if __name__ == "__main__":
    main()
