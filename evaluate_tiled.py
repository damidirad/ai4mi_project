"""Restore working labels to original CT geometry and evaluate whole volumes."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt

from geometry import Grid, grid_from_nifti, voxel_mapping
from preprocessing_metadata import validate_metadata
from reconstruct_tiled import predict_working_volume
from utils import surface


def _labels(array, shape):
    array = np.asarray(array)
    if array.shape != tuple(shape) or not np.issubdtype(array.dtype, np.integer) or not np.isin(array, range(5)).all():
        raise ValueError('Expected integer class labels 0..4 on the specified grid')
    return array


def restore_original_grid(working_labels, metadata):
    validate_metadata(metadata)
    if metadata['schema_version'] != 2:
        raise ValueError('Original-grid restoration requires tiled metadata')
    original = Grid(metadata['original']['shape'], metadata['original']['affine_ras_mm'])
    working = Grid(metadata['working']['shape'], metadata['working']['affine_ras_mm'])
    labels = _labels(working_labels, working.shape)
    mapping = voxel_mapping(working, original)
    # Remove numerical zero terms from solving rotated/flipped affines.
    mapping[np.abs(mapping) < 1e-12] = 0
    return affine_transform(labels.astype(np.uint8), mapping[:3, :3], offset=mapping[:3, 3],
                            output_shape=original.shape, order=0, prefilter=False,
                            mode='nearest')


def reference_image(metadata, source_ct=None):
    validate_metadata(metadata)
    image = nib.load(str(source_ct or metadata['source_ct']))
    grid = grid_from_nifti(image)
    expected = Grid(metadata['original']['shape'], metadata['original']['affine_ras_mm'])
    if grid.shape != expected.shape or not np.allclose(grid.affine, expected.affine, rtol=0, atol=1e-5):
        raise ValueError('Source CT no longer matches recorded geometry')
    return image, grid


def save_original_prediction(labels, metadata, destination, source_ct=None):
    reference, grid = reference_image(metadata, source_ct)
    labels = _labels(labels, grid.shape)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not (str(destination).endswith('.nii') or str(destination).endswith('.nii.gz')):
        raise ValueError('Prediction destination must end in .nii or .nii.gz')
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(labels.astype(np.uint8), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)


def volume_metrics(prediction, target, grid):
    """Foreground Dice, pooled bidirectional surface HD95/ASSD in mm.

    Match utils.surface's 26-neighbour erosion convention. Undefined distances
    are null with an explicit empty-mask status, never silently dropped.
    """
    prediction, target = _labels(prediction, grid.shape), _labels(target, grid.shape)
    if not np.allclose(grid.direction.T @ grid.direction, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError('Surface distances require an orthogonal grid; shear is unsupported')
    result = {}
    for label in range(1, 5):
        gt, pred = target == label, prediction == label
        ng, npred = int(gt.sum()), int(pred.sum())
        if ng == 0 and npred == 0:
            dice, hd95, assd, status = 1., 0., 0., 'both_empty'
        elif ng == 0 or npred == 0:
            dice, hd95, assd = 0., None, None
            status = 'target_empty' if ng == 0 else 'prediction_empty'
        else:
            dice = float(2 * np.count_nonzero(gt & pred) / (ng + npred))
            gs, ps = surface(gt), surface(pred)
            a = distance_transform_edt(~ps, sampling=grid.spacing)[gs]
            b = distance_transform_edt(~gs, sampling=grid.spacing)[ps]
            distances = np.concatenate([a, b])
            hd95, assd, status = float(np.percentile(distances, 95)), float(distances.mean()), 'ok'
        result[str(label)] = {'dice': dice, 'hd95_mm': hd95, 'assd_mm': assd, 'status': status}
    return result


def evaluate_original_prediction(prediction, metadata, source_ct=None, source_gt=None):
    _, grid = reference_image(metadata, source_ct)
    ct_path = Path(source_ct or metadata['source_ct'])
    gt = nib.load(str(source_gt or ct_path.with_name('GT.nii.gz')))
    gt_grid = grid_from_nifti(gt)
    if gt_grid.shape != grid.shape or not np.allclose(gt_grid.affine, grid.affine, rtol=0, atol=1e-5):
        raise ValueError('Original GT and CT must share the same physical grid')
    return volume_metrics(prediction, np.asarray(gt.dataobj), grid)


def evaluate_model(model, dataset, destination, batch_size=8, device='cpu'):
    """Infer every complete patient, export original-grid NIfTI and JSON metrics."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new evaluation destination')
    destination.mkdir(parents=True)
    patients = {}
    for pid, metadata in dataset.records.items():
        working = predict_working_volume(model, dataset, pid, batch_size, device)
        prediction = restore_original_grid(working, metadata)
        metrics = None if dataset.test_mode else evaluate_original_prediction(prediction, metadata)
        save_original_prediction(prediction, metadata, destination / f'{pid}.nii.gz')
        patients[pid] = metrics
    scored = [v for v in patients.values() if v is not None]
    report = {'grid': 'original_CT', 'patients': patients,
              'mean_foreground_dice': float(np.mean([m[str(k)]['dice'] for m in scored for k in range(1, 5)])) if scored else None}
    (destination / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
