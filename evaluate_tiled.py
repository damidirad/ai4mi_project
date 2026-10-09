"""Restore working labels to original CT geometry and evaluate whole volumes."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform, distance_transform_edt

from geometry import Grid, grid_from_nifti, voxel_mapping
from preprocessing_metadata import validate_metadata, load_metadata
from reconstruct_tiled import predict_working_volume
from utils import surface


def _labels(array, shape):
    array = np.asarray(array)
    if array.shape != tuple(shape) or not np.issubdtype(array.dtype, np.integer) or not np.isin(array, range(5)).all():
        raise ValueError('Expected integer class labels 0..4 on the specified grid')
    return array


def restore_original_grid(working_labels, metadata):
    validate_metadata(metadata)
    original = Grid(metadata['original']['shape'], metadata['original']['affine_ras_mm'])
    working = Grid(metadata['output']['shape'], metadata['output']['affine_ras_mm'])
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


def load_full_metadata(directory, split):
    """Load full-slice geometry and reject missing or misidentified patients."""
    paths = sorted(Path(directory).glob('*.json'))
    if not paths:
        raise ValueError('Original-grid evaluation requires full-slice metadata')
    records = {}
    for path in paths:
        record = load_metadata(path)
        if record['schema_version'] != 1 or record['split'] != split or record['patient_id'] != path.stem:
            raise ValueError('Expected matching full-slice metadata for the requested split')
        records[path.stem] = record
    return records


def evaluate_full_predictions(records, predictions, destination, test_mode=False):
    """Restore complete full-slice label volumes and use the tiled route's metrics."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new evaluation destination')
    destination.mkdir(parents=True)
    patients = {}
    for pid, labels in predictions:
        if pid not in records or pid in patients:
            raise ValueError('Unknown or duplicate prediction patient')
        record = records[pid]
        prediction = restore_original_grid(labels, record)
        metrics = None if test_mode else evaluate_original_prediction(prediction, record)
        save_original_prediction(prediction, record, destination / f'{pid}.nii.gz')
        patients[pid] = metrics
    if set(patients) != set(records):
        raise ValueError('Missing patient predictions')
    scored = [value for value in patients.values() if value is not None]
    report = {'grid': 'original_CT', 'patients': patients,
              'mean_foreground_dice': float(np.mean([m[str(k)]['dice'] for m in scored
                                                    for k in range(1, 5)])) if scored else None}
    (destination / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def evaluate_full_model(model, dataset, records, destination, batch_size=8, device='cpu'):
    """Predict the central slice with the existing 2D/2.5D full-slice loader."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    indices = {path.stem: index for index, (path, _) in enumerate(dataset.files)}
    expected = {f'{pid}_{z:04d}' for pid, record in records.items()
                for z in range(record['output']['shape'][2])}
    if len(indices) != len(dataset.files) or set(indices) != expected:
        raise ValueError('Inference requires exactly all patient slices')
    modes = [(module, module.training) for module in model.modules()]

    def predictions():
        for pid, record in records.items():
            shape = tuple(record['output']['shape'])
            labels = np.empty(shape, dtype=np.uint8)
            for start in range(0, shape[2], batch_size):
                zs = list(range(start, min(start + batch_size, shape[2])))
                images = torch.stack([dataset._load_img_stack(indices[f'{pid}_{z:04d}']) for z in zs]).to(device)
                logits = model(images)
                if tuple(logits.shape) != (len(zs), 5, *shape[:2]) or not torch.isfinite(logits).all():
                    raise ValueError('Expected finite B5XY logits matching full-slice dimensions')
                labels[:, :, zs] = logits.argmax(1).cpu().numpy().transpose(1, 2, 0)
            yield pid, labels
    try:
        model.eval()
        with torch.inference_mode():
            return evaluate_full_predictions(records, predictions(), destination, dataset.test_mode)
    finally:
        for module, training in modes:
            module.training = training
