"""Versioned, validated per-patient geometry for optional preprocessing output."""
import json
from pathlib import Path

import numpy as np

from geometry import Grid, grid_after_resampling, voxel_mapping
from tiling import TileLayout


SCHEMA_VERSION = 1


def resized_slice_grid(working, shape):
    """skimage resize uses cell-centred coordinates (SciPy grid_mode=True)."""
    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError('Slice shape must contain two dimensions')
    # Grid validates positive integral dimensions.
    output_shape = Grid((*shape, working.shape[2]), working.affine).shape
    scale = np.array(working.shape) / output_shape
    mapping = np.diag([*scale, 1.])
    mapping[:3, 3] = (scale - 1) / 2
    return Grid(output_shape, working.affine @ mapping)


def _encode(grid):
    return {'shape': list(grid.shape), 'affine_ras_mm': grid.affine.tolist()}


def build_metadata(patient_id, split, source_path, original, working_shape,
                   target_spacing, slice_shape, hu_window, source_spacing=None, tile_layout=None):
    mode = 'none' if target_spacing is None else {2: 'xy', 3: 'xyz'}.get(len(target_spacing))
    working = grid_after_resampling(original, working_shape)
    output = resized_slice_grid(working, slice_shape)
    value = {
        'schema_version': SCHEMA_VERSION,
        'patient_id': patient_id,
        'split': split,
        'source_ct': str(Path(source_path).resolve()),
        'coordinate_system': 'RAS',
        'spatial_units': 'mm',
        'array_axes': 'XYZ',
        'resample': mode,
        'resampling_source_spacing_mm': list(map(float, original.spacing if source_spacing is None else source_spacing)),
        'requested_spacing_mm': None if target_spacing is None else list(map(float, target_spacing)),
        'intensity': {'method': 'volume_minmax' if hu_window is None else 'hu_window',
                      'window': None if hu_window is None else list(map(float, hu_window))},
        'interpolation': {'resampling': 'scipy_zoom_grid_mode_false',
                          'slice_resize': 'skimage_resize_grid_mode_true',
                          'ct_order': 1, 'label_order': 0, 'antialias': False},
        'png_label_multiplier': 63,
        'original': _encode(original),
        'working': _encode(working),
        'output': _encode(output),
        'output_to_original': voxel_mapping(original, output).tolist(),
    }
    if tile_layout is not None:
        if tile_layout.shape != working.shape[:2] or output.shape != working.shape:
            raise ValueError('Tiled output must preserve the working grid')
        value['schema_version'] = 2
        value['layout'] = 'tiles'
        value['tiling'] = tile_layout.to_dict()
        value['interpolation']['slice_resize'] = 'none'
    validate_metadata(value)
    return value


def validate_metadata(value):
    """Reject missing, unknown-version or internally inconsistent records."""
    try:
        if type(value['schema_version']) is not int or value['schema_version'] not in (SCHEMA_VERSION, 2):
            raise ValueError('Unsupported metadata schema version')
        if (value['coordinate_system'], value['spatial_units'], value['array_axes']) != ('RAS', 'mm', 'XYZ'):
            raise ValueError('Unsupported coordinate convention')
        for key in ('patient_id', 'source_ct'):
            if not isinstance(value[key], str) or not value[key]:
                raise ValueError(f'Invalid {key}')
        if value['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        original, working, output = [Grid(value[key]['shape'], value[key]['affine_ras_mm'])
                                     for key in ('original', 'working', 'output')]
        source_spacing = np.asarray(value['resampling_source_spacing_mm'], dtype=float)
        if source_spacing.shape != (3,) or not np.isfinite(source_spacing).all() or np.any(source_spacing <= 0):
            raise ValueError('Invalid source spacing')
        if not np.allclose(source_spacing, original.spacing, rtol=1e-6, atol=1e-8):
            raise ValueError('Source spacing and affine disagree')
        mode, target = value['resample'], value['requested_spacing_mm']
        if mode == 'none':
            if target is not None or working.shape != original.shape:
                raise ValueError('No-resampling mode must preserve the original grid')
        elif mode in ('xy', 'xyz'):
            target = np.asarray(target, dtype=float)
            dims = 2 if mode == 'xy' else 3
            if target.shape != (dims,) or not np.isfinite(target).all() or np.any(target <= 0):
                raise ValueError('Invalid requested spacing')
            expected_shape = list(original.shape)
            for axis in range(dims):
                expected_shape[axis] = int(round(original.shape[axis] * source_spacing[axis] / target[axis]))
            if tuple(expected_shape) != working.shape:
                raise ValueError('Working shape does not match requested resampling')
        else:
            raise ValueError('Invalid resampling mode')
        expected_working = grid_after_resampling(original, working.shape)
        tiled = value['schema_version'] == 2
        if tiled:
            if value['layout'] != 'tiles':
                raise ValueError('Schema 2 requires tiled layout')
            layout = TileLayout(working.shape[:2], value['tiling']['size'], value['tiling']['stride'])
            if value['tiling'] != layout.to_dict():
                raise ValueError('Inconsistent tile positions, padding or storage convention')
        expected_output = working if tiled else resized_slice_grid(working, output.shape[:2])
        for actual, expected in [(working, expected_working), (output, expected_output)]:
            if actual.shape != expected.shape or not np.allclose(actual.affine, expected.affine, rtol=0, atol=1e-8):
                raise ValueError('Inconsistent grid geometry')
        mapping = np.asarray(value['output_to_original'], dtype=float)
        if mapping.shape != (4, 4) or not np.allclose(mapping, voxel_mapping(original, output), rtol=0, atol=1e-8):
            raise ValueError('Inconsistent voxel mapping')
        intensity = value['intensity']
        if intensity['method'] == 'volume_minmax':
            if intensity['window'] is not None:
                raise ValueError('Unexpected intensity window')
        elif intensity['method'] == 'hu_window':
            window = np.asarray(intensity['window'], dtype=float)
            if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
                raise ValueError('Invalid HU window')
        else:
            raise ValueError('Unknown intensity normalization')
        if value['interpolation'] != {'resampling': 'scipy_zoom_grid_mode_false',
                'slice_resize': 'none' if tiled else 'skimage_resize_grid_mode_true', 'ct_order': 1,
                'label_order': 0, 'antialias': False} or value['png_label_multiplier'] != 63:
            raise ValueError('Unsupported interpolation or label encoding')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Incomplete or malformed preprocessing metadata') from error
    return value


def save_metadata(path, value):
    validate_metadata(value)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only the complete JSON; patient-specific paths support Pool workers.
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_metadata(path):
    return validate_metadata(json.loads(Path(path).read_text()))
