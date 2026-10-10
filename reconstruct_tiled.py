""" 
Blend tile probabilities on the unpadded working grid.

No inverse resampling, original-grid metrics or training changes are performed.
Arrays returned here use XYZ order; probability tiles use KXY order.
"""
import numpy as np
import torch

from preprocessing_metadata import validate_metadata
from tiling import TileLayout


class SliceAccumulator:
    """Accept every tile exactly once, in any tile order, then average chances."""
    def __init__(self, layout, classes=5):
        if type(classes) is not int or not 2 <= classes <= 256:
            raise ValueError('classes must be an integer between 2 and 256')
        self.layout = layout
        self.classes = classes
        self.starts = layout.starts
        self.total = np.zeros((classes, *layout.shape), dtype=np.float64)
        self.count = np.zeros(layout.shape, dtype=np.int32)
        self.seen = set()

    def add(self, tile_index, probabilities):
        if isinstance(tile_index, (bool, np.bool_)) or not isinstance(tile_index, (int, np.integer)) or not 0 <= tile_index < len(self.starts):
            raise ValueError('Invalid tile index')
        if tile_index in self.seen:
            raise ValueError('Duplicate tile prediction')
        p = np.asarray(probabilities)
        if p.shape != (self.classes, *self.layout.size):
            raise ValueError('Probability tile has wrong shape')
        if not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all():
            raise ValueError('Expected finite floating-point probabilities')
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(0), 1, rtol=0, atol=1e-5):
            raise ValueError('Probabilities must be in [0,1] and sum to one')
        x, y = self.starts[tile_index]
        nx, ny = min(self.layout.size[0], self.layout.shape[0]-x), min(self.layout.size[1], self.layout.shape[1]-y)
        self.total[:, x:x+nx, y:y+ny] += p[:, :nx, :ny]
        self.count[x:x+nx, y:y+ny] += 1
        self.seen.add(int(tile_index))

    def probabilities(self):
        if len(self.seen) != len(self.starts) or np.any(self.count == 0):
            raise ValueError('Missing tile predictions; cannot reconstruct a complete slice')
        return (self.total / self.count[None]).astype(np.float32)

    def labels(self):
        # Ties deterministically select the lowest class index, like np.argmax.
        return self.probabilities().argmax(0).astype(np.uint8)


def reconstruct_working_volume(metadata, predictions):
    """Consume (z, tile_index, probabilities) grouped by increasing Z.

    All working slices must occur from Z=0, with every tile exactly once in
    arbitrary tile order within each slice. Only one probability slice is held
    at a time, plus the resulting uint8 XYZ label volume. Output is classes
    0..4, not PNG class*63 encoding. Padding is discarded during accumulation.
    """
    validate_metadata(metadata)
    if metadata['schema_version'] != 2:
        raise ValueError('Reconstruction requires tiled metadata')
    shape = tuple(metadata['working']['shape'])
    layout = TileLayout(shape[:2], metadata['tiling']['size'], metadata['tiling']['stride'])
    result = np.empty(shape, dtype=np.uint8)
    current_z = 0
    accumulator = SliceAccumulator(layout)
    for z, tile_index, probabilities in predictions:
        if isinstance(z, (bool, np.bool_)) or not isinstance(z, (int, np.integer)) or not 0 <= z < shape[2]:
            raise ValueError('Invalid Z index')
        if z != current_z:
            if z != current_z + 1:
                raise ValueError('Predictions must be grouped by increasing Z without gaps')
            result[:, :, current_z] = accumulator.labels()
            current_z = int(z)
            accumulator = SliceAccumulator(layout)
        accumulator.add(tile_index, probabilities)
    result[:, :, current_z] = accumulator.labels()
    if current_z != shape[2] - 1:
        raise ValueError('Missing working slices')
    return result


def predict_working_volume(model, dataset, patient_id, batch_size=8, device='cpu'):
    """Run a supplied model through the 2D/2.5D tileloader and reconstruct XYZ.

    The caller places the model on the requested device. No checkpoint loading,
    optimizer, output writing, inverse resampling or metric calculation occurs.
    A debug/truncated patient is rejected before inference.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if patient_id not in dataset.records:
        raise ValueError('Unknown patient')
    metadata = dataset.records[patient_id]
    depth = metadata['working']['shape'][2]
    tiles = len(metadata['tiling']['starts_xy'])
    indices = {}
    for index, (pid, z, tile) in enumerate(dataset.samples):
        if pid == patient_id:
            if (z, tile) in indices:
                raise ValueError('Duplicate patient sample')
            indices[z, tile] = index
    if set(indices) != {(z, t) for z in range(depth) for t in range(tiles)}:
        raise ValueError('Inference requires all patient tiles; disable debug truncation')
    modes = [(module, module.training) for module in model.modules()]

    def predictions():
        for z in range(depth):
            for start in range(0, tiles, batch_size):
                tile_ids = list(range(start, min(start+batch_size, tiles)))
                images = torch.stack([dataset[indices[z, t]]['images'] for t in tile_ids]).to(device)
                logits = model(images)
                expected = (len(tile_ids), 5, *metadata['tiling']['size'])
                if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                    raise ValueError('Model must return finite B5XY logits matching the tile size')
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for t, p in zip(tile_ids, probabilities):
                    yield z, t, p

    try:
        model.eval()
        with torch.inference_mode():
            return reconstruct_working_volume(metadata, predictions())
    finally:
        for module, training in modes:
            module.training = training
