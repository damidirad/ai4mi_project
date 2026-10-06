> This document records its implementation stage. The full route is now
> connected in [stage 8](tiled_training.md); historical statements below about
> training being blocked applied before that integration.

# Stage 6: reconstruct predictions on the working grid

`reconstruct_tiled.py` combines probability tiles without resizing them.
For every pixel, it sums all covering tiles' class probabilities, divides by
the coverage count, and then selects the highest-probability class. It does
not average hard labels or logits. Ties select the lowest class index.

Only real working pixels enter the accumulator: the high-edge padding is
cropped from each tile. Every tile must be present exactly once per slice.
Wrong shapes, duplicate/invalid tile indices, nonfinite or out-of-range
probabilities, and probability sums different from one are rejected. Missing
tiles and slices prevent a partial volume from being returned.

## Building blocks

- `SliceAccumulator(layout, classes=5)` accepts tiles in any tile order.
  `probabilities()` returns KXY averages; `labels()` returns XY uint8 classes.
- `reconstruct_working_volume(metadata, predictions)` accepts `(z, tile_index,
  probabilities)` records grouped by increasing Z, starting at zero. Tile
  order within a slice is arbitrary. It returns an XYZ uint8 volume of classes
  0..4, not PNG class×63 values. Schema-2 metadata determines the layout.
- `predict_working_volume(model, dataset, patient_id, batch_size=8, device='cpu')`
  calls a supplied model on the existing 2D/2.5D loader, applies softmax per tile,
  and reconstructs one patient's working volume. Model output must be finite
  B5XY logits. The caller places the model on the requested device.

The accumulator holds one probability slice at a time plus the final label
volume, avoiding a full 5-channel probability volume. It sums in float64 and
returns averaged probabilities in float32. Cross-Z shuffling is deliberately
rejected to keep this memory bound explicit.

Inference disables gradients and temporarily selects evaluation mode. Every
module's previous training/evaluation state is restored, also on failure.
Debug-truncated patients are rejected before inference. The helper does not
load checkpoints, move model parameters, write files or create an optimizer.
On train/val datasets the current loader also reads the central label; test
split datasets do not require labels.

```python
from reconstruct_tiled import predict_working_volume

# Supply an already configured model and a complete TiledDataset.
working_labels = predict_working_volume(
    model, dataset, 'Patient_01', batch_size=8, device='cpu'
)
```

This is a callable component, not yet a new inference CLI. No existing training
or preprocessing route changes automatically. In particular, the returned
volume is on the **working grid after resampling**. It must not be saved under
the original CT affine unless both grids coincide. Original-grid restoration,
NIfTI export and volume metrics are now available in
[stage 7](original_grid_evaluation.md). Full tiled training remains blocked
until those pieces are connected.

## Verification

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_reconstruct_tiled test_tiled_losses test_tiled_dataset test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```

Tests cover probability-before-argmax behavior, exact reconstruction of spatial
and Z-varying synthetic class patterns, padding removal, shuffled tile order,
missing/duplicate/invalid predictions, and inference through 2D/2.5D batches.
A synthetic model checks evaluation/gradient state and restoration on errors.
These checks do not measure trained ENet accuracy, original-grid reconstruction
or GPU behavior.
