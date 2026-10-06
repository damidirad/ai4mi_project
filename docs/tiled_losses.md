# Stage 5: padding and overlap in tiled cross-entropy

The loader now supplies `pixel_weights` with shape `[B,X,Y]`: inverse coverage
on real pixels and zero on padding. Coverage comes from the validated tile
layout and is cached once per patient, shared across Z. Across all tiles the
weights for a real working voxel sum to one.

`TiledDataset.class_counts()` reads the selected label tiles with these weights.
On a complete dataset each working voxel contributes once, without counting
padding as background. This also makes the training-set inverse-square-root
class weighting independent of how often a voxel appears in overlapping tiles.
Class counts read label PNGs, so this is an additional dataset-wide I/O pass.

## Loss and sampling contract

`TiledCrossEntropy` consumes **logits**, one-hot targets and pixel weights.
It uses stable log-softmax and supports both selected supervised classes and
optional nonnegative class weights. Existing losses in `losses.py` are unchanged.

Let N be the number of dataset tile entries, c(p) the number of tiles covering
pixel p, and D the sum of supervised class weights over all real dataset voxels.
For a batch of B uniformly sampled tile entries, the loss is:

`N / (B * D) * sum(tile pixels: class_weight[label] * CE / c(p))`.

Padding and unsupervised target classes contribute zero. D is fixed for the
dataset, not estimated from the batch. A batch/tile-local denominator would
upweight small edge tiles and change the intended objective. Average reported
batch losses by their sample counts, especially for a short final batch.

This objective weights working voxels equally before class weighting; it does
not weight patients or slices equally. Larger volumes contribute more. The
contract assumes uniform sampling over the current dataset entries, consistent
with shuffled full tile epochs. A future patient/slice-balanced or foreground
sampler must change the normalization/sampling correction explicitly.

For the same per-voxel predictions, summing tile contributions reproduces global
weighted CE, including partial supervision. Actual network predictions can
differ across tile contexts; this is not CE of the averaged inference
probabilities. Zero padding weights remove direct loss/logit gradients from
padded positions. They do not make padded input values invisible to convolution
receptive fields at nearby real pixels.

Debug mode counts only its selected samples with the full-grid overlap weights.
It is a diagnostic subset objective, not a representative whole-volume score.

## Supported combinations

Only `ce` is currently supported by the tiled loss factory. `dice`, `ce_dice`,
`boundary` and `ce_dice_boundary` fail explicitly. Dice needs a separate
aggregation definition because a mean of tile ratios is not global Dice.
Boundary losses require full-grid distance information; computing distances
inside cropped tiles creates artificial boundaries. Existing full-slice loss
choices remain available unchanged.

## Optional CLI diagnostic

```bash
python main.py --dataset SEGTHOR --data_root data --dest results/check \
  --tiling --check-data --check-loss --loss ce --slices 3 --workers 0
```

`--check-loss` requires `--tiling --check-data`. It computes corrected class
counts, derives class weights from training only, and evaluates CE/backward on
zero diagnostic logits for the loaded batches. Validation uses the training
class weights and its own normalization mass. No network, optimizer, training
or output files are created by this command. The value is a diagnostic, not a
model performance score. Without `--check-loss`, batch checking retains its
previous scope.

Full tiled training still stops before model setup: reconstruction and volume
metric integration remain stages 6–7. This commit introduces the reusable loss
and dataset weights, not a partially working full training run.

## Verification

```bash
PYTHONPATH=tests ai4mi/bin/python -m unittest test_tiled_losses test_tiled_dataset test_tiling test_preprocessing_metadata test_geometry test_resampling_cli -v
```

Tests verify unit summed overlap weights; zero padding and ignored-class logit
gradients; global CE and gradient equivalence across overlap patterns, unequal
volumes, class weighting and partial supervision; short-batch aggregation;
correct class counts; unsupported-loss errors; CLI diagnostic integration;
and one synthetic Conv2d/SGD step reducing CE. The synthetic model step does not
validate ENet convergence, real-data training or GPU behavior.
