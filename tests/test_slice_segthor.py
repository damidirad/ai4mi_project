import io
import pickle
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import slice_segthor as pipeline


class BaselineTests(unittest.TestCase):
    def test_hu_windowing(self):
        ct = np.array(
            [-1000, -400, -200, 0, 200, 400, 1000],
            dtype=np.int16,
        )

        actual = pipeline.norm_arr(ct)

        self.assertEqual(actual.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(
            actual,
            [0, 0, 63, 127, 191, 255, 255],
        )

    def test_slice_patient_baseline(self):
        depth = 135
        spacing = (1.0, 1.0, 2.5)

        # Alternating columns: linear resize averages each pair.
        # After windowing, -400 and 400 become 0 and 255.
        ct_plane = np.empty((512, 512), dtype=np.int16)
        ct_plane[:, 0::2] = -400
        ct_plane[:, 1::2] = 400
        ct = np.broadcast_to(
            ct_plane[:, :, None], (512, 512, depth)
        ).copy()

        # Distinct first/last slices detect ordering and slice loss.
        ct[:, :, 0] = -1000
        ct[:, :, -1] = 1000

        # Each pair has different labels: nearest-neighbor selects
        # the odd column when resizing from 512 to 256.
        gt_plane = np.empty((512, 512), dtype=np.uint8)
        labels = (np.arange(256) % 5).astype(np.uint8)
        gt_plane[:, 0::2] = (labels + 1) % 5
        gt_plane[:, 1::2] = labels
        gt = np.broadcast_to(
            gt_plane[:, :, None], ct.shape
        )

        expected_gt = np.broadcast_to(
            labels * 63, (256, 256)
        )

        image = SimpleNamespace(
            dataobj=ct,
            header=SimpleNamespace(get_zooms=lambda: spacing),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            patient = source / "train" / "Patient_01"
            patient.mkdir(parents=True)
            (patient / "labels.seg.nrrd").touch()

            # Both calls must independently match the same reference.
            for name, options in (
                ("default", {}),
                ("explicit_off", {"resample": False}),
            ):
                with self.subTest(mode=name):
                    destination = root / name
                    written = []

                    def check_output(filename, data):
                        path = Path(filename)
                        index = int(path.stem.rsplit("_", 1)[1])

                        self.assertEqual(data.shape, (256, 256))
                        self.assertEqual(
                            data.dtype, np.dtype("uint8")
                        )

                        if path.parent.name == "img":
                            value = (
                                0 if index == 0
                                else 255 if index == depth - 1
                                else 127
                            )
                            np.testing.assert_array_equal(
                                data,
                                np.full(
                                    (256, 256), value, dtype=np.uint8
                                ),
                            )
                        else:
                            self.assertEqual(path.parent.name, "gt")
                            np.testing.assert_array_equal(
                                data, expected_gt
                            )

                        written.append(path.relative_to(destination))

                    with (
                        patch.object(
                            pipeline.nib, "load", return_value=image
                        ),
                        patch.object(
                            pipeline,
                            "build_gt_from_segnrrd",
                            return_value=gt,
                        ),
                        patch.object(
                            pipeline, "imsave", side_effect=check_output
                        ),
                        patch.object(
                            pipeline,
                            "resample_voxels",
                            side_effect=AssertionError(
                                "Baseline must not resample"
                            ),
                        ),
                    ):
                        result = pipeline.slice_patient(
                            "Patient_01",
                            destination,
                            source,
                            (256, 256),
                            **options,
                        )

                    self.assertEqual(result, spacing)
                    self.assertEqual(
                        written,
                        [
                            Path(folder) / f"Patient_01_{index:04d}.png"
                            for index in range(depth)
                            for folder in ("img", "gt")
                        ],
                    )

    def test_main_preserves_original_spacing_metadata(self):
        original_spacings = {
            "Patient_01": (1.0, 1.0, 2.5),
            "Patient_02": (1.25, 1.25, 3.0),
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            destination = Path(directory) / "output"

            args = Namespace(
                source_dir=str(source),
                dest_dir=str(destination),
                shape=[256, 256],
                retains=1,
                fold=0,
                process=1,
                resample=False,
                target_spacing=None,
            )

            def fake_slice(id_, *, dest_path, **kwargs):
                self.assertFalse(kwargs["resample"])
                self.assertEqual(
                    dest_path.name,
                    "train" if id_ == "Patient_01" else "val",
                )
                dest_path.mkdir(parents=True, exist_ok=True)
                return original_spacings[id_]

            with (
                patch.object(
                    pipeline,
                    "get_splits",
                    return_value=(
                        ["Patient_01"], ["Patient_02"], []
                    ),
                ),
                patch.object(
                    pipeline, "slice_patient", side_effect=fake_slice
                ) as slicer,
                patch.object(
                    pipeline,
                    "compute_median_spacing",
                    side_effect=AssertionError(
                        "Baseline must not compute target spacing"
                    ),
                ),
            ):
                pipeline.main(args)

            self.assertEqual(slicer.call_count, 2)
            with (destination / "spacing.pkl").open("rb") as handle:
                actual = pickle.load(handle)

            self.assertEqual(actual, original_spacings)


class TargetSpacingTests(unittest.TestCase):
    def test_validate_spacing_accepts_positive_finite_values(self):
        self.assertEqual(
            pipeline.validate_spacing(
                [0.75, 1.25], dimensions=2, name="Target"
            ),
            (0.75, 1.25),
        )

    def test_validate_spacing_rejects_invalid_values_on_every_axis(self):
        for dimensions in (2, 3):
            for axis in range(dimensions):
                for value in (0, -1, np.nan, np.inf, -np.inf):
                    with self.subTest(dimensions=dimensions, axis=axis, value=value):
                        spacing = [1.0] * dimensions
                        spacing[axis] = value
                        with self.assertRaisesRegex(ValueError, "finite, positive"):
                            pipeline.validate_spacing(
                                spacing, dimensions=dimensions, name="Spacing"
                            )

    def test_validate_spacing_rejects_wrong_dimensions(self):
        for spacing in ([], [1.0], [1.0, 1.0, 1.0], [[1.0, 1.0]]):
            with self.subTest(spacing=spacing):
                with self.assertRaisesRegex(ValueError, "2 values"):
                    pipeline.validate_spacing(
                        spacing, dimensions=2, name="Target"
                    )

    def test_median_spacing_rejects_empty_training_set(self):
        with patch.object(pipeline.nib, "load") as loader:
            with self.assertRaisesRegex(ValueError, "empty training set"):
                pipeline.compute_median_spacing([], Path("source"))
            loader.assert_not_called()

    def test_median_spacing_rejects_invalid_source_spacing(self):
        for axis in range(3):
            for value in (0, -1, np.nan, np.inf, -np.inf):
                with self.subTest(axis=axis, value=value):
                    spacing = [1.0, 1.0, 2.5]
                    spacing[axis] = value
                    image = SimpleNamespace(
                        header=SimpleNamespace(get_zooms=lambda: spacing)
                    )
                    with patch.object(pipeline.nib, "load", return_value=image):
                        with self.assertRaisesRegex(ValueError, "Patient_01"):
                            pipeline.compute_median_spacing(
                                ["Patient_01"], Path("source")
                            )

    def test_main_uses_training_xy_median_or_manual_override(self):
        # Different axis medians detect swapped axes; held-out values would
        # substantially change the result if they leaked into the estimate.
        spacings = {
            "Train_01": (0.75, 1.0, 2.0),
            "Train_02": (1.25, 1.5, 3.0),
            "Validation": (10.0, 20.0, 4.0),
            "Test": (30.0, 40.0, 5.0),
        }
        for override, expected, source_label in (
            (None, (1.0, 1.25), "training-set median"),
            ([0.8, 0.9], (0.8, 0.9), "manual override"),
        ):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                source.mkdir()
                destination = Path(directory) / "output"
                args = Namespace(
                    source_dir=str(source), dest_dir=str(destination),
                    shape=[256, 256], retains=1, fold=0, process=1,
                    resample=True, target_spacing=override,
                )
                loaded = []
                processed = []

                def load_header(filename):
                    path = Path(filename)
                    patient = path.parent.name
                    self.assertEqual(
                        path, source / "train" / patient / f"{patient}.nii.gz"
                    )
                    loaded.append(patient)
                    return SimpleNamespace(
                        header=SimpleNamespace(get_zooms=lambda: spacings[patient])
                    )

                def fake_slice(id_, *, dest_path, **kwargs):
                    self.assertTrue(kwargs["resample"])
                    self.assertEqual(kwargs["target_spacing"], expected)
                    processed.append((id_, dest_path.name))
                    dest_path.mkdir(parents=True, exist_ok=True)
                    return spacings[id_]

                output = io.StringIO()
                with (
                    patch.object(pipeline, "get_splits", return_value=(
                        ["Train_01", "Train_02"], ["Validation"], ["Test"]
                    )),
                    patch.object(pipeline.nib, "load", side_effect=load_header),
                    patch.object(pipeline, "slice_patient", side_effect=fake_slice),
                    patch.object(
                        pipeline, "compute_median_spacing",
                        wraps=pipeline.compute_median_spacing,
                    ) as median,
                    redirect_stdout(output),
                ):
                    pipeline.main(args)

                if override is None:
                    median.assert_called_once_with(["Train_01", "Train_02"], source)
                    self.assertEqual(loaded, ["Train_01", "Train_02"])
                else:
                    median.assert_not_called()
                    self.assertEqual(loaded, [])
                self.assertEqual(processed, [
                    ("Train_01", "train"), ("Train_02", "train"),
                    ("Validation", "val"),
                ])
                self.assertIn(source_label, output.getvalue())
                self.assertIn(str(expected), output.getvalue())

    def test_main_rejects_invalid_override_before_processing(self):
        for override in ([0, 1], [1, -1], [np.nan, 1], [1, np.inf], [1, 1, 1]):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                args = Namespace(
                    source_dir=directory,
                    dest_dir=str(Path(directory) / "output"),
                    retains=1, fold=0, resample=True, target_spacing=override,
                )
                with (
                    patch.object(pipeline, "get_splits", return_value=(
                        ["Train"], ["Validation"], []
                    )),
                    patch.object(pipeline, "slice_patient") as slicer,
                    patch.object(pipeline, "compute_median_spacing") as median,
                ):
                    with self.assertRaisesRegex(ValueError, "Manual target"):
                        pipeline.main(args)
                    slicer.assert_not_called()
                    median.assert_not_called()
                self.assertFalse(Path(args.dest_dir).exists())


if __name__ == "__main__":
    unittest.main()
