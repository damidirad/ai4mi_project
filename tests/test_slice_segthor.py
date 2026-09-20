import pickle
import tempfile
import unittest
from argparse import Namespace
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


if __name__ == "__main__":
    unittest.main()