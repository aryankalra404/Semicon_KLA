from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np
import torch

from kla_restore.data import (
    PairedNpyDataset,
    all_pair_names,
    deterministic_split_names,
    names_for_split,
    synthetic_compound_degradation,
)
from kla_restore.losses import RestorationLoss, low_frequency_data_consistency
from kla_restore.ensemble import restore, transform_indices
from kla_restore.metrics import psnr, ssim
from kla_restore.model import (
    KLARestoreNet,
    build_model,
    initialize_v3_from_v2,
    initialize_v4a_from_v2,
    initialize_v4b_from_v2,
    model_config,
    range_aware_input,
)
from kla_restore.runtime import choose_device
from compare_paired import compare, exact_sign_pvalue, read_rows
from train import set_v4b_stage, v4b_parameter_groups


class SplitTests(unittest.TestCase):
    def test_splits_are_disjoint_and_complete(self) -> None:
        train, val = (set(names_for_split(name)) for name in ("train", "val"))
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 3200)
        self.assertEqual((len(train), len(val)), (2880, 320))

    def test_seeded_splits_are_reproducible_and_disjoint(self) -> None:
        first_train, first_val = deterministic_split_names(3407)
        second_train, second_val = deterministic_split_names(3407)
        other_train, other_val = deterministic_split_names(8119)
        self.assertEqual((first_train, first_val), (second_train, second_val))
        self.assertNotEqual(first_val, other_val)
        self.assertFalse(set(first_train) & set(first_val))
        self.assertEqual(set(first_train) | set(first_val), set(all_pair_names()))
        self.assertEqual((len(first_train), len(first_val)), (2880, 320))


class PairedComparisonTests(unittest.TestCase):
    def test_sign_test_is_symmetric(self) -> None:
        self.assertEqual(exact_sign_pvalue(3, 1), exact_sign_pvalue(1, 3))
        self.assertEqual(exact_sign_pvalue(0, 0), 1.0)

    def test_csv_comparison_tracks_metric_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{name}.csv" for name in ("a", "b")]
            rows = (
                (("x.npy", 20.0, 0.5, 0.4), ("y.npy", 22.0, 0.6, 0.3)),
                (("x.npy", 21.0, 0.6, 0.3), ("y.npy", 23.0, 0.7, 0.2)),
            )
            for path, values in zip(paths, rows, strict=True):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("filename", "psnr", "ssim", "lpips"))
                    writer.writerows(values)
            result = compare(
                read_rows(paths[0]),
                read_rows(paths[1]),
                100,
                np.random.default_rng(1),
            )
            self.assertEqual(result["psnr"]["wins"], 2)
            self.assertEqual(result["ssim"]["wins"], 2)
            self.assertEqual(result["lpips"]["wins"], 2)


class DatasetTests(unittest.TestCase):
    def test_pair_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lr").mkdir()
            (root / "gt").mkdir()
            np.save(root / "lr/sample.npy", np.zeros((8, 8), dtype=np.float32))
            np.save(root / "gt/sample.npy", np.zeros((16, 16), dtype=np.float32))
            dataset = PairedNpyDataset(root / "lr", root / "gt", ["sample.npy"])
            lr, gt, name = dataset[0]
            self.assertEqual((lr.shape, gt.shape, name), ((1, 8, 8), (1, 16, 16), "sample.npy"))

    def test_truncated_array_fails_to_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.npy"
            path.write_bytes(b"not a numpy array")
            with self.assertRaises(ValueError):
                np.load(path, allow_pickle=False)

    def test_synthetic_degradation_shape_and_range_freedom(self) -> None:
        gt = torch.rand(1, 32, 32)
        for policy in ("fixed", "randomized"):
            lr = synthetic_compound_degradation(gt, policy=policy)
            self.assertEqual(lr.shape, (1, 16, 16))
            self.assertTrue(torch.isfinite(lr).all())


class ModelAndMetricTests(unittest.TestCase):
    def test_self_ensemble_inverts_geometric_transforms(self) -> None:
        identity = torch.nn.Identity()
        image = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
        for mode, count in (("x1", 1), ("x4", 4), ("x8", 8)):
            self.assertEqual(len(transform_indices(mode)), count)
            self.assertTrue(torch.equal(restore([identity], image, mode), image))

    def test_checkpoint_ensemble_averages_predictions(self) -> None:
        class Add(torch.nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.value = value

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs + self.value

        inputs = torch.zeros(1, 1, 5, 7)
        output = restore([Add(1.0), Add(3.0)], inputs)
        self.assertTrue(torch.equal(output, torch.full_like(inputs, 2.0)))

    def test_model_doubles_resolution(self) -> None:
        model = KLARestoreNet(width=8, blocks=1)
        inputs = torch.rand(2, 1, 12, 10)
        output = model(inputs)
        self.assertEqual(output.shape, (2, 1, 24, 20))
        self.assertTrue(torch.all((0 <= output) & (output <= 1)))

    def test_model_supports_second_kla_resolution_pair(self) -> None:
        model = KLARestoreNet(width=8, blocks=1)
        inputs = torch.rand(1, 1, 256, 256)
        output = model(inputs)
        self.assertEqual(output.shape, (1, 1, 512, 512))

    def test_v3_model_doubles_resolution_and_backpropagates(self) -> None:
        model = KLARestoreNet(
            width=8, blocks=2, variant="v3", condition_dim=8, hr_width=8, hr_blocks=1
        )
        inputs = torch.rand(2, 1, 12, 10)
        target = torch.rand(2, 1, 24, 20)
        output = model(inputs)
        self.assertEqual(output.shape, target.shape)
        output.sub(target).square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_range_aware_input_preserves_raw_and_encodes_excursions(self) -> None:
        inputs = torch.tensor([[[[-0.2, 0.4, 1.3]]]])
        encoded = range_aware_input(inputs)
        self.assertEqual(encoded.shape, (1, 4, 1, 3))
        self.assertTrue(torch.equal(encoded[:, 0:1], inputs))
        self.assertTrue(
            torch.equal(encoded[:, 1], torch.tensor([[[0.0, 0.4, 1.0]]]))
        )
        self.assertTrue(
            torch.allclose(encoded[:, 2], torch.tensor([[[0.0, 0.0, 0.3]]]))
        )
        self.assertTrue(
            torch.equal(encoded[:, 3], torch.tensor([[[0.2, 0.0, 0.0]]]))
        )

    def test_model_config_preserves_all_variants(self) -> None:
        legacy = build_model({"width": 8, "blocks": 1})
        self.assertEqual(
            model_config(legacy), {"variant": "v2", "width": 8, "blocks": 1}
        )
        v3 = KLARestoreNet(
            8, 1, variant="v3", condition_dim=8, hr_width=8, hr_blocks=1
        )
        self.assertEqual(model_config(build_model(model_config(v3))), model_config(v3))
        v4a = KLARestoreNet(8, 1, variant="v4a")
        self.assertEqual(model_config(build_model(model_config(v4a))), model_config(v4a))
        v4b = KLARestoreNet(
            8, 1, variant="v4b", frequency_width=8, frequency_blocks=1
        )
        self.assertEqual(model_config(build_model(model_config(v4b))), model_config(v4b))

    def test_v3_warm_start_exactly_preserves_v2_output(self) -> None:
        torch.manual_seed(3)
        v2 = KLARestoreNet(width=8, blocks=2)
        torch.nn.init.normal_(v2.upsample[-1].weight, std=0.02)
        torch.nn.init.normal_(v2.upsample[-1].bias, std=0.02)
        v3 = KLARestoreNet(
            width=8,
            blocks=2,
            variant="v3",
            condition_dim=8,
            hr_width=8,
            hr_blocks=1,
        )
        initialize_v3_from_v2(v3, v2.state_dict())
        inputs = torch.randn(2, 1, 12, 10)
        self.assertTrue(torch.equal(v2(inputs), v3(inputs)))

    def test_v4a_warm_start_exactly_preserves_v2_output(self) -> None:
        torch.manual_seed(5)
        v2 = KLARestoreNet(width=8, blocks=2)
        torch.nn.init.normal_(v2.upsample[-1].weight, std=0.02)
        torch.nn.init.normal_(v2.upsample[-1].bias, std=0.02)
        v4a = KLARestoreNet(width=8, blocks=2, variant="v4a")
        copied, parameters = initialize_v4a_from_v2(v4a, v2.state_dict())
        self.assertEqual(copied, len(v2.state_dict()))
        self.assertEqual(parameters, sum(x.numel() for x in v2.state_dict().values()))
        inputs = torch.randn(2, 1, 12, 10)
        self.assertTrue(torch.equal(v2(inputs), v4a(inputs)))
        self.assertTrue(torch.count_nonzero(v4a.range_stem.weight) == 0)

    def test_v4b_warm_start_exactly_preserves_v2_output(self) -> None:
        torch.manual_seed(7)
        v2 = KLARestoreNet(width=8, blocks=2)
        torch.nn.init.normal_(v2.upsample[-1].weight, std=0.02)
        torch.nn.init.normal_(v2.upsample[-1].bias, std=0.02)
        v4b = KLARestoreNet(
            width=8,
            blocks=2,
            variant="v4b",
            frequency_width=8,
            frequency_blocks=1,
        )
        copied, parameters = initialize_v4b_from_v2(v4b, v2.state_dict())
        self.assertEqual(copied, len(v2.state_dict()))
        self.assertEqual(parameters, sum(x.numel() for x in v2.state_dict().values()))
        inputs = torch.randn(2, 1, 12, 12)
        self.assertTrue(torch.equal(v2(inputs), v4b(inputs)))
        self.assertTrue(
            torch.count_nonzero(v4b.frequency_branch.project.weight) == 0
        )

    def test_v4b_branch_learns_after_zero_initialized_projection(self) -> None:
        model = KLARestoreNet(
            width=8,
            blocks=1,
            variant="v4b",
            frequency_width=8,
            frequency_blocks=1,
        )
        # Production v4b is always warm-started from a trained v2 whose output
        # projection is nonzero; reproduce that contract in this isolated test.
        torch.nn.init.normal_(model.upsample[-1].weight, std=0.02)
        inputs = torch.rand(2, 1, 12, 12)
        target = torch.rand(2, 1, 24, 24)
        model(inputs).sub(target).square().mean().backward()
        gradient = model.frequency_branch.project.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_v4b_staging_freezes_only_the_backbone(self) -> None:
        model = KLARestoreNet(
            width=8,
            blocks=1,
            variant="v4b",
            frequency_width=8,
            frequency_blocks=1,
        )
        backbone, branch = v4b_parameter_groups(model)
        self.assertFalse(set(backbone) & set(branch))
        self.assertEqual(
            sum(x.numel() for x in backbone + branch),
            sum(x.numel() for x in model.parameters()),
        )
        set_v4b_stage(model, branch_only=True)
        self.assertTrue(all(not x.requires_grad for x in backbone))
        self.assertTrue(all(x.requires_grad for x in branch))
        set_v4b_stage(model, branch_only=False)
        self.assertTrue(all(x.requires_grad for x in backbone + branch))

    def test_data_consistency_is_lower_for_matching_projection(self) -> None:
        high = torch.rand(2, 1, 24, 20)
        observation = torch.nn.functional.interpolate(high, size=(12, 10), mode="area")
        matching = low_frequency_data_consistency(high, observation)
        mismatching = low_frequency_data_consistency(1.0 - high, observation)
        self.assertLess(float(matching), float(mismatching))

    def test_consistency_loss_requires_observation(self) -> None:
        loss = RestorationLoss(consistency_weight=0.05)
        image = torch.rand(1, 1, 16, 16)
        with self.assertRaises(ValueError):
            loss(image, image)

    def test_identical_metrics(self) -> None:
        image = torch.rand(2, 1, 32, 32)
        self.assertTrue(torch.all(psnr(image, image) > 100))
        self.assertTrue(torch.allclose(ssim(image, image), torch.ones(2), atol=1e-5))

    def test_auto_device_is_usable(self) -> None:
        device = choose_device("auto")
        tensor = torch.ones(1).to(device)
        self.assertEqual(tensor.device.type, device.type)


if __name__ == "__main__":
    unittest.main()
