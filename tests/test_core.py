from __future__ import annotations

import tempfile
import unittest
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
    model_config,
    range_aware_input,
)
from kla_restore.runtime import choose_device


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
