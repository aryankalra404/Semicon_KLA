from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from kla_restore.data import PairedNpyDataset, names_for_split
from kla_restore.metrics import psnr, ssim
from kla_restore.model import KLARestoreNet
from kla_restore.runtime import choose_device


class SplitTests(unittest.TestCase):
    def test_splits_are_disjoint_and_complete(self) -> None:
        train, val = (set(names_for_split(name)) for name in ("train", "val"))
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 3200)
        self.assertEqual((len(train), len(val)), (2880, 320))


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


class ModelAndMetricTests(unittest.TestCase):
    def test_model_doubles_resolution(self) -> None:
        model = KLARestoreNet(width=8, blocks=1)
        inputs = torch.rand(2, 1, 12, 10)
        output = model(inputs)
        self.assertEqual(output.shape, (2, 1, 24, 20))
        self.assertTrue(torch.all((0 <= output) & (output <= 1)))

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
