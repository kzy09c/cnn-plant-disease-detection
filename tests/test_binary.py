"""Meaningful checks for the shared model, balancing, and frozen result contracts."""
import unittest
import numpy as np
import torch

from model import TomatoCNN, CLASSES
from training import balanced_batches
from verification import verify


class BinaryModelTests(unittest.TestCase):
    def test_model_has_two_outputs_and_recorded_parameter_count(self):
        torch.set_num_threads(2)
        model=TomatoCNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),939874)
        with torch.inference_mode():scores=model(torch.zeros(2,3,128,128)).softmax(1)
        self.assertEqual(tuple(scores.shape),(2,2))
        self.assertTrue(torch.allclose(scores.sum(1),torch.ones(2)))
        self.assertEqual(CLASSES,["Healthy","Unhealthy"])

    def test_balancing_covers_both_pools_without_drawing_other_ids(self):
        healthy=np.arange(10,45)
        unhealthy=np.arange(100,177)
        labels={**{int(i):0 for i in healthy},**{int(i):1 for i in unhealthy}}
        batches=balanced_batches([healthy,unhealthy],64,np.random.default_rng(42))
        self.assertEqual(set(np.concatenate(batches)),set(healthy)|set(unhealthy))
        for batch in batches:
            self.assertEqual(sum(labels[int(i)]==0 for i in batch),32)
            self.assertEqual(sum(labels[int(i)]==1 for i in batch),32)

    def test_saved_binary_results_and_checkpoint_contracts(self):
        for dataset in ("plantvillage","bangladesh"):
            with self.subTest(dataset=dataset):self.assertTrue(verify(dataset)["passed"])


if __name__ == "__main__":
    unittest.main()
