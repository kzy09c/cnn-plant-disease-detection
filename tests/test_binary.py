"""Meaningful checks for the shared model, balancing, and frozen result contracts."""
import unittest
import numpy as np
import torch

from model import TomatoCNN, CLASSES
from training import balanced_batches
from verification import verify
from io_utils import ROOT
from transfer_config import RECIPES, initialize_transfer
from PIL import Image
from metrics import image_hash, distances


class BinaryModelTests(unittest.TestCase):
    def test_overlap_fingerprints_accept_pil_images_and_distinguish_content(self):
        pattern=np.random.default_rng(42).integers(0,256,(128,128,3),dtype=np.uint8)
        fingerprint=image_hash(Image.fromarray(pattern))
        self.assertEqual(fingerprint.shape,(8,))
        self.assertEqual(int(distances(fingerprint[None],fingerprint[None])[0,0]),0)
        other=image_hash(Image.fromarray(np.flip(pattern,axis=1).copy()))
        self.assertGreater(int(distances(fingerprint[None],other[None])[0,0]),4)

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

    def test_full_transfer_retains_binary_head_and_covers_all_trainable_parameters(self):
        checkpoint=torch.load(ROOT/"outputs/tomato_cnn.pt",map_location="cpu",weights_only=True)
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                model=TomatoCNN()
                optimizer=initialize_transfer(model,checkpoint,recipe)
                self.assertTrue(all(torch.equal(value,checkpoint["state_dict"][name]) for name,value in model.state_dict().items()))
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                optimized=[id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimized),len(set(optimized)))
                self.assertEqual(set(optimized),{id(p) for p in model.parameters()})
                self.assertEqual([g["lr"] for g in optimizer.param_groups],list(RECIPES[recipe].values()))


if __name__ == "__main__":
    unittest.main()
