"""Predeclared full-model fine-tuning recipes; both retain the binary head."""
import torch

from model import CLASSES, CHANNELS

RECIPES = {
    "full_finetune": {"features_lr": 1e-4, "classifier_lr": 1e-4},
    "differential_finetune": {"features_lr": 3e-5, "classifier_lr": 3e-4},
}


def initialize_transfer(model, checkpoint, recipe):
    if checkpoint["classes"] != CLASSES or tuple(checkpoint["channels"]) != CHANNELS:
        raise ValueError("Pretraining checkpoint must be the matching binary tomato CNN")
    if checkpoint["image_size"] != 128:
        raise ValueError("Full-model transfer requires the recorded 128-pixel input")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    rates = RECIPES[recipe]
    return torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": rates["features_lr"]},
        {"params": model.classifier.parameters(), "lr": rates["classifier_lr"]},
    ], weight_decay=1e-4)
