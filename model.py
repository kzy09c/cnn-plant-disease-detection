"""Shared two-output tomato CNN and the recorded preprocessing."""
import random
import numpy as np
import torch
from torch import nn

CLASSES = ["Healthy", "Unhealthy"]
CHANNELS = (32, 64, 128, 192)

class TomatoCNN(nn.Module):

    def __init__(self, channels=(32, 64, 128, 192), classes=2):
        super().__init__()
        blocks = []
        previous = 3
        for channel in channels:
            blocks.extend([nn.Conv2d(previous, channel, 3, padding=1, bias=False), nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 3, padding=1, bias=False), nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            previous = channel
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(previous * 4, 128), nn.ReLU(), nn.Dropout(0.35), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_batch(images, indices, augment):
    x = images[indices].float().div_(255)
    if augment:
        for dimension in (2, 3):
            mask = torch.rand(len(x), device=x.device) < 0.5
            x[mask] = x[mask].flip(dimension)
        mask = torch.rand(len(x), device=x.device) < 0.5
        x[mask] = x[mask].transpose(2, 3)
        brightness = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.85, 1.15)
        colour = torch.empty((len(x), 3, 1, 1), device=x.device).uniform_(0.95, 1.05)
        x = (x * brightness * colour).clamp_(0, 1)
    return x.sub_(0.5).div_(0.5).contiguous(memory_format=torch.channels_last)
