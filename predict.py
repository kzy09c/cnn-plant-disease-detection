"""Classify a tomato leaf photograph using the saved CNN."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from train import TomatoCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "outputs/tomato_cnn.pt")
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = TomatoCNN(checkpoint["channels"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    size = checkpoint["image_size"]
    with Image.open(args.image) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0].numpy()
    ranked = sorted(zip(checkpoint["classes"], probabilities.tolist()), key=lambda pair: pair[1], reverse=True)
    print(json.dumps({"prediction": ranked[0][0], "softmax_probability": ranked[0][1],
                      "all_classes": dict(ranked),
                      "scope": "Single tomato leaf classification; field-photo performance has not been evaluated."}, indent=2))


if __name__ == "__main__":
    main()
