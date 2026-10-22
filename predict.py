"""Classify a tomato leaf image with one of the two saved binary CNNs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from model import TomatoCNN, CLASSES

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", choices=["plantvillage", "bangladesh"], default="plantvillage")
    selection.add_argument("--model", type=Path, help="Optional custom binary checkpoint")
    args = parser.parse_args()
    default = "outputs/tomato_cnn.pt" if args.dataset == "plantvillage" else "bangladesh_outputs/tomato_cnn_bangladesh.pt"
    model_path = args.model or ROOT / default
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        parser.error("The checkpoint must use exactly Healthy and Unhealthy, in that order")
    model = TomatoCNN(checkpoint["channels"], 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(6)
    size = checkpoint["image_size"]
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
    with torch.inference_mode():
        scores = model((tensor-.5)/.5).softmax(1)[0].numpy()
    print(json.dumps({"model": str(model_path), "prediction": CLASSES[int(scores.argmax())],
                      "softmax_probability": float(scores.max()), "all_classes": dict(zip(CLASSES, scores.tolist())),
                      "scope": checkpoint.get("scope", "Single tomato leaf health classification")}, indent=2))

if __name__ == "__main__":
    main()
