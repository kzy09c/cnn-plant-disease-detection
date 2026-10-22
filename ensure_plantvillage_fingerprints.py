"""Validate/create fingerprints used to audit Bangladesh pretraining overlap."""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np
from PIL import Image, ImageOps

from io_utils import ROOT, read_csv
from metrics import image_hash


def ensure_fingerprints():
    manifest = read_csv(ROOT / "outputs/split_manifest.csv")
    expected = {r["pixel_sha256"] for r in manifest}
    cache = ROOT / "data/plantvillage/fingerprints.npz"
    if cache.exists():
        with np.load(cache) as saved:
            assert set(saved["pixels"].tolist()) == expected
        return cache
    paths = sorted((ROOT / "data/raw/color").glob("Tomato___*/*"))
    if len(paths) != len(manifest):
        raise FileNotFoundError("Run python download_data.py first; original images are needed for the overlap audit")

    def fingerprint(path):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return image_hash(image), hashlib.sha256(image.tobytes()).hexdigest(), np.asarray(image.convert("L").resize((64, 64), Image.Resampling.LANCZOS))

    with ThreadPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(fingerprint, paths))
    pixels = np.array([v[1] for v in values])
    assert set(pixels.tolist()) == expected
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, hashes=np.stack([v[0] for v in values]), pixels=pixels, thumbnails=np.stack([v[2] for v in values]))
    return cache


if __name__ == "__main__":
    print(ensure_fingerprints())
