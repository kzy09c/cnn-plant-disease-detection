"""Load the recorded PlantVillage split or audit the Bangladesh publisher split."""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np
from PIL import Image, ImageOps

from io_utils import ROOT, read_csv, save
from model import CLASSES
from prepare_bangladesh import prepare, write_rows


def load_dataset(dataset, size, output):
    if dataset == "bangladesh":
        return prepare(size, output=output)
    rows = read_csv(ROOT / "outputs/split_manifest.csv")

    def load(row):
        assert row["binary_class"] == ("Healthy" if row["source_class"] == "Tomato___healthy" else "Unhealthy")
        path = ROOT / row["path"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == row["sha256"], row["path"]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))

    with ThreadPoolExecutor(max_workers=12) as pool:
        arrays = np.stack(list(pool.map(load, rows))).transpose(0, 3, 1, 2).copy()
    labels = np.array([CLASSES.index(r["binary_class"]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    development = np.array([i for i,r in enumerate(rows) if r["split"] == "development"])
    test = np.array([i for i,r in enumerate(rows) if r["split"] == "test"])
    assert not set(groups[development]) & set(groups[test])
    folds = [(np.array([i for i in development if int(rows[i]["cv_validation_fold"]) != f]),
              np.array([i for i in development if int(rows[i]["cv_validation_fold"]) == f])) for f in range(1, 6)]
    for training,validation in folds:
        assert not set(groups[training]) & set(groups[validation])
    write_rows(output / "split_manifest.csv", rows)
    audit = {"dataset":"PlantVillage binary tomato leaves", "development_images":len(development), "test_images":len(test),
             "binary_counts":{c:{"development":int(sum(labels[development]==j)),"test":int(sum(labels[test]==j))} for j,c in enumerate(CLASSES)},
             "grouping":"Preserved original same-leaf/duplicate groups and five folds; incomplete source metadata",
             "test_status":"Reused benchmark test, not a fresh independent cohort"}
    save(output / "data_audit.json", audit)
    return rows, arrays, labels, groups, development, test, test, folds, audit
