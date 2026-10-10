"""Independently audit saved predictions and split boundaries; no model inference."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    info = json.loads((OUT / "experiment.json").read_text())
    metrics = json.loads((OUT / "test_metrics.json").read_text())
    manifest = list(csv.DictReader((OUT / "split_manifest.csv").open(encoding="utf-8")))
    predictions = list(csv.DictReader((OUT / "test_predictions.csv").open(encoding="utf-8")))
    assert len(manifest) == info["images"]
    partitions = defaultdict(set)
    for row in manifest:
        partitions["test" if row["split"] == "test" else row["cv_validation_fold"]].add(row["group"])
    for name, groups in partitions.items():
        for other, other_groups in partitions.items():
            if name != other:
                assert not groups & other_groups, f"Overlapping groups: {name} / {other}"
    for hash_key in ["sha256", "pixel_sha256"]:
        locations = defaultdict(set)
        for row in manifest:
            locations[row[hash_key]].add("test" if row["split"] == "test" else row["cv_validation_fold"])
        assert all(len(values) == 1 for values in locations.values()), f"Duplicate leakage: {hash_key}"
    expected_paths = {row["path"] for row in manifest if row["split"] == "test"}
    manifest_by_path = {row["path"]: row for row in manifest}
    assert len(predictions) == len(expected_paths) == metrics["test_images"]
    assert {row["path"] for row in predictions} == expected_paths
    assert len({row["path"] for row in predictions}) == len(predictions)
    classes = info["classes"]
    matrix = np.zeros((10, 10), dtype=int)
    for row in predictions:
        assert row["true_class"] == manifest_by_path[row["path"]]["class"]
        assert row["group"] == manifest_by_path[row["path"]]["group"]
        probabilities = np.array([float(row["probability_" + cls]) for cls in classes])
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert abs(probabilities.sum() - 1) < 0.00001
        assert classes[probabilities.argmax()] == row["predicted_class"]
        matrix[classes.index(row["true_class"]), classes.index(row["predicted_class"])] += 1
    assert np.array_equal(matrix, metrics["confusion_matrix"])
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(10), where=matrix.sum(0) > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(10), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall > 0)
    independent = {"accuracy": tp.sum()/matrix.sum(), "macro_precision": precision.mean(),
                   "macro_recall": recall.mean(), "macro_f1": f1.mean(),
                   "weighted_f1": (f1 * matrix.sum(1)).sum()/matrix.sum()}
    for key, value in independent.items():
        assert abs(value - metrics[key]) < 1e-10, (key, value, metrics[key])
    result = {"verified": True, "predictions": len(predictions), "disjoint_partitions": len(partitions),
              "duplicate_hashes_disjoint": True, "metrics_recalculated_from_counts": independent}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
