"""Compare two scratch-trained CNNs by grouped five-fold CV; test only once."""
import argparse
import csv
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, log_loss)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CONFIGS = {"compact": (16, 32, 64), "deeper": (32, 64, 128, 192)}


class TomatoCNN(nn.Module):
    def __init__(self, channels, classes=10):
        super().__init__()
        blocks = []
        previous = 3
        for channel in channels:
            blocks.extend([nn.Conv2d(previous, channel, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
                           nn.Conv2d(channel, channel, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            previous = channel
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                                        nn.Linear(previous * 4, 128), nn.ReLU(),
                                        nn.Dropout(0.35), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_data(size):
    paths = sorted((ROOT / "data/raw/color").glob("Tomato___*/*"))
    classes = sorted({p.parent.name for p in paths})
    assert len(classes) == 10 and len(paths) == 18160
    leaf_map = json.loads((ROOT / "data/leaf-map.json").read_text())
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    def read(path):
        contents = path.read_bytes()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            resized = np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR))
        identifier = path.stem.split("___")[-1].split("copy")[0].strip().lower()
        candidates = [c for c in leaf_map.get(identifier, []) if c.startswith(path.parent.name + ":::")]
        group = candidates[0] if len(candidates) == 1 else path.parent.name + ":::fallback:" + identifier
        return resized, group, hashlib.sha256(contents).hexdigest(), pixel_hash, bool(candidates)

    print("Loading images and auditing same-leaf / duplicate groups", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        items = list(pool.map(read, paths))
    seen = {}
    for i, (_, group, byte_hash, pixel_hash, _) in enumerate(items):
        for key in [("leaf", group), ("bytes", byte_hash), ("pixels", pixel_hash)]:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = np.array([f"group_{find(i)}" for i in range(len(paths))])
    labels = np.array([classes.index(p.parent.name) for p in paths])
    for group in np.unique(groups):
        if len(set(labels[groups == group])) != 1:
            raise ValueError("Duplicate image or leaf group has conflicting labels")
    images = np.stack([item[0] for item in items]).transpose(0, 3, 1, 2).copy()
    return paths, classes, images, labels, groups, items


def prepare_batch(images, indices, augment):
    x = images[indices].float().div_(255)
    if augment:
        # Apply geometry and photometric augmentation only after splitting.
        for dimension in (2, 3):
            mask = torch.rand(len(x), device=x.device) < 0.5
            x[mask] = x[mask].flip(dimension)
        mask = torch.rand(len(x), device=x.device) < 0.5
        x[mask] = x[mask].transpose(2, 3)
        brightness = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.85, 1.15)
        colour = torch.empty((len(x), 3, 1, 1), device=x.device).uniform_(0.95, 1.05)
        x = (x * brightness * colour).clamp_(0, 1)
    return x.sub_(0.5).div_(0.5).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, images, indices, batch_size):
    model.eval()
    parts = []
    for start in range(0, len(indices), batch_size):
        x = prepare_batch(images, indices[start:start + batch_size], False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities = model(x).float().softmax(1)
        parts.append(probabilities.cpu().numpy())
    return np.concatenate(parts)


def fit(config, images, labels_gpu, train_ids, validation_ids, labels, epochs, batch_size, seed, run_name):
    seed_everything(seed)
    model = TomatoCNN(CONFIGS[config]).cuda().to(memory_format=torch.channels_last)
    counts = np.bincount(labels[train_ids], minlength=10)
    weights = torch.tensor(len(train_ids) / (10 * counts), dtype=torch.float32, device="cuda")
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    scaler = torch.amp.GradScaler("cuda")
    ids = torch.as_tensor(train_ids, device="cuda")
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        shuffled = ids[torch.randperm(len(ids), device="cuda")]
        loss_sum = 0.0
        for start in range(0, len(shuffled), batch_size):
            batch_ids = shuffled[start:start + batch_size]
            x = prepare_batch(images, batch_ids, True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_function(model(x), labels_gpu[batch_ids])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * len(batch_ids)
        schedule.step()
        result = {"epoch": epoch, "train_loss": float(loss_sum / len(ids))}
        if validation_ids is not None:
            probabilities = predict(model, images, validation_ids, batch_size)
            predicted = probabilities.argmax(1)
            result.update(accuracy=float(accuracy_score(labels[validation_ids], predicted)),
                          macro_f1=float(f1_score(labels[validation_ids], predicted, average="macro")))
            suffix = f" val accuracy={result['accuracy']:.4f} macro F1={result['macro_f1']:.4f}"
        else:
            suffix = ""
        result["seconds"] = time.time() - started
        history.append(result)
        write_json(OUT / f"{run_name}_history.json", history)
        print(f"{run_name} epoch {epoch}/{epochs}: loss={result['train_loss']:.4f}{suffix}, {result['seconds']:.1f}s", flush=True)
    return model, history


def evaluate(y, probabilities, classes):
    predicted = probabilities.argmax(1)
    healthy = classes.index("Tomato___healthy")
    unhealthy_y = (y != healthy).astype(int)
    unhealthy_pred = (predicted != healthy).astype(int)
    unhealthy_probability = 1 - probabilities[:, healthy]
    binary = {"accuracy": float(accuracy_score(unhealthy_y, unhealthy_pred)),
              "precision": float(precision_score(unhealthy_y, unhealthy_pred, zero_division=0)),
              "recall_sensitivity": float(recall_score(unhealthy_y, unhealthy_pred)),
              "specificity": float(recall_score(1 - unhealthy_y, 1 - unhealthy_pred)),
              "f1": float(f1_score(unhealthy_y, unhealthy_pred)),
              "roc_auc": float(roc_auc_score(unhealthy_y, unhealthy_probability)),
              "average_precision": float(average_precision_score(unhealthy_y, unhealthy_probability)),
              "confusion_matrix": confusion_matrix(unhealthy_y, unhealthy_pred).tolist(),
              "decision_rule": "Predicted multiclass label is anything other than healthy"}
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_precision": float(precision_score(y, predicted, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y, predicted, average="macro")),
            "macro_f1": float(f1_score(y, predicted, average="macro")),
            "weighted_f1": float(f1_score(y, predicted, average="weighted")),
            "macro_roc_auc_ovr": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
            "log_loss": float(log_loss(y, probabilities)),
            "per_class": classification_report(y, predicted, target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted).tolist(),
            "healthy_vs_unhealthy": binary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if (OUT / "test_metrics.json").exists():
        raise RuntimeError("This experiment has already used its test set. Use a separate output folder for a new experiment.")
    assert torch.cuda.is_available(), "GPU required for this training configuration"
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    started = time.time()
    paths, classes, images_np, labels, groups, items = load_data(args.image_size)
    # Deterministic approximate-stratified 80/20 split, respecting leaf groups.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    development, test = next(outer.split(np.zeros(len(labels)), labels, groups))
    assert not set(groups[development]) & set(groups[test])
    folds = []
    fold_assignment = np.full(len(labels), -1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 1)
    for fold, (train_local, val_local) in enumerate(cv.split(np.zeros(len(development)), labels[development], groups[development])):
        train_ids, val_ids = development[train_local], development[val_local]
        assert not set(groups[train_ids]) & set(groups[val_ids])
        assert set(labels[val_ids]) == set(range(10))
        folds.append((train_ids, val_ids))
        fold_assignment[val_ids] = fold
    assert set(labels[test]) == set(range(10))
    with (OUT / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "class", "group", "split", "cv_validation_fold", "sha256", "pixel_sha256", "leaf_mapping_found"])
        for i, path in enumerate(paths):
            writer.writerow([path.relative_to(ROOT).as_posix(), classes[labels[i]], groups[i],
                             "test" if fold_assignment[i] < 0 else "development", int(fold_assignment[i]), items[i][2], items[i][3], items[i][4]])
    info = {"dataset": "Original PlantVillage tomato RGB images", "classes": classes,
            "images": len(labels), "development_images": len(development), "test_images": len(test),
            "unique_groups": len(np.unique(groups)), "mapped_images": sum(item[4] for item in items),
            "unique_byte_hashes": len({item[2] for item in items}), "unique_pixel_hashes": len({item[3] for item in items}),
            "class_counts": {c: {"total": int(sum(labels == i)), "development": int(sum(labels[development] == i)), "test": int(sum(labels[test] == i))} for i, c in enumerate(classes)},
            "configurations": {name: list(channels) for name, channels in CONFIGS.items()},
            "arguments": vars(args), "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
            "protocol": "Grouped 80/20 holdout; two configurations with grouped five-fold CV on development only; select by mean final-epoch macro F1; retrain from scratch on all development images for same epoch budget; one test evaluation",
            "numerical_reproducibility": "Seeds fixed. CUDA/cuDNN benchmark may cause small differences across reruns."}
    write_json(OUT / "experiment.json", info)
    print(json.dumps(info, indent=2), flush=True)
    images = torch.from_numpy(images_np).cuda()
    del images_np, items
    labels_gpu = torch.as_tensor(labels, device="cuda")
    all_cv = {}
    for config in CONFIGS:
        results = []
        for fold, (train_ids, val_ids) in enumerate(folds, 1):
            run_name = f"{config}_fold{fold}"
            history_path = OUT / f"{run_name}_history.json"
            if history_path.exists():
                previous = json.loads(history_path.read_text())
                if len(previous) == args.epochs:
                    print(f"Reusing completed CV history: {run_name}", flush=True)
                    results.append(previous[-1])
                    continue
            model, history = fit(config, images, labels_gpu, train_ids, val_ids, labels,
                                 args.epochs, args.batch_size, args.seed + fold, run_name)
            results.append(history[-1])
            del model
        all_cv[config] = {"folds": results,
                          "mean_macro_f1": float(np.mean([x["macro_f1"] for x in results])),
                          "sd_macro_f1": float(np.std([x["macro_f1"] for x in results], ddof=1)),
                          "mean_accuracy": float(np.mean([x["accuracy"] for x in results])),
                          "sd_accuracy": float(np.std([x["accuracy"] for x in results], ddof=1))}
        write_json(OUT / "cross_validation.json", all_cv)
    selected = max(all_cv, key=lambda c: all_cv[c]["mean_macro_f1"])
    print(f"Selected {selected} using CV macro F1; retraining on all development images", flush=True)
    model, _ = fit(selected, images, labels_gpu, development, None, labels,
                   args.epochs, args.batch_size, args.seed + 100, "final")
    torch.save({"state_dict": model.state_dict(), "channels": CONFIGS[selected], "classes": classes,
                "image_size": args.image_size, "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
                "selected_configuration": selected}, OUT / "tomato_cnn.pt")
    # The sole access to test images for model inference in this experiment.
    probabilities = predict(model, images, test, args.batch_size)
    metrics = evaluate(labels[test], probabilities, classes)
    metrics.update(selected_configuration=selected, test_images=len(test), test_groups=len(np.unique(groups[test])),
                   elapsed_seconds=time.time() - started,
                   completed_training_validation_seconds=sum(sum(epoch["seconds"] for epoch in json.loads(p.read_text())) for p in OUT.glob("*_history.json")),
                   parameter_count=sum(p.numel() for p in model.parameters()))
    np.savez_compressed(OUT / "test_predictions.npz", indices=test, true_labels=labels[test], probabilities=probabilities)
    with (OUT / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "group", "true_class", "predicted_class"] + ["probability_" + c for c in classes])
        for i, probs in zip(test, probabilities):
            writer.writerow([paths[i].relative_to(ROOT).as_posix(), groups[i], classes[labels[i]], classes[probs.argmax()]] + probs.tolist())
    write_json(OUT / "test_metrics.json", metrics)
    print("FINAL HELD-OUT TEST METRICS", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
