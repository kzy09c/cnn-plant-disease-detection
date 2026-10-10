"""Generate a report and charts from saved predictions without rerunning test inference."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def pretty(label):
    return label.replace("Tomato___", "").replace("Tomato_", "").replace("_", " ").capitalize()


def bootstrap_intervals(prediction_rows):
    classes = sorted({row["true_class"] for row in prediction_rows})
    groups = sorted({row["group"] for row in prediction_rows})
    group_ids = {group: i for i, group in enumerate(groups)}
    matrices = np.zeros((len(groups), 10, 10), dtype=int)
    for row in prediction_rows:
        matrices[group_ids[row["group"]], classes.index(row["true_class"]), classes.index(row["predicted_class"])] += 1
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(2000):
        matrix = matrices[rng.integers(0, len(groups), len(groups))].sum(0)
        tp = np.diag(matrix)
        recall = np.divide(tp, matrix.sum(1), out=np.zeros(10), where=matrix.sum(1) > 0)
        precision = np.divide(tp, matrix.sum(0), out=np.zeros(10), where=matrix.sum(0) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) > 0)
        values.append([tp.sum() / matrix.sum(), precision.mean(), recall.mean(), f1.mean(), (f1 * matrix.sum(1)).sum() / matrix.sum()])
    names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    return {name: np.quantile(np.array(values)[:, i], [0.025, 0.975]).tolist() for i, name in enumerate(names)}


def main():
    info = json.loads((OUT / "experiment.json").read_text())
    cv = json.loads((OUT / "cross_validation.json").read_text())
    metrics = json.loads((OUT / "test_metrics.json").read_text())
    rows = list(csv.DictReader((OUT / "test_predictions.csv").open(encoding="utf-8")))
    intervals = bootstrap_intervals(rows)
    (OUT / "confidence_intervals.json").write_text(json.dumps({"method": "2000 percentile bootstrap resamples of whole test leaf/duplicate groups", "intervals_95_percent": intervals}, indent=2))
    classes = info["classes"]
    display_classes = [pretty(c) for c in classes]
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(12, 10))
    fractions = matrix / matrix.sum(1, keepdims=True)
    plot = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if fractions[i, j] > 0.5 else "black")
    ax.set_xticks(range(10), display_classes, rotation=50, ha="right")
    ax.set_yticks(range(10), display_classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Held-out tomato test set: {metrics['test_images']:,} images\nCells show counts; shading shows fraction of each true class")
    fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.02, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for config in cv:
        histories = [json.loads((OUT / f"{config}_fold{fold}_history.json").read_text()) for fold in range(1, 6)]
        epochs = np.arange(1, len(histories[0]) + 1)
        for ax, key, title in zip(axes, ["accuracy", "macro_f1"], ["Validation accuracy", "Validation macro F1"]):
            values = np.array([[epoch[key] for epoch in history] for history in histories])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            line = ax.plot(epochs, mean, label=config.capitalize())[0]
            ax.fill_between(epochs, mean - sd, mean + sd, color=line.get_color(), alpha=0.15)
            ax.set_title(title + " across five folds")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cross_validation_curves.png", dpi=160)
    plt.close(fig)
    percent = lambda value: f"{value * 100:.2f}%"
    lines = ["# Tomato leaf disease CNN - measured results", "",
             f"Trained from scratch on {info['images']:,} original PlantVillage RGB tomato images. Selected **{metrics['selected_configuration']} CNN**, with **{metrics['parameter_count']:,} trainable parameters**, using five-fold cross-validation. The final model was trained on all {info['development_images']:,} development images and evaluated once on {info['test_images']:,} held-out images.", "",
             "## Held-out test results", "", "| Metric | Score | 95% group bootstrap interval |", "|---|---:|---:|"]
    for label, key in [("Accuracy", "accuracy"), ("Macro precision", "macro_precision"), ("Macro recall / balanced accuracy", "macro_recall"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")]:
        lower, upper = intervals[key]
        lines.append(f"| {label} | {percent(metrics[key])} | {percent(lower)} - {percent(upper)} |")
    lines += ["", f"Macro one-vs-rest ROC AUC: **{metrics['macro_roc_auc_ovr']:.4f}**. Multiclass log loss: **{metrics['log_loss']:.4f}**.", "",
              "Macro F1 gives each class equal importance. Weighted F1 gives each class importance proportional to its test support. Accuracy is the fraction of correctly classified images. Precision measures how often a predicted class is correct; recall measures how many real examples of that class are found. F1 is their harmonic mean. Balanced accuracy equals macro recall here.", "",
              "The intervals resample complete test leaf/duplicate groups 2,000 times. They describe test-sample uncertainty conditional on this trained model and this dataset. They do not include variation from retraining, uncertain leaf metadata, or deployment on field images.", "",
              "## Per-class test results", "", "| Class | Test images | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for cls in classes:
        result = metrics["per_class"][cls]
        lines.append(f"| {pretty(cls)} | {int(result['support'])} | {percent(result['precision'])} | {percent(result['recall'])} | {percent(result['f1-score'])} |")
    binary = metrics["healthy_vs_unhealthy"]
    lines += ["", "## Detecting unhealthy versus healthy leaves", "",
              "For this secondary task, disease and spider-mite classes are grouped as unhealthy. The positive prediction is any multiclass prediction other than healthy. This rule was set before seeing the test results; no threshold was tuned on test data.", "",
              "| Metric | Score |", "|---|---:|"]
    for label, key in [("Accuracy", "accuracy"), ("Unhealthy precision", "precision"), ("Unhealthy recall / sensitivity", "recall_sensitivity"), ("Healthy recall / specificity", "specificity"), ("Unhealthy F1", "f1"), ("ROC AUC", "roc_auc"), ("Average precision (PR)", "average_precision")]:
        lines.append(f"| {label} | {percent(binary[key]) if key not in ('roc_auc', 'average_precision') else f'{binary[key]:.4f}'} |")
    tn, fp = binary["confusion_matrix"][0]
    fn, tp = binary["confusion_matrix"][1]
    lines += ["", f"Counts: {tn} healthy leaves correctly called healthy, {fp} healthy leaves incorrectly called unhealthy, {fn} unhealthy leaves incorrectly called healthy, and {tp} unhealthy leaves correctly called unhealthy.", "",
              "## Five-fold model comparison", "",
              "| Configuration | Convolution channels | CV accuracy (mean +/- sample SD) | CV macro F1 (mean +/- sample SD) |", "|---|---|---:|---:|"]
    for config, result in cv.items():
        lines.append(f"| {config.capitalize()} | {info['configurations'][config]} | {percent(result['mean_accuracy'])} +/- {percent(result['sd_accuracy'])} | {percent(result['mean_macro_f1'])} +/- {percent(result['sd_macro_f1'])} |")
    lines += ["", "Each model was reinitialized for every fold. Both configurations used the same folds and the same fixed 20-epoch budget. Selection used mean macro F1 at epoch 20. These CV estimates were used for model selection; the independent held-out test is the final evaluation.", "",
              "## Training and evaluation protocol", "",
              "1. Use original RGB tomato images only; exclude grayscale, segmented, and pre-augmented copies.",
              "2. Group known same-leaf images using the author's leaf map. Group unmatched images by normalized original filename identifiers. Merge exact file duplicates and exact decoded-pixel duplicates into the same group.",
              "3. Create an approximate-stratified 80/20 split with `StratifiedGroupKFold` (seed 42). Create five further grouped folds within development data (seed 43). Verify there are no group overlaps and that every validation/test partition contains all ten classes.",
              "4. Resize to 96 x 96 RGB, scale to [0,1], normalize using fixed mean and standard deviation of 0.5. Apply random flips, transpose, brightness and mild colour shifts to training batches only.",
              "5. Use two 3x3 convolutions, batch normalization and ReLU per block, then max pooling. Use adaptive 2x2 average pooling, a 128-unit dense layer, ReLU, dropout 0.35, and ten logits. The compact CNN has three blocks and the deeper CNN has four.",
              "6. Train with AdamW (initial learning rate 0.001, weight decay 0.0001), cosine decay to 0.00001, batch size 128, 20 epochs and CUDA mixed precision. Calculate inverse-frequency cross-entropy weights from each fold's training labels only.",
              "7. Select the configuration by mean CV macro F1. Reinitialize and retrain on all development images for 20 epochs. Run inference on the held-out test once. Compute all reported metrics and plots from those saved predictions.", "",
              f"GPU: {info['gpu']}. PyTorch: {info['torch_version']}. Recorded completed training/validation time: {metrics['completed_training_validation_seconds'] / 60:.1f} minutes (excluding downloads, package installation, image auditing and interrupted partial runs). Seeds are fixed; CUDA kernels may introduce small numerical differences across reruns. A terminal session reset interrupted deeper fold 4; its incomplete history was replaced by a fresh run using the same seed and settings. The eight completed folds were retained. No test inference had occurred before the interruption.", "",
              "## Dataset and limitations", "",
              f"The dataset contains {info['unique_groups']:,} identified leaf/duplicate groups. Leaf-map metadata matched {info['mapped_images']:,} / {info['images']:,} images ({percent(info['mapped_images']/info['images'])}). There are {info['images']-info['unique_byte_hashes']} repeated exact images. Known groups and duplicates do not cross the test/development or CV boundaries. For unmatched images, filename grouping cannot guarantee that every photograph of the same physical leaf has been identified.", "",
              "The labels are healthy, bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow leaf curl virus, and mosaic virus. Spider mites represent pest damage. This is classification of a single leaf photograph, not localization of diseased leaves in a whole-plant scene or a diagnosis of tomato fruit disease.", "",
              f"The largest class is yellow leaf curl virus. A development-majority classifier would obtain {percent(info['class_counts']['Tomato___Tomato_Yellow_Leaf_Curl_Virus']['test']/info['test_images'])} test accuracy by always predicting that class. An always-unhealthy binary classifier would obtain {percent(1-info['class_counts']['Tomato___healthy']['test']/info['test_images'])} accuracy; this is why healthy specificity and macro F1 are useful alongside accuracy.", "",
              "PlantVillage images were collected under controlled conditions. These measured scores establish performance on held-out PlantVillage tomato images. They do not establish accuracy on new farms, backgrounds, cameras, weather conditions, mixed infections, or unseen diseases. Field use needs an independent field-photo evaluation. Softmax probabilities have not been calibrated.", "",
              "## Files and reproduction", "",
              "- `tomato_cnn.pt`: trained model weights and preprocessing/class metadata.",
              "- `test_metrics.json`: complete measured test results.",
              "- `test_predictions.csv` and `test_predictions.npz`: saved predictions, allowing metric verification without another test inference pass.",
              "- `split_manifest.csv`: class, split, fold, grouping and SHA-256 audit for every image.",
              "- `cross_validation.json` and `*_history.json`: CV summaries and epoch histories.",
              "- `confusion_matrix.png` and `cross_validation_curves.png`: evaluation plots.", "",
              "Run from the repository root in PowerShell:", "", "```powershell",
              ".\\plant_tomato\\.venv\\Scripts\\python.exe plant_tomato\\predict.py 'C:\\path\\to\\leaf.jpg'",
              ".\\plant_tomato\\.venv\\Scripts\\python.exe plant_tomato\\make_report.py", "```", "",
              "For a fresh experiment, create a separate copy/output directory before running `train.py`. It refuses to rerun when `test_metrics.json` exists, protecting the completed test evaluation. To install elsewhere, install GPU PyTorch for your CUDA driver, then `pip install -r requirements.txt`, run `download_data.py`, and run `train.py`.", "",
              "## Sources", "",
              "The user's Tutorial 5, Exercise 2 (page 4) motivated the two-configuration, five-fold CV and one-time held-out evaluation protocol. Its CIFAR-10 example was adapted to PlantVillage tomatoes. The supplied Canvas template notebook was not available, so this implementation was created independently.", "",
              "- [Author's PlantVillage dataset repository and leaf metadata](https://github.com/spMohanty/PlantVillage-Dataset)",
              "- [Author's PlantVillage download](https://huggingface.co/datasets/mohanty/PlantVillage)",
              "- [Mohanty, Hughes and Salathe (2016), Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419)",
              "- [scikit-learn grouped stratified cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved REPORT.md, confidence intervals and evaluation charts")


if __name__ == "__main__":
    main()
