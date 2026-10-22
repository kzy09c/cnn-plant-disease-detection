"""Build the two result-viewing notebooks; execute_notebook.py records real outputs."""
import hashlib
import nbformat as nbf
from io_utils import ROOT


def build(dataset, folder, filename, notebook):
    title = "PlantVillage" if dataset == "plantvillage" else "Bangladesh"
    book = nbf.v4.new_notebook()
    book.metadata.kernelspec = {"name":"plant-tomato-cnn","display_name":"Tomato CNN (Python 3.12)","language":"python"}
    cells = [nbf.v4.new_markdown_cell(f"# {title}: binary tomato leaf health CNN\n\nThis model predicts **Healthy** or **Unhealthy**, without identifying a disease name. The cells review actual saved evidence on CPU. They do not retrain the model or run more test inference.\n\nSee README.md for setup, RESULTS.md for the full report, and figures/README.md for reusable PNG/SVG captions."),
             nbf.v4.new_code_cell(f'from pathlib import Path\nimport json, sys\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nif not (ROOT / "model.py").exists():\n    ROOT = ROOT / "cnn-plant-disease-detection"\nsys.path.insert(0, str(ROOT))\nOUT = ROOT / "{folder}"\nmetrics = json.loads((OUT / "test_metrics.json").read_text())\naudit = json.loads((OUT / "data_audit.json").read_text())\ncv = json.loads((OUT / "cross_validation.json").read_text())\nprint("Dataset:", "{title}")\nprint("Development:", audit["development_images"], "Primary test:", metrics["test_images"])\nprint(audit["binary_counts"])'),
             nbf.v4.new_markdown_cell("## Architecture and experimental design\n\nBoth models have the same 939,874-parameter, four-block CNN and two-output head. RGB inputs are resized to 128×128 and normalized to [-1,1]. Validation uses five groups of development images; the final fit uses twenty epochs and fixed argmax decisions.\n\nPlantVillage estimates a fixed scratch recipe. Bangladesh compares scratch and PlantVillage feature adaptation using development-only balanced accuracy. Test scores do not select recipes or thresholds."),
             nbf.v4.new_code_cell(f'import torch\nfrom model import TomatoCNN\ncheckpoint = torch.load(OUT / "{filename}", map_location="cpu", weights_only=True)\nmodel = TomatoCNN(checkpoint["channels"], 2)\nmodel.load_state_dict(checkpoint["state_dict"])\nmodel.eval()\nprint(model)\nprint("Parameters:", sum(p.numel() for p in model.parameters()))\nprint("Classes:", checkpoint["classes"])\ndisplay(Image(filename=str(ROOT / "figures/cnn_architecture.png"), width=1000))\ndisplay(Image(filename=str(ROOT / "figures/experiment_workflow.png"), width=1000))'),
             nbf.v4.new_code_cell(f'for mode, scores in cv.items():\n    print(mode, "CV balanced accuracy:", f"{{scores[\"mean_balanced_accuracy\"]:.2%}}", "CV macro F1:", f"{{scores[\"mean_macro_f1\"]:.2%}}")\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_cv_learning_curves.png"), width=1000))\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_final_training_curves.png"), width=1000))'),
             nbf.v4.new_markdown_cell("## Primary test\n\nAccuracy is shown alongside balanced accuracy and macro F1. Healthy recall is specificity with Unhealthy as the positive class. A high own-dataset score does not establish performance on new gardens. Bangladesh's primary cohort contains only predefined eligible images whose supplied labels agree."),
             nbf.v4.new_code_cell(f'for key in ("accuracy", "balanced_accuracy", "macro_f1", "specificity_healthy", "recall_sensitivity_unhealthy", "precision_unhealthy", "f1_unhealthy"):\n    print(key, f"{{metrics[key]:.2%}}")\nprint("Reference rows / prediction columns:", metrics["confusion_matrix"])\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_confusion_counts.png"), width=650))\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_confusion_normalized.png"), width=650))\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_roc_precision_recall.png"), width=1000))'),
             nbf.v4.new_code_cell(f'display(Image(filename=str(ROOT / "figures/{dataset}_class_metrics.png"), width=900))\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_confidence_intervals.png"), width=900))\ndisplay(Image(filename=str(ROOT / "figures/{dataset}_test_examples.jpg"), width=900))\nerror_gallery = ROOT / "figures/{dataset}_errors.jpg"\nif error_gallery.exists():\n    display(Image(filename=str(error_gallery), width=900))')]
    if dataset == "bangladesh":
        cells += [nbf.v4.new_markdown_cell("## Full released holdout: uncertain labels\n\nThe full 160-image publisher holdout contains 21 filename labels that conflict with their annotations. These descriptive scores use the uncertain filename references. The model and threshold are unchanged; these results did not select the recipe."),
                  nbf.v4.new_code_cell('raw = json.loads((OUT / "released_holdout_metrics.json").read_text())\nfor key in ("accuracy", "balanced_accuracy", "macro_f1", "specificity_healthy", "recall_sensitivity_unhealthy"):\n    print(key, f"{raw[key]:.2%}")\ndisplay(Image(filename=str(ROOT / "figures/bangladesh_full_holdout_confusion_counts.png"), width=650))\ndisplay(Image(filename=str(ROOT / "figures/bangladesh_label_audit.png"), width=1000))')]
    cells += [nbf.v4.new_markdown_cell(f'## Try a photograph\n\n```powershell\npython predict.py "C:\\path\\to\\leaf.jpg" --dataset {dataset}\n```\n\nSee CNN_GUIDE.md for the metric definitions and DATA_SOURCES.md for source/label limitations.'),
              nbf.v4.new_code_cell('print(json.loads((OUT / "verification.json").read_text()))\nreport = (OUT / "REPORT.md").read_text(encoding="utf-8").replace("](../", "](")\ndisplay(Markdown(report))')]
    for cell in cells:cell.id = hashlib.sha1(cell.source.encode()).hexdigest()[:12]
    book.cells = cells
    nbf.write(book,ROOT/notebook)


if __name__ == "__main__":
    build("plantvillage","outputs","tomato_cnn.pt","Tomato_CNN.ipynb")
    build("bangladesh","bangladesh_outputs","tomato_cnn_bangladesh.pt","Bangladesh_CNN.ipynb")
