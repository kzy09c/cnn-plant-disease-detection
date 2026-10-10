# PlantVillage tomato CNN

This project adapts the supplied Tutorial 5 CNN workflow to classifying tomato leaf photographs into ten PlantVillage classes: healthy, eight disease categories, and spider-mite damage.

Two custom CNNs are trained from scratch using grouped five-fold cross-validation. The configuration with the higher mean macro F1 is retrained on all development images and evaluated once on a held-out test set. All images of known same-leaf groups and exact duplicates stay in one partition. Leaf metadata is incomplete, so the report documents the grouping coverage and remaining uncertainty.

The measured results are in **outputs/REPORT.md**. The trained checkpoint is **outputs/tomato_cnn.pt**. No accuracy or F1 values are assumed or taken from published models.

`Tomato_CNN.ipynb` explains the architecture and protocol and displays the saved metrics and charts. Its fresh-training cell is disabled by default to preserve the completed test evaluation.

To execute the explanatory notebook without retraining, install `notebook-requirements.txt` and run `execute_notebook.py` using the same Python environment. The delivered notebook includes actual executed outputs.

## Run locally

The isolated environment already created for this experiment is `.venv`. From the repository root in PowerShell:

```powershell
.\plant_tomato\.venv\Scripts\python.exe plant_tomato\predict.py 'C:\path\to\tomato-leaf.jpg'
.\plant_tomato\.venv\Scripts\python.exe plant_tomato\verify_results.py
.\plant_tomato\.venv\Scripts\python.exe plant_tomato\make_report.py
```

The prediction utility runs on CPU; training uses CUDA. Input images are resized to 96 x 96 pixels. The model is for single tomato leaf images and has not been evaluated on field photos, whole-plant images, or tomato fruit.

## Fresh training elsewhere

Use Python 3.12 and an NVIDIA GPU. Create a virtual environment and install the appropriate GPU PyTorch build, then the other dependencies. The exact installed package versions for this experiment are in `environment-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install scikit-learn matplotlib pillow numpy
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py --epochs 20 --batch-size 128 --image-size 96
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe make_report.py
```

`download_data.py` downloads only the original RGB tomato images from the author's archive, using HTTP byte ranges. It excludes grayscale and segmented copies. It also retrieves the author's leaf map. Downloaded data and the virtual environment are excluded from the Git working tree and the deliverable ZIP.

Training refuses to rerun in a directory where `outputs/test_metrics.json` exists. Use a separate project/output directory for another experiment; repeated tuning against this completed test set would undermine its independence.

## Evidence

- `outputs/experiment.json`: dataset counts, class definitions, training settings, GPU and protocol.
- `outputs/split_manifest.csv`: image paths, group assignments, split/fold membership and hashes.
- `outputs/cross_validation.json`: model-selection results.
- `outputs/*_history.json`: each training epoch's loss and validation metrics.
- `outputs/test_predictions.csv`: every test prediction and class probability.
- `outputs/test_metrics.json`: overall, per-class, and healthy/unhealthy metrics.
- `outputs/confidence_intervals.json`: group bootstrap uncertainty estimates.
- `outputs/verification.json`: independent audit of split boundaries and metrics calculated from confusion counts.
- `outputs/confusion_matrix.png` and `outputs/cross_validation_curves.png`: evaluation charts.

Dataset: [original author's repository](https://github.com/spMohanty/PlantVillage-Dataset), [author's download](https://huggingface.co/datasets/mohanty/PlantVillage). Method reference: [Mohanty et al., 2016](https://doi.org/10.3389/fpls.2016.01419).
