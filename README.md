# Tomato leaf health: PlantVillage + Bangladesh CNNs

<p align="center"><img src="images/spinn.gif" alt="Project GIF" width="400"></p>

Two trained binary models classify tomato leaf images as **Healthy** or **Unhealthy**. Both use the same custom PyTorch CNN with 939,874 parameters and 128×128 RGB input. Unhealthy includes the source disease, pest-damage, and stress categories; no specific disease name is predicted.

The original PlantVillage workflow is now binary. The separate Bangladesh experiment uses natural-background field photos and balanced training batches. The trained checkpoints are included in this repository.

## Actual results

| Dataset / test cohort | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |
|---|---:|---:|---:|---:|---:|---:|
| PlantVillage binary | 3,632 | **99.86%** | **99.92%** | **99.57%** | 100.00% (318/318) | 99.85% (3,309/3,314) |
| Bangladesh: primary label-agreement cohort | 139 | **94.96%** | **94.07%** | **94.75%** | 88.14% (52/59) | 100.00% (80/80) |
| Bangladesh: full holdout, uncertain filename labels | 160 | **83.75%** | **83.75%** | **83.31%** | 67.50% (54/80) | 100.00% (80/80) |

These are different image populations, not a controlled ranking of field performance. PlantVillage uses controlled leaf images and a previously inspected benchmark test. Bangladesh's full holdout includes 21 filename labels that contradict the annotations; the 139-image primary cohort was defined by label agreement before training. Agreement does not establish expert diagnosis. Performance in new gardens remains untested.

**Report material:** [full results](RESULTS.md), [CSV summary](RESULTS.csv), [figure captions and SVG/PNG index](figures/README.md), and [CNN explanation](CNN_GUIDE.md). Each output folder also includes per-image predictions, class metrics, uncertainty intervals, CV summaries, ROC points, and precision–recall points as CSV files.

![Shared CNN architecture](figures/cnn_architecture.png)

![Experiment workflow](figures/experiment_workflow.png)

![Dataset counts](figures/dataset_counts.png)

## Setup in the VS Code terminal (Windows)

Use Python 3.12, the version recorded for these experiments. Install the VS Code Python and Jupyter extensions.

```powershell
git clone https://github.com/kzy09c/cnn-plant-disease-detection.git
cd cnn-plant-disease-detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r notebook-requirements.txt
python -m ipykernel install --user --name plant-tomato-cnn --display-name "Tomato CNN (Python 3.12)"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`. Each computer creates its own virtual environment; it is not part of the repo. CPU supports prediction, notebooks, verification, and regenerating figures. Fresh training needs a compatible CUDA-enabled PyTorch installation and NVIDIA GPU; the recorded training environment is in `environment-lock.txt`.

## Predict an image

PlantVillage is the default binary model:

```powershell
python predict.py "C:\path\to\leaf.jpg"
```

Use the Bangladesh binary model:

```powershell
python predict.py "C:\path\to\leaf.jpg" --dataset bangladesh
```

An explicit checkpoint also works:

```powershell
python predict.py "C:\path\to\leaf.jpg" --model bangladesh_outputs/tomato_cnn_bangladesh.pt
```

The JSON output includes the selected model, Healthy/Unhealthy prediction, and both softmax scores. Softmax values are not guaranteed calibrated confidence levels. The classifier expects a tomato leaf image; it has no detector or abstention rule. Example JPGs in `manual_test/` have provenance in the result-folder sample_images.json files; fixed examples are not guaranteed correct predictions. Existing internet-image filenames are not verified ground-truth labels.

## Explore the completed experiments

Select the project virtual environment as the VS Code notebook kernel:

- [Tomato_CNN.ipynb](Tomato_CNN.ipynb): PlantVillage binary.
- [Bangladesh_CNN.ipynb](Bangladesh_CNN.ipynb): Bangladesh binary, with primary/full-holdout results separated.

The notebooks show actual saved outputs without retraining or another test inference pass.

```powershell
python verify_results.py
python make_report.py
python execute_notebook.py Tomato_CNN.ipynb
python execute_notebook.py Bangladesh_CNN.ipynb
python -m unittest discover -s tests -v
```

`make_report.py` generates PNG and editable SVG figures plus CSV tables from the tracked results. Source images are needed only to regenerate error-photo galleries; the existing galleries remain available without a dataset download.

![PlantVillage confusion matrix](figures/plantvillage_confusion_counts.png)

![Bangladesh primary confusion matrix](figures/bangladesh_confusion_counts.png)

## Experiment files

| Experiment | Checkpoint | Results | Notebook |
|---|---|---|---|
| PlantVillage binary | `outputs/tomato_cnn.pt` | [outputs/REPORT.md](outputs/REPORT.md) | `Tomato_CNN.ipynb` |
| Bangladesh binary | `bangladesh_outputs/tomato_cnn_bangladesh.pt` | [bangladesh_outputs/REPORT.md](bangladesh_outputs/REPORT.md) | `Bangladesh_CNN.ipynb` |

The checked-in model bytes and saved test predictions are imported unchanged from the completed binary training experiments. `experiment.json` records the original experiment/manifest hashes and import provenance. The refactored standalone training code retains the recipes, but reruns are not guaranteed bit-identical because GPU kernels and random-number consumption can vary. No scores were regenerated by retraining during the repository cleanup.

## Fresh training

Use a new output directory; completed results are protected from overwrite. The recorded PlantVillage manifest remains the reference for its original image groups, test partition, and five folds.

```powershell
python download_data.py
python train.py --dataset plantvillage --output runs/plantvillage
python verify_results.py --dataset plantvillage --output runs/plantvillage

python download_bangladesh.py
python ensure_plantvillage_fingerprints.py
python train.py --dataset bangladesh --output runs/bangladesh
python verify_results.py --dataset bangladesh --output runs/bangladesh
```

Defaults are 10 epochs per grouped CV fit, 20 final-fit epochs, batch size 64, and 128×128 input. PlantVillage uses a fixed scratch recipe with class-weighted cross entropy. Bangladesh compares scratch with PlantVillage feature adaptation, selecting by mean final-epoch validation balanced accuracy, and uses equal Healthy/Unhealthy draws in every batch without additional class weights. Bangladesh's original released val folder stays reserved as test. Neither test cohort selects the recipe or threshold.

Downloaded data, local runs, and virtual environments are ignored by Git. Interrupted new runs are preserved; use a different `--output` directory for a clean rerun.

## Dataset provenance

See [DATA_SOURCES.md](DATA_SOURCES.md) for source links, original counts, annotation conflicts, duplicate/pretraining checks, and limitations. See [BANGLADESH_LICENSE.txt](BANGLADESH_LICENSE.txt) for Bangladesh CC BY 4.0 attribution and changes.

<p align="center"><img src="images/anime-bocchi.gif" alt="Project GIF" width="400"></p>
