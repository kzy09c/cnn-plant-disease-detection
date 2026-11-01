
# Tomato leaf disease detection with a custom CNN

A PyTorch CNN that classifies tomato foliage as **Healthy** or **Unhealthy**. It was trained from random weights across seven data sources. Later stages continue only this project's own checkpoints; no ResNet, ImageNet weights or other externally pretrained model is used.

The selected model is included as [`best_model.pt`](best_model.pt). It has 955,194 parameters and accepts 160 × 160 RGB images. Unhealthy includes source-labelled disease, insect damage, nutrient deficiency and other visible damage. It does not identify a specific disease.

**First time using the project?** Follow the [installation guide](#installation-guide-windows--vs-code) at the bottom of this README.

<p align="center"><img src="images/spinn.gif" alt="Project GIF" width="400"></p>

## Run a prediction

After installing the project, run this command from its folder with the virtual environment active:

```powershell
python predict.py "C:\path\to\tomato_leaf.jpg"
```

The command returns JSON with the predicted class and both model probabilities. It uses CUDA when available and otherwise runs on CPU. Choose another checkpoint with `--model path/to/checkpoint.pt`. Use a clear, close photograph of tomato foliage. Scores are uncalibrated; the classifier does not verify plant species, locate leaves or reliably reject unfamiliar inputs.

For a standalone copy, download [`tomato_cnn_inference.zip`](tomato_cnn_inference.zip), extract it, install `inference-requirements.txt`, and run `python predict.py your_image.jpg` from the extracted folder. Prediction requires no training data.

## Measured results

These held-out results use the delivered seven-source model, selected using validation data. Each source below also contributed separate training images. Balanced accuracy averages Healthy and Unhealthy recall, which matters because several tests contain very few Healthy images.

| Dataset | Test images | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| PlantVillage | 3,632 | 94.47% | 96.97% |
| Bangladesh | 139 | 87.77% | 86.04% |
| TLID | 3,050 | 92.59% | 94.48% |
| PlantDoc | 69 | 86.96% | 78.33% |
| Tomato-Village | 1,764 | 95.98% | 97.96% |
| PlantWild | 433 | 90.99% | 88.01% |
| Tomato-Disease annotation regions | 488 regions | 93.85% | 93.62% |

PlantWild is a curated internet-scene check with 26 apparently Healthy and 407 source-labelled Unhealthy test photos. Always predicting Unhealthy would achieve 94.00% accuracy but only 50% balanced accuracy. The model also agreed with 13 of 16 weak reference labels in an older web-photo convenience sample. Neither check establishes accuracy on arbitrary random internet images.

The original target of 90% balanced accuracy on each of the five initial benchmark sources was met on three: Bangladesh and PlantDoc remained below target. The later PlantWild check also remains below 90%. Older benchmarks were inspected in previous work, so their comparisons are exploratory. The annotated-region check assumes supplied bounding boxes.

See [`RESULTS.md`](RESULTS.md) for the original training experiments, class recalls, macro F1, confusion matrices and uncertainty. [`MODEL_CARD.md`](MODEL_CARD.md) describes intended use and limitations.

![Held-out balanced accuracy](figures/comparison.png)

## Pakistan dataset: external category-matched test

The unchanged model was also tested using the [Tomato Leaf Disease Classification Dataset in Pakistan, release 2](https://data.mendeley.com/datasets/3mbnb82mxd/2). Its six category names also occur in the training data. The model still predicts **Healthy or Unhealthy**; the table breaks down correct binary decisions by the publisher's disease category.

The archive contains 830 raw images and 7,200 augmented images. Only raw images were considered. Before prediction, 357 entries from photo families overlapping earlier data and 12 additional duplicate/cropped/resized copies were excluded. The remaining **461 images** had no detected matches in the recorded hash, structural and geometric checks. The whole release cannot be described as an independent unseen collection; undetected overlap and shared acquisition sessions can remain.

**408/461 correct (88.50% accuracy), 85.37% balanced accuracy, 84.87% macro F1.** The model was not retrained, and its threshold stayed at 0.5.

| Reference category | Correct / tested | Incorrect | Correct rate |
|---|---:|---:|---:|
| Healthy | 91 / 115 | 24 | 79.13% |
| Early blight | 68 / 76 | 8 | 89.47% |
| Late blight | 26 / 30 | 4 | 86.67% |
| Leaf mold | 146 / 146 | 0 | 100.00% |
| Septoria leaf spot | 60 / 61 | 1 | 98.36% |
| Yellow leaf curl virus | 17 / 33 | 16 | 51.52% |

Leaf mold and Septoria performed strongly in this cohort. Yellow leaf curl was the weakest check: 16 of its 33 images were incorrectly marked Healthy. The overall balanced accuracy remains below the 90% target, and this result does not establish internet-wide accuracy.

Read the [full report](evaluation_unseen_matching/REPORT.md), [individual predictions](evaluation_unseen_matching/predictions.csv), [all 830 audit decisions](evaluation_unseen_matching/audit_manifest.csv) and [dataset source/download guide](dataset_sources/pakistan/README.md). The source photographs are available from the original CC BY 4.0 release; this repository includes metadata, download tools and evaluation evidence. Saved metrics can be checked without downloading the images using `python verify_results.py`.

## Data and modelling pipeline

The preprocessing pipeline combines seven tomato-image sources into a binary task: **Healthy = 0** and **Unhealthy = 1**. Unhealthy includes source-labelled disease, insect damage and nutrient deficiency. Source labels are retained as reference labels; preprocessing does not establish a biological diagnosis.

### 1. Data loading

[`prepare_data.py`](prepare_data.py) reads existing audited manifests for PlantVillage, Bangladesh, whole-leaf TLID and tomato-only PlantDoc. Already excluded records are skipped, existing test images stay reserved, and development fold 1 becomes validation while the remaining development folds become training. The older 16-image internet sample is reserved for testing.

Source paths are configured through [`local_config.example.json`](local_config.example.json), with local overrides in `local_config.json`. Source links, licensing and label decisions are documented in [the experiment protocol](EXPERIMENT_PROTOCOL.md#sources).

### 2. Cleaning and preprocessing

The script verifies image-file SHA-256 hashes against the manifests, corrects EXIF orientation and converts images to RGB. It also records decoded-pixel hashes and perceptual fingerprints for duplicate checks.

#### Duplicate checks and split protection

Exact decoded-pixel duplicates, existing source groups and near duplicates are joined into groups. Near-duplicate candidates use perceptual-hash Hamming distance **≤ 4**, followed by block structural similarity **≥ 0.97**. If a group spans partitions, the entire group moves to the most reserved partition: **test > validation > train**. Groups containing identical decoded pixels with conflicting binary labels are excluded.

This reduces leakage between training and evaluation, but incomplete plant, garden and capture-session metadata means the checks cannot prove full independence. See [the grouping implementation](prepare_data.py) and [the protocol's limitations](EXPERIMENT_PROTOCOL.md#reporting-and-limits).

#### Source-specific preparation

| Source / stage | Preprocessing | Relevant code and documentation |
| --- | --- | --- |
| Tomato-Village | Initially audit the full source as an external test, check overlap with earlier data and exclude overlapping photo families. For the later training stage, group related filename families and duplicates, preserving publisher partitions with group promotion. | [External-source preparation](prepare_external.py), [field-data preparation](prepare_field.py), [experiment B protocol](EXPERIMENT_PROTOCOL.md#field-data-experiment-b) |
| Tomato-Disease annotation regions | Extract supplied bounding boxes with a fixed 10% context margin; skip boxes with an original edge below 32 pixels. Audit parent photos and crops for overlap and keep related regions together. The Healthy class ID is inferred from the authors' annotation counts and visually reviewed, rather than supplied by an official class-map file. | [Region preparation](prepare_zenodo.py), [experiment C protocol](EXPERIMENT_PROTOCOL.md#additional-annotated-source-continuation-c) |
| PlantWild | Use the tomato subset, apply the saved manual review of candidate Healthy images, group by source-page URLs and duplicate fingerprints, and exclude detected overlap with earlier photos and annotation parents. | [PlantWild preparation](prepare_plantwild.py), [experiment D protocol](EXPERIMENT_PROTOCOL.md#final-natural-scene-continuation-d) |

#### Image preparation and caching

Eligible whole images or annotation crops are resized to **160 × 160** using bilinear interpolation. Preparation stores channel-first **uint8** arrays with shape `N × 3 × 160 × 160` in memory-mapped `images.npy` caches. CSV manifests retain paths, labels, groups, split assignments and exclusion reasons; JSON audits record counts and manifest hashes.

Normalization happens when a batch enters the model: `x / 127.5 - 1` maps pixel values from 0–255 to −1–1. [Prediction preprocessing](predict.py) applies the same EXIF correction, RGB conversion, resize and normalization to a new input image.

### 3. Feature engineering

Feature extraction is learned jointly with the classifier in [`model.py`](model.py). The pipeline feeds RGB image tensors into four convolutional stages with **32, 64, 128 and 192 channels**. Each stage contains two 3 × 3 convolutions, BatchNorm and SiLU activations, followed by max pooling. These learned filters can represent visual patterns such as edges, colour variations and leaf textures; they are not manually assigned disease rules.

The selected model uses [`ChannelGate`](model.py) to learn channel weights, then adaptive 2 × 2 average pooling creates a fixed-size representation. With the default width, the resulting **768 values** feed a 128-unit dense layer and the two-class output. No handcrafted colour histogram, texture descriptor, segmentation mask or external pretrained embedding is supplied as a model feature.

The perceptual hashes used in preprocessing serve only to audit duplicate images and protect data splits; they are not classifier inputs.

### 4. Model training

[`train.py`](train.py) trains the custom CNN using AdamW, cross-entropy loss with label smoothing, a learning-rate warm-up and cosine schedule, mixed precision and gradient clipping. The initial experiments compare mild augmentation, strong augmentation with mixup, and strong augmentation with channel attention and mixup. The initial four-source model and the subsequent five-source model start from random weights. Later stages continue only this project's own checkpoints.

| Stage | Training code | Data and initialization |
| --- | --- | --- |
| A | [`train.py`](train.py) | Four sources; compare three recipes from random initialization. |
| B | [`train_field.py`](train_field.py) | Add Tomato-Village and retrain the selected recipe from random initialization. |
| C | [`train_extra.py`](train_extra.py) | Add Tomato-Disease regions while jointly replaying all six sources, starting from the selected B checkpoint. |
| D | [`train_wild.py`](train_wild.py) | Add PlantWild while jointly replaying all seven sources, starting from this project's previously selected checkpoint. |

Checkpoint selection uses validation data: `0.6 × mean source balanced accuracy + 0.4 × minimum source balanced accuracy`. The decision threshold stays at **0.5**; test predictions do not choose the training recipe, epoch or threshold. Run configurations, histories, checkpoint files and per-image training draw counts are recorded under [`runs/`](runs/). See [Architecture and training](#architecture-and-training), [the experiment protocol](EXPERIMENT_PROTOCOL.md) and [saved results](RESULTS.md).

#### Training augmentation and sampling

[`train.py`](train.py) applies random horizontal/vertical flips and brightness/colour changes. Strong recipes also use affine crop/rotation, saturation and contrast changes, occasional grayscale, blur and masking; selected recipes additionally use mixup. Validation, test and prediction inputs use deterministic preprocessing without these random augmentations.

Training samples equally from each **source × health-class** bucket with replacement. This balances training draws rather than creating equal numbers of unique images.

**Reproduction:** A fresh clone does not include the raw datasets, image caches or all historical audit inputs. Configure the source paths using [`local_config.example.json`](local_config.example.json), restore the required audited manifests and PlantWild review decisions, and follow [Training reproduction](#training-reproduction). Prediction with the included model does not require preparing the training datasets.

## Architecture and training

`model.py` defines four convolutional stages with 32, 64, 128 and 192 channels. Each stage uses two 3 × 3 convolutions, BatchNorm, SiLU and max pooling. The selected recipe adds channel attention, adaptive 2 × 2 pooling and a dense classifier. Images are EXIF-corrected, converted to RGB, resized with bilinear interpolation and normalized with `x / 127.5 - 1`.

Training draws equally from each source × health-class bucket with replacement. Duplicate families and source-photo groups stay in one partition. Validation balanced accuracy determines checkpoint selection; the classification threshold stays at 0.5. Six completed fits compare augmentation, mixup, attention and additional sources.

| Stage | Training sources | Initialization |
|---|---|---|
| A | PlantVillage, Bangladesh, TLID, PlantDoc | Random weights; three recipes |
| B | A + Tomato-Village | Random weights with the selected recipe |
| C | B + Tomato-Disease regions | This project's B checkpoint with joint replay |
| D, delivered | C + PlantWild | This project's C checkpoint with joint replay |

[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) records source links, attribution, licences, label decisions and grouping rules. Dataset photographs, caches and local review sheets are excluded from this repository. Source datasets retain their own terms, including PlantWild's CC BY-NC-ND 4.0 terms.

## Verify saved results

Install the full requirements to review the experiment. Saved prediction metrics and checkpoint identities can be verified on CPU without datasets:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dataset-dependent unit checks are skipped when local audited manifests are absent. Saved draw-count checks still run against the included training records. Frozen records preserve original run paths as provenance; the verifier resolves checkpoint identities to the included `runs/` files.

## Training reproduction

Training and fresh dataset evaluation require CUDA. The recorded environment used Python 3.12, PyTorch 2.11.0+cu128 and an RTX 4070 Ti SUPER. Install a compatible CUDA-enabled PyTorch build and the full requirements.

**Full reproduction requires additional local inputs.** Initial preparation reads audited manifests and images from earlier PlantVillage, Bangladesh, TLID and PlantDoc experiments. A fresh clone does not download those inputs. Copy `local_config.example.json` to `local_config.json` and configure both source-project paths. Git ignores your local configuration.

The preparation and training sequence was:

```text
prepare_data.py
inspect_sources.py → download_tomatovillage.py → prepare_external.py → prepare_field.py
train.py --epochs 30 --steps 64 --suffix _v1
train_field.py
download_zenodo.py → prepare_zenodo.py → train_extra.py → select_final.py
inspect_plantwild.py → download_plantwild.py → prepare_plantwild.py
train_wild.py → select_wild.py
```

Arrows indicate order, not a shell command. PlantWild also requires archived source metadata and the frozen manual decisions in `data/plantwild/healthy_review.csv`. Restore those local audit inputs to reproduce the same splits; treating every generic leaf photo as Healthy changes the experiment. The optional GTLD check contains just seven publisher examples. Read the protocol before creating new experiments. Completed run names and evaluation directories are protected against overwrite.

## Files

| Path | Purpose |
|---|---|
| `predict.py`, `model.py`, `best_model.pt` | Prediction code, architecture and selected weights |
| `prepare_*.py`, `download_*.py`, `inspect_*.py` | Data preparation, source checks, grouping and downloads |
| `train*.py`, `select_*.py` | Training and validation selection |
| `evaluate.py`, `verify_results.py`, `report.py` | Fresh evaluation, saved-result verification and local reporting |
| `runs/` | Completed checkpoints, configurations, histories, draw counts and historical source snapshots |
| `evaluation_a/` through `evaluation_final/` | Frozen identities, individual predictions and metrics |
| `evaluation_unseen_matching/` | Pakistan external-test report, predictions, counts and frozen audit evidence |
| `dataset_sources/pakistan/` | Original dataset link, attribution, licence and frozen release metadata |
| `RESULTS.md`, `RESULTS.csv`, `figures/` | Measured results and numerical plots |
| `tomato_cnn_inference.zip` | Standalone inference package |

Model SHA-256: `e905416007b09d826595af17fa4f5d5b15c103c973790853f707de73a38f1108`.

## Installation guide (Windows + VS Code)

### 1. Install the prerequisites

- [Python 3.12 for Windows](https://www.python.org/downloads/windows/), 64-bit, including the Python launcher so `py -3.12` works.
- [Git](https://git-scm.com/downloads).
- [Visual Studio Code](https://code.visualstudio.com/) with the **Python** extension installed from its Extensions panel.

The project was tested with Python 3.12. Restart VS Code after installing Python or Git.

### 2. Download and open the project

In VS Code, choose **Terminal > New Terminal** and select a **PowerShell** terminal. Enter these commands in the terminal:

```powershell
git clone https://github.com/kzy09c/cnn-plant-disease-detection.git
cd cnn-plant-disease-detection
code .
```

If you already downloaded the project, use **File > Open Folder** to open the folder containing `predict.py`, `best_model.pt` and `requirements.txt`. Open a terminal in that folder and skip cloning again.

### 3. Create and activate a virtual environment

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The prompt should start with `(.venv)`. Activate it again when opening a new terminal. Each computer creates its own `.venv`; Git ignores that folder.

If PowerShell blocks activation, skip it and use `.\.venv\Scripts\python.exe` instead of `python` in the commands below. For example:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install the dependencies

With `(.venv)` active, run:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

This installs PyTorch, NumPy, Pillow, SciPy, scikit-learn and Matplotlib. The CPU installation supports prediction, saved-result verification and the available unit checks. The included trained model works without downloading datasets or retraining. For prediction alone, `inference-requirements.txt` can replace `requirements.txt` in the last command.

### 5. Select the Python interpreter

Press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select this project's `.venv\Scripts\python.exe`. If it is missing from the list, choose **Enter interpreter path** and browse to that file. See the [VS Code environment guide](https://code.visualstudio.com/docs/python/environments) for environment selection.

### 6. Check the installation and predict an image

From the project folder, run:

```powershell
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python verify_results.py
python -m unittest discover -s tests -v
python predict.py "C:\path\to\tomato_leaf.jpg"
```

Replace the image path with one of your own tomato-leaf photographs. `CUDA available: False` is expected for the CPU installation. Verification checks 33 saved evaluation cohorts, including the Pakistan test, and their checkpoint identities. The two checks requiring local dataset manifests are skipped in a fresh clone. Prediction prints a Healthy/Unhealthy result and both model probabilities.

### Optional: GPU installation for fresh training

Fresh training requires an NVIDIA GPU, a compatible CUDA-enabled PyTorch installation and the additional data/audit inputs described in [Training reproduction](#training-reproduction). Use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install a CUDA build compatible with your system and the pinned PyTorch 2.11.0 version, in place of the CPU installation command above. The recorded training environment used PyTorch 2.11.0+cu128 on an RTX 4070 Ti SUPER.

### Linux and macOS

Create the environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`. Install `requirements.txt` with `python -m pip install -r requirements.txt`, then run the same verification and prediction commands with your own image path. CPU inference is supported; the training scripts require CUDA.

<p align="center"><img src="images/anime-bocchi.gif" alt="Project GIF" width="400"></p>
