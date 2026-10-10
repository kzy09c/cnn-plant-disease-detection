# 🍅 PlantVillage Tomato Disease Detection Using CNN

<p align="center">
  <img src="images/spinn.gif" alt="Project GIF" width="500">
</p>

<p align="center">
  <strong>Tomato Leaf Disease Classification Using Convolutional Neural Networks and PyTorch</strong>
</p>

---

## 📖 Project Overview

This project implements Convolutional Neural Networks (CNNs) to classify tomato leaf images from the **PlantVillage dataset** into 10 categories, including healthy leaves, eight disease categories, and spider-mite damage.

The project adapts the Tutorial 5 CNN workflow and uses PyTorch for model development, training, and evaluation.

Two custom CNN architectures are trained from scratch using grouped five-fold cross-validation. The model configuration achieving the highest mean macro F1-score is retrained on the development dataset and evaluated on a held-out test set.

To minimise data leakage, known same-leaf groups and exact duplicate images are assigned to the same partition. Incomplete leaf metadata introduces some uncertainty, which is documented in the experiment report.

### Key Features

- Tomato leaf classification across 10 classes
- Two custom CNN architectures built using PyTorch
- Grouped five-fold cross-validation
- Model selection based on mean macro F1-score
- Held-out test evaluation
- Trained model checkpoint for prediction
- Confusion matrix and performance visualisations
- Reproducible training and evaluation workflow
- Jupyter Notebook for explaining the experiment

---

## 📊 Results and Model Outputs

The experiment's measured results are stored in the `outputs/` directory.

| File | Description |
|---|---|
| `outputs/REPORT.md` | Detailed experiment results and analysis |
| `outputs/tomato_cnn.pt` | Trained CNN model checkpoint |
| `outputs/test_metrics.json` | Test evaluation metrics |
| `outputs/confusion_matrix.png` | Confusion matrix visualisation |
| `outputs/cross_validation_curves.png` | Cross-validation performance curves |

The reported metrics are calculated from the trained models and held-out evaluation, rather than assumed from published models.

The `Tomato_CNN.ipynb` notebook explains the network architectures, experimental protocol, and recorded performance. Fresh training is disabled by default to preserve the completed test evaluation.

---

## 🚀 Getting Started

### 1. Prerequisites

Before starting, make sure you have installed:

- [Python 3.12 (64-bit)](https://www.python.org/downloads/release/python-31210/)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/)
- VS Code Python extension
- VS Code Jupyter extension

An NVIDIA GPU with a compatible CUDA driver is required for GPU training. CPU execution is supported for prediction and reviewing the completed experiment.

### 2. Clone the Repository

Open a terminal in VS Code and enter:

```bash
git clone https://github.com/tzhe0700/cnn-plant-disease-detection.git
cd cnn-plant-disease-detection
code .
```

This downloads the project and opens it in Visual Studio Code.

### 3. Create a Virtual Environment

Each computer should use its own Python virtual environment.

Open the VS Code PowerShell terminal and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run Python commands directly with:

```powershell
.\.venv\Scripts\python.exe
```

Virtual environments should not be committed to GitHub because they are specific to the computer where they were created.

### 4. Install Dependencies

For CPU-based inference and reviewing results, install PyTorch and Torchvision using the CPU package index:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt -r notebook-requirements.txt
```

The principal Python dependencies include:

- PyTorch and Torchvision
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- IPython Kernel
- nbformat
- nbclient

For GPU-based training, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for a CUDA-compatible installation command.

**Note:** The pinned PyTorch and Torchvision versions should be checked for compatibility and package availability before installation. The experiment's recorded package versions are documented in `environment-lock.txt`.

### 5. Configure the Jupyter Kernel

Register the virtual environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name plant-tomato-cnn --display-name "Tomato CNN (Python 3.12)"
```

In VS Code:

1. Open `Tomato_CNN.ipynb`.
2. Click **Select Kernel** in the top-right corner.
3. Choose **Jupyter Kernels**.
4. Select **Tomato CNN (Python 3.12)**.
5. Run the notebook cells to explore the completed experiment.

To select the Python interpreter for scripts, press `Ctrl + Shift + P`, search for **Python: Select Interpreter**, and choose the `.venv` interpreter.

---

## 🔍 Using the Trained Model

The trained model can be used to classify new tomato leaf images.

Activate the virtual environment, then run:

```powershell
python predict.py "C:\path\to\tomato-leaf.jpg"
```

Replace the example path with the location of your own image.

If stored in this repository in manual_test:
```powershell
python predict.py manual_test\name_of_test.jpg
```

To verify the stored experiment results:

```powershell
python verify_results.py
```

To generate the experiment report:

```powershell
python make_report.py
```

To execute the explanatory notebook without retraining:

```powershell
python execute_notebook.py
```

### Model Limitations

- Input images are resized to **96 × 96 pixels**.
- The model is designed for individual tomato leaf images.
- Prediction can be performed on CPU.
- Training is configured for CUDA.
- The model has not been evaluated on field photographs, entire tomato plants, or tomato fruits.

---

## 🧠 Training the CNN

Fresh training requires Python 3.12 and an NVIDIA GPU with a compatible CUDA driver.

To train a new model, create a separate experiment directory without the existing `outputs/` folder. This prevents the completed evaluation results from being overwritten.

### Step 1: Install GPU Dependencies

For a CUDA 12.8 environment, use a compatible PyTorch and Torchvision build from the official PyTorch package index.

Then install the remaining requirements:

```powershell
python -m pip install -r requirements.txt -r notebook-requirements.txt
```

### Step 2: Download the Dataset

```powershell
python download_data.py
```

The download script retrieves original RGB tomato leaf images and available leaf-group metadata from the PlantVillage dataset.

Grayscale and segmented image copies are excluded.

### Step 3: Train the Model

```powershell
python train.py --epochs 20 --batch-size 128 --image-size 96
```

The training configuration uses:

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 128 |
| Image size | 96 × 96 |
| Cross-validation | Grouped 5-fold |
| Model selection | Mean macro F1-score |
| Training framework | PyTorch |

### Step 4: Evaluate and Generate Reports

```powershell
python verify_results.py
python make_report.py
```

The results are saved in the `outputs/` directory.

**Important:** Training is designed to refuse a rerun when `outputs/test_metrics.json` already exists. Use a separate experiment directory for additional training to preserve the independence of the held-out test evaluation.

---

## 📁 Experiment Evidence

The experiment generates supporting files for reproducibility and evaluation.

| File | Purpose |
|---|---|
| `outputs/experiment.json` | Dataset counts, class definitions, and training configuration |
| `outputs/split_manifest.csv` | Image paths, group assignments, dataset splits, and hashes |
| `outputs/cross_validation.json` | Cross-validation model-selection results |
| `outputs/*_history.json` | Training loss and validation metrics across epochs |
| `outputs/test_predictions.csv` | Individual predictions and class probabilities |
| `outputs/test_metrics.json` | Overall and per-class performance metrics |
| `outputs/confidence_intervals.json` | Bootstrap uncertainty estimates |
| `outputs/verification.json` | Independent verification of evaluation results |
| `outputs/confusion_matrix.png` | Test-set confusion matrix |
| `outputs/cross_validation_curves.png` | Cross-validation performance visualisations |

---

## 🔄 GitHub Basics

This section explains how to keep the project synchronised between GitHub and different computers.

### Getting the Latest Code

Before starting work, pull the latest changes:

```bash
git pull origin main
```

### Saving and Uploading Changes

After editing your code:

```bash
git add .
git commit -m "Updated project"
git push origin main
```

### Returning to a Previous Version

View previous commits:

```bash
git log --oneline
```

Temporarily view an older version:

```bash
git switch --detach COMMIT_ID
```

Return to the latest version of the main branch:

```bash
git switch main
```

To safely reverse a committed change while preserving Git history:

```bash
git revert COMMIT_ID
git push origin main
```

### Working Across Multiple Computers

To use this project across different computers:

1. **First-time setup:** Clone the repository and install its dependencies.
2. **Before working:** Run `git pull origin main`.
3. **Make changes:** Modify and test your code.
4. **Save changes:** Commit your work.
5. **After working:** Run `git push origin main`.
6. **Switching computers:** Pull the latest changes before continuing.

Always commit or otherwise protect uncommitted changes before switching branches or restoring older versions.

---

## 📚 References

**Dataset**

- [PlantVillage Dataset — Original GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
- [PlantVillage Dataset — Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)

**Research**

- [Mohanty et al. (2016) — Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419)

**Documentation**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Python Virtual Environments](https://docs.python.org/3.12/library/venv.html)
- [Git Documentation](https://git-scm.com/doc)

---

## 📝 Project Notes

This project is intended for educational and experimental purposes.

The trained CNN demonstrates image-based tomato leaf classification under the PlantVillage dataset conditions. Performance on real-world agricultural images may differ due to lighting, backgrounds, image quality, and other environmental factors.

For detailed evaluation results and documented experimental limitations, refer to `outputs/REPORT.md`.


<p align="center">
  <img src="images/anime-bocchi.gif" alt="Project GIF" width="500">
</p>