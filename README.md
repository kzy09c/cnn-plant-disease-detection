# Tomato leaf health: PlantVillage + Bangladesh CNNs

<p align="center"><img src="images/spinn.gif" alt="Project GIF" width="400"></p>

Three trained binary models classify tomato leaf images as **Healthy** or **Unhealthy**: PlantVillage, Bangladesh from scratch, and PlantVillage followed by Bangladesh fine-tuning. All use the same custom PyTorch CNN with 939,874 parameters and 128×128 RGB input. Unhealthy includes the source disease, pest-damage, and stress categories; no specific disease name is predicted.

The original PlantVillage workflow is now binary. The separate Bangladesh experiment uses natural-background field photos and balanced training batches. The trained checkpoints are included in this repository.

**First time using the project?** Follow the [installation guide](#installation-guide-windows--vs-code) below.

## Actual results

| Dataset / test cohort | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |
|---|---:|---:|---:|---:|---:|---:|
| PlantVillage binary | 3,632 | **99.86%** | **99.92%** | **99.57%** | 100.00% (318/318) | 99.85% (3,309/3,314) |
| Bangladesh: primary label-agreement cohort | 139 | **94.96%** | **94.07%** | **94.75%** | 88.14% (52/59) | 100.00% (80/80) |
| Bangladesh: full holdout, uncertain filename labels | 160 | **83.75%** | **83.75%** | **83.31%** | 67.50% (54/80) | 100.00% (80/80) |
| PlantVillage → Bangladesh fine-tuned: primary cohort | 139 | **87.77%** | **85.82%** | **86.89%** | 72.88% (43/59) | 98.75% (79/80) |
| PlantVillage → Bangladesh fine-tuned: full holdout, uncertain labels | 160 | **79.38%** | **79.38%** | **78.57%** | 60.00% (48/80) | 98.75% (79/80) |

These are different image populations, not a controlled ranking of field performance. PlantVillage uses controlled leaf images and a previously inspected benchmark test. Bangladesh's full holdout includes 21 filename labels that contradict the annotations; the 139-image primary cohort was defined by label agreement before training. Agreement does not establish expert diagnosis. Performance in new gardens remains untested.

**Report material:** [full results](RESULTS.md), [CSV summary](RESULTS.csv), [figure captions and SVG/PNG index](figures/README.md), and [CNN explanation](CNN_GUIDE.md). Each output folder also includes per-image predictions, class metrics, uncertainty intervals, CV summaries, ROC points, and precision–recall points as CSV files.

**New sequential-training experiment:** [fine-tuning results and comparison](FINETUNING_RESULTS.md), [comparison CSV](FINETUNING_RESULTS.csv), and [executed notebook](PlantVillage_Bangladesh_Finetuning.ipynb). This model keeps the entire trained PlantVillage binary CNN and updates all layers on Bangladesh. It scored below the Bangladesh scratch model on the same previously inspected holdout, so scratch remains the default Bangladesh model. The comparison is exploratory and uses different training budgets; it is not a new independent test.

![Shared CNN architecture](figures/cnn_architecture.png)

![Experiment workflow](figures/experiment_workflow.png)

![Dataset counts](figures/dataset_counts.png)

## Installation guide (Windows + VS Code)

### 1. Install the prerequisites

- [Python 3.12, 64-bit Windows installer](https://www.python.org/downloads/release/python-31210/). Include the Python launcher during installation so that `py -3.12` works.
- [Git](https://git-scm.com/downloads).
- [Visual Studio Code](https://code.visualstudio.com/), with the **Python** and **Jupyter** extensions installed from its Extensions panel.

The experiments used Python 3.12. You can install it alongside another Python version. Restart VS Code after installing Python or Git.

### 2. Download and open the project

In VS Code, choose **Terminal > New Terminal** and use a **PowerShell** terminal. Run these commands in the terminal, not inside the Python `>>>` prompt or a notebook cell:

```powershell
git clone https://github.com/kzy09c/cnn-plant-disease-detection.git
cd cnn-plant-disease-detection
code .
```

If you already downloaded the project, choose **File > Open Folder** and select the folder containing `train.py` and `requirements.txt`. Open a terminal in that folder; skip cloning it again.

### 3. Create and activate your virtual environment

Each computer creates its own `.venv` folder. It is ignored by Git and should not be uploaded to the repository.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should now start with `(.venv)`. Activate the environment again whenever you open a new terminal for this project.

### 4. Install the dependencies

With `(.venv)` active, run:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r notebook-requirements.txt
```

These install PyTorch, Torchvision, NumPy, Pillow, scikit-learn, Matplotlib, SciPy, and the Jupyter packages needed by the notebooks. The CPU installation supports prediction, reviewing the notebooks, verifying results, and generating figures. You do not need to download either dataset or train a model to use the included checkpoints.

If PowerShell blocks activation, run these equivalent commands directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r notebook-requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name plant-tomato-cnn --display-name "Tomato CNN (Python 3.12)"
```

For subsequent terminal commands, use `.\.venv\Scripts\python.exe` in place of `python` if activation is blocked.

### 5. Select the Python interpreter and notebook kernel

For Python scripts, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select this project's `.venv\Scripts\python.exe`.

With the environment active, register its notebook kernel:

```powershell
python -m ipykernel install --user --name plant-tomato-cnn --display-name "Tomato CNN (Python 3.12)"
```

Open `Tomato_CNN.ipynb`, `Bangladesh_CNN.ipynb`, or `PlantVillage_Bangladesh_Finetuning.ipynb`. Click **Select Kernel** in the top-right corner, then choose **Tomato CNN (Python 3.12)** under the available Jupyter kernels. You can also select the project's `.venv` under **Python Environments**. Run the cells to explore the saved experiment.

If a notebook says `ModuleNotFoundError: No module named 'torch'`, select the same `.venv` kernel you installed the packages into and restart the notebook kernel before rerunning the cells.

### 6. Check the installation

From the project folder, run:

```powershell
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python verify_results.py
python predict.py manual_test/plantvillage_binary_sample_healthy.jpg
python predict.py manual_test/bangladesh_sample_unhealthy.jpg --dataset bangladesh
```

`CUDA available: False` is expected for the CPU installation. Verification should pass for all saved experiments; prediction prints a Healthy/Unhealthy result and both class scores.

### Optional: GPU installation for fresh training

Fresh training requires an NVIDIA GPU and a compatible CUDA-enabled PyTorch installation. Use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to choose a CUDA build matching your system and the project's pinned PyTorch/Torchvision versions, instead of the CPU installation command above. The recorded training environment is in [environment-lock.txt](environment-lock.txt). Then follow [Fresh training](#fresh-training) below, using a new output directory.

## Predict an image

PlantVillage is the default binary model:

```powershell
python predict.py "C:\path\to\leaf.jpg"
```

Use the Bangladesh binary model:

```powershell
python predict.py "C:\path\to\leaf.jpg" --dataset bangladesh
```

Use the new PlantVillage → Bangladesh fine-tuned model:

```powershell
python predict.py "C:\path\to\leaf.jpg" --dataset bangladesh-finetuned
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
- [PlantVillage_Bangladesh_Finetuning.ipynb](PlantVillage_Bangladesh_Finetuning.ipynb): the full PlantVillage model fine-tuned on Bangladesh, compared with scratch training.

The notebooks show actual saved outputs without retraining or another test inference pass.

```powershell
python verify_results.py
python make_report.py
python report_finetune.py
python execute_notebook.py Tomato_CNN.ipynb
python execute_notebook.py Bangladesh_CNN.ipynb
python execute_notebook.py PlantVillage_Bangladesh_Finetuning.ipynb
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
| PlantVillage → Bangladesh fine-tuned | `bangladesh_finetuned_outputs/tomato_cnn_finetuned.pt` | [FINETUNING_RESULTS.md](FINETUNING_RESULTS.md) | `PlantVillage_Bangladesh_Finetuning.ipynb` |

The original two models and saved test predictions are imported unchanged from the completed binary training experiments. Their `experiment.json` files record import provenance. The additional fine-tuned model was actually trained for this experiment; its new output folder records source-checkpoint hashes, learning-rate recipes, fold histories, individual balanced training draws, the validation-only selection and final predictions. Reruns are not guaranteed bit-identical because GPU kernels and random-number consumption can vary.

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

For sequential PlantVillage → Bangladesh training, download both datasets using the commands above, retain the included PlantVillage checkpoint, and run:

```powershell
python train_finetune.py --output runs/bangladesh-finetuned-new
python verify_results.py --dataset bangladesh-finetuned --output runs/bangladesh-finetuned-new
```

The default fine-tuning experiment compares two learning-rate settings across five grouped development folds for 20 epochs each, then reloads the original full PlantVillage checkpoint for 30 final Bangladesh epochs. Every layer, the retained binary output head, and BatchNorm running statistics adapt. Original PlantVillage test images and Bangladesh test images do not update weights. See [FINETUNING_RESULTS.md](FINETUNING_RESULTS.md) for the complete protocol and the repeated-test limitation.

## Dataset provenance

See [DATA_SOURCES.md](DATA_SOURCES.md) for source links, original counts, annotation conflicts, duplicate/pretraining checks, and limitations. See [BANGLADESH_LICENSE.txt](BANGLADESH_LICENSE.txt) for Bangladesh CC BY 4.0 attribution and changes.

<p align="center"><img src="images/anime-bocchi.gif" alt="Project GIF" width="400"></p>
