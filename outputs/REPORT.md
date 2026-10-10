# Tomato leaf disease CNN - measured results

Trained from scratch on 18,160 original PlantVillage RGB tomato images. Selected **deeper CNN**, with **940,906 trainable parameters**, using five-fold cross-validation. The final model was trained on all 14,528 development images and evaluated once on 3,632 held-out images.

## Held-out test results

| Metric | Score | 95% group bootstrap interval |
|---|---:|---:|
| Accuracy | 98.46% | 97.93% - 98.94% |
| Macro precision | 98.09% | 97.36% - 98.73% |
| Macro recall / balanced accuracy | 98.38% | 97.84% - 98.88% |
| Macro F1 | 98.23% | 97.62% - 98.79% |
| Weighted F1 | 98.46% | 97.94% - 98.94% |

Macro one-vs-rest ROC AUC: **0.9998**. Multiclass log loss: **0.0464**.

Macro F1 gives each class equal importance. Weighted F1 gives each class importance proportional to its test support. Accuracy is the fraction of correctly classified images. Precision measures how often a predicted class is correct; recall measures how many real examples of that class are found. F1 is their harmonic mean. Balanced accuracy equals macro recall here.

The intervals resample complete test leaf/duplicate groups 2,000 times. They describe test-sample uncertainty conditional on this trained model and this dataset. They do not include variation from retraining, uncertain leaf metadata, or deployment on field images.

## Per-class test results

| Class | Test images | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Bacterial spot | 427 | 97.44% | 97.89% | 97.66% |
| Early blight | 200 | 95.52% | 96.00% | 95.76% |
| Late blight | 382 | 98.18% | 98.95% | 98.57% |
| Leaf mold | 188 | 98.94% | 98.94% | 98.94% |
| Septoria leaf spot | 354 | 98.58% | 98.02% | 98.30% |
| Spider mites two-spotted spider mite | 336 | 96.49% | 98.21% | 97.35% |
| Target spot | 281 | 97.14% | 96.80% | 96.97% |
| Yellow leaf curl virus | 1072 | 99.91% | 98.97% | 99.44% |
| Mosaic virus | 74 | 98.67% | 100.00% | 99.33% |
| Healthy | 318 | 100.00% | 100.00% | 100.00% |

## Detecting unhealthy versus healthy leaves

For this secondary task, disease and spider-mite classes are grouped as unhealthy. The positive prediction is any multiclass prediction other than healthy. This rule was set before seeing the test results; no threshold was tuned on test data.

| Metric | Score |
|---|---:|
| Accuracy | 100.00% |
| Unhealthy precision | 100.00% |
| Unhealthy recall / sensitivity | 100.00% |
| Healthy recall / specificity | 100.00% |
| Unhealthy F1 | 100.00% |
| ROC AUC | 1.0000 |
| Average precision (PR) | 1.0000 |

Counts: 318 healthy leaves correctly called healthy, 0 healthy leaves incorrectly called unhealthy, 0 unhealthy leaves incorrectly called healthy, and 3314 unhealthy leaves correctly called unhealthy.

## Five-fold model comparison

| Configuration | Convolution channels | CV accuracy (mean +/- sample SD) | CV macro F1 (mean +/- sample SD) |
|---|---|---:|---:|
| Compact | [16, 32, 64] | 95.28% +/- 0.60% | 94.04% +/- 0.88% |
| Deeper | [32, 64, 128, 192] | 97.56% +/- 0.29% | 97.04% +/- 0.31% |

Each model was reinitialized for every fold. Both configurations used the same folds and the same fixed 20-epoch budget. Selection used mean macro F1 at epoch 20. These CV estimates were used for model selection; the independent held-out test is the final evaluation.

## Training and evaluation protocol

1. Use original RGB tomato images only; exclude grayscale, segmented, and pre-augmented copies.
2. Group known same-leaf images using the author's leaf map. Group unmatched images by normalized original filename identifiers. Merge exact file duplicates and exact decoded-pixel duplicates into the same group.
3. Create an approximate-stratified 80/20 split with `StratifiedGroupKFold` (seed 42). Create five further grouped folds within development data (seed 43). Verify there are no group overlaps and that every validation/test partition contains all ten classes.
4. Resize to 96 x 96 RGB, scale to [0,1], normalize using fixed mean and standard deviation of 0.5. Apply random flips, transpose, brightness and mild colour shifts to training batches only.
5. Use two 3x3 convolutions, batch normalization and ReLU per block, then max pooling. Use adaptive 2x2 average pooling, a 128-unit dense layer, ReLU, dropout 0.35, and ten logits. The compact CNN has three blocks and the deeper CNN has four.
6. Train with AdamW (initial learning rate 0.001, weight decay 0.0001), cosine decay to 0.00001, batch size 128, 20 epochs and CUDA mixed precision. Calculate inverse-frequency cross-entropy weights from each fold's training labels only.
7. Select the configuration by mean CV macro F1. Reinitialize and retrain on all development images for 20 epochs. Run inference on the held-out test once. Compute all reported metrics and plots from those saved predictions.

GPU: NVIDIA GeForce RTX 4070 Ti SUPER. PyTorch: 2.11.0+cu128. Recorded completed training/validation time: 24.8 minutes (excluding downloads, package installation, image auditing and interrupted partial runs). Seeds are fixed; CUDA kernels may introduce small numerical differences across reruns. A terminal session reset interrupted deeper fold 4; its incomplete history was replaced by a fresh run using the same seed and settings. The eight completed folds were retained. No test inference had occurred before the interruption.

## Dataset and limitations

The dataset contains 9,589 identified leaf/duplicate groups. Leaf-map metadata matched 11,411 / 18,160 images (62.84%). There are 14 repeated exact images. Known groups and duplicates do not cross the test/development or CV boundaries. For unmatched images, filename grouping cannot guarantee that every photograph of the same physical leaf has been identified.

The labels are healthy, bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow leaf curl virus, and mosaic virus. Spider mites represent pest damage. This is classification of a single leaf photograph, not localization of diseased leaves in a whole-plant scene or a diagnosis of tomato fruit disease.

The largest class is yellow leaf curl virus. A development-majority classifier would obtain 29.52% test accuracy by always predicting that class. An always-unhealthy binary classifier would obtain 91.24% accuracy; this is why healthy specificity and macro F1 are useful alongside accuracy.

PlantVillage images were collected under controlled conditions. These measured scores establish performance on held-out PlantVillage tomato images. They do not establish accuracy on new farms, backgrounds, cameras, weather conditions, mixed infections, or unseen diseases. Field use needs an independent field-photo evaluation. Softmax probabilities have not been calibrated.

## Files and reproduction

- `tomato_cnn.pt`: trained model weights and preprocessing/class metadata.
- `test_metrics.json`: complete measured test results.
- `test_predictions.csv` and `test_predictions.npz`: saved predictions, allowing metric verification without another test inference pass.
- `split_manifest.csv`: class, split, fold, grouping and SHA-256 audit for every image.
- `cross_validation.json` and `*_history.json`: CV summaries and epoch histories.
- `confusion_matrix.png` and `cross_validation_curves.png`: evaluation plots.

Run from the repository root in PowerShell:

```powershell
.\plant_tomato\.venv\Scripts\python.exe plant_tomato\predict.py 'C:\path\to\leaf.jpg'
.\plant_tomato\.venv\Scripts\python.exe plant_tomato\make_report.py
```

For a fresh experiment, create a separate copy/output directory before running `train.py`. It refuses to rerun when `test_metrics.json` exists, protecting the completed test evaluation. To install elsewhere, install GPU PyTorch for your CUDA driver, then `pip install -r requirements.txt`, run `download_data.py`, and run `train.py`.

## Sources

The user's Tutorial 5, Exercise 2 (page 4) motivated the two-configuration, five-fold CV and one-time held-out evaluation protocol. Its CIFAR-10 example was adapted to PlantVillage tomatoes. The supplied Canvas template notebook was not available, so this implementation was created independently.

- [Author's PlantVillage dataset repository and leaf metadata](https://github.com/spMohanty/PlantVillage-Dataset)
- [Author's PlantVillage download](https://huggingface.co/datasets/mohanty/PlantVillage)
- [Mohanty, Hughes and Salathe (2016), Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419)
- [scikit-learn grouped stratified cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
