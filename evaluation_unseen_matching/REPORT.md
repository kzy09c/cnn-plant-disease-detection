# Category-matched external dataset evaluation

Source: [Tomato Leaf Disease Classification Dataset in Pakistan, release 2](https://data.mendeley.com/datasets/3mbnb82mxd/2); Mohammad Naeem Malik, Syed Hashir Ali and Hamid Ali; CC BY 4.0.

## What was tested

The existing delivered custom CNN was tested once without retraining. It predicts Healthy or Unhealthy. The disease categories below are publisher reference groups used to break down correct binary decisions; they are not disease names predicted by this model. All six categories are also present in the PlantVillage portion of its training sources.

Checkpoint: `best_model.pt`; SHA-256 `e905416007b09d826595af17fa4f5d5b15c103c973790853f707de73a38f1108`. Fixed threshold 0.5. Original preprocessing: EXIF-corrected RGB, bilinear resize 160x160, x/127.5-1. No test-specific crop or background adjustment.

## Data and unseen-image audit

The archive contains 830 raw images plus 7,200 augmented images. All raw images were considered; no augmented copies were included. After pre-prediction duplicate/overlap/reference checks, **461 eligible images** remained: 115 Healthy and 346 Unhealthy. Exclusions were decided before predictions.

The audit compared images with 51,059 earlier image/parent records, covering the seven training sources and earlier evaluation sources. Initial decoded-image hashes and pHash<=4 with SSIM>=.97 checks were followed by SIFT feature matching, RANSAC homography and aligned-pixel correlation checks for cropped/resized copies. Confirmed overlap families were excluded and one representative retained per remaining photo family. The final 461 images are smaller than the initial 498-image structural-audit cohort; both audits were completed before any model inference.

In total, 357 raw entries belonged to detected photo families overlapping earlier data; another 12 were duplicate/cropped/resized copies within this release. Initial direct overlap matches included PlantDoc, Tomato-Disease parent/crop images, PlantWild and two older web examples. Therefore the whole release is not a clean independent new collection. Results below apply to the remaining images with no detected matches; undetected crops, views and session relationships can still remain.

| Reference category | Matching training category | Raw candidates | Excluded | Tested | Correct binary predictions | Incorrect | Correct rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Early blight | Tomato___Early_blight | 119 | 43 | 76 | 68 | 8 | 89.47% |
| Healthy | Tomato___healthy | 137 | 22 | 115 | 91 | 24 | 79.13% |
| Late blight | Tomato___Late_blight | 113 | 83 | 30 | 26 | 4 | 86.67% |
| Leaf mold | Tomato___Leaf_Mold | 186 | 40 | 146 | 146 | 0 | 100.00% |
| Septoria leaf spot | Tomato___Septoria_leaf_spot | 195 | 134 | 61 | 60 | 1 | 98.36% |
| Yellow leaf curl virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 80 | 47 | 33 | 17 | 16 | 51.52% |

## Overall result

**408/461 classified correctly; 53 incorrectly.**

| Metric | Result |
|---|---:|
| Accuracy | 88.50% |
| Balanced accuracy | 85.37% |
| Macro F1 | 84.87% |
| Healthy recall | 79.13% |
| Unhealthy recall | 91.62% |
| Balanced accuracy bootstrap 95% interval | 81.74%–89.33% |

Confusion matrix: reference rows and prediction columns.

| Reference / prediction | Healthy | Unhealthy |
|---|---:|---:|
| Healthy | 91 | 24 |
| Unhealthy | 29 | 317 |

Always predicting Unhealthy would score 75.05% ordinary accuracy and 50.00% balanced accuracy.

## Main findings

- **408/461 correct; 53 incorrect.** Of the 115 Healthy images, 91 were recognized and 24 were incorrectly marked Unhealthy. Of the 346 Unhealthy images, 317 were recognized and 29 were missed.
- **Leaf mold: 146/146** and **Septoria leaf spot: 60/61** were the strongest category checks in this cohort. These are correct binary Unhealthy decisions, not disease-name diagnoses.
- **Yellow leaf curl virus: 17/33** was the weakest check. Its 16 missed images account for 16 of the 29 total Unhealthy-to-Healthy mistakes.
- Compared with the earlier Agri-Vision4 check, accuracy was higher (88.50% versus 80.15%) and BA was higher (85.37% versus 80.92%). This uses different images, class proportions, references and audit methods; it does not demonstrate that matching category names alone caused the difference.
- The 85.37% BA remains below the 90% working target. The dataset had substantial overlap with earlier sources, so the result concerns the audit-retained subset; it does not establish independence of acquisition sessions or internet-wide performance.

## Interpretation and limits

This directly checks transfer to new images whose category names already occur in the training sources. Matching category names does not establish matching lighting, background, severity, acquisition or reference-label quality. The source labels are publisher references, not independent diagnoses performed here. Dataset acquisition/location claims are not independently verified. Any comparison with other datasets uses different photos and cannot isolate the causal effect of matching category names.

The per-category table reports correct Healthy/Unhealthy decisions only. A correct Unhealthy prediction on Early Blight does not mean that the model identified Early Blight by name. No additional model fitting or threshold tuning followed this test.

The repository includes [individual predictions](predictions.csv), [category metrics](category_metrics.csv), the [complete 830-row audit manifest](audit_manifest.csv), frozen protocol/audit records, geometric match evidence and [independent verification](verification.json). Source attribution, licence and download instructions are in the [dataset guide](../dataset_sources/pakistan/README.md). Original photographs remain available from the official release and in the local `data/3mbnb82mxd_unseen/` cache.
