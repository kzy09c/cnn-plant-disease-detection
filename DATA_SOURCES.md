# Data sources, eligibility, and provenance

## PlantVillage

- Source: [original PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset), [author-hosted archive](https://huggingface.co/datasets/mohanty/PlantVillage).
- Research: [Mohanty et al., Using Deep Learning for Image-Based Plant Disease Detection (2016)](https://doi.org/10.3389/fpls.2016.01419).
- Only original RGB tomato images are used; grayscale and segmented copies are omitted. Retain attribution and consult the source's terms when redistributing images.
- There are 18,160 images: 14,528 development and 3,632 test. Healthy maps from Tomato___healthy; every other source tomato category maps to Unhealthy, including spider-mite damage.
- Development: 1,273 Healthy / 13,255 Unhealthy. Test: 318 Healthy / 3,314 Unhealthy.
- The recorded same-leaf/duplicate groups, test partition, and five validation folds are preserved from the original experiment. Source-class names remain in the manifest for auditing; the model's output head has only two classes.
- Leaf metadata are incomplete. The test partition was inspected in earlier experiments, so it is a reused benchmark rather than a fresh independent cohort. Controlled-background results do not establish field-photo accuracy.

## Bangladesh

- Authors: Mangsura Kabir Oni and Tabia Tanzin Prama (2024).
- Source: [Tomato Leaf Image Dataset for Disease Analysis in Real-World Environment, Mendeley V1](https://data.mendeley.com/datasets/rnbsw72zb5/1), DOI 10.17632/rnbsw72zb5.1.
- Paper: [A comprehensive dataset of tomato leaf images for disease analysis in Bangladesh](https://pmc.ncbi.nlm.nih.gov/articles/PMC11835585/).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [BANGLADESH_LICENSE.txt](BANGLADESH_LICENSE.txt) for attribution and modifications.
- Publisher archive: Tomato_Leaves.zip, 1,084,433,928 bytes, SHA-256 `5d3ca45803b0a3391e0cdaa3017db09cdd1119bac3725a8667356cdc090c9536`. Archive SHA-256 and ZIP CRC were verified.

Actual download: **1,030 images**, comprising **484 H/Healthy and 546 D/Unhealthy**, with **1,028 annotation files**. The released train folder has 870 images and val has 160. These counts differ from the paper and download-page description. Full provenance and hashes remain in the downloadable-data index and the tracked split manifest.

The H/D filenames supply whole-image labels. Annotation IDs 0=Healthy and 1=Unhealthy are inferred from dominant agreement with those names; no new biological diagnosis is performed. Training and the primary test require every supplied annotation class to agree with its filename. The predefined audit excludes:

| Reason | Development-source exclusions | Holdout exclusions | Total |
|---|---:|---:|---:|
| Filename/annotation disagreement or mixed labels | 66 | 21 | 87 |
| Missing annotations | 2 | 0 | 2 |
| Invalid zero-area annotation | 1 | 0 | 1 |

The primary cohort has **801 development images (349 Healthy / 452 Unhealthy)** and **139 test images (59 Healthy / 80 Unhealthy)**. The original val folder remains reserved: five grouped folds are created only inside the eligible original train images. The full 160-image val folder is additionally scored descriptively against filename labels, with its 21 contradictory references clearly marked.

Whole images are used, retaining natural backgrounds; boxes are used for label auditing rather than input cropping. The publication describes photos from two gardens. This is limited geographic/source coverage, not a prospective evaluation from new gardens.

## Similarity and pretraining checks

Bangladesh copy families are inferred only from explicit class-local filename IDs and copy suffixes. Exact file/pixel duplicates and pHash distances <=4 confirmed by 64×64 block SSIM >=0.97 form connected groups. Any training group overlapping a released holdout image is removed. The archive's two unannotated extra copies are captured by this audit; the recorded eligible development/test images have distinct groups.

All 18,160 original PlantVillage tomato images are checked for exact/structurally confirmed overlap before considering PlantVillage feature adaptation for Bangladesh. No additional confirmed overlap exclusions were identified in the completed run. The cached fingerprints can be recreated with ensure_plantvillage_fingerprints.py; no other dataset is needed.

These checks cannot recover all missing plant/photo-session relationships, and similarity thresholds are not a guarantee of complete independence. Label agreement is not expert ground truth. Both limitations are reflected in the report interpretation.

## Imported evidence and reproducibility

The binary checkpoints and image-level test predictions were copied unchanged from the completed training runs. Each experiment.json records the original experiment/manifest hashes and import provenance. Repository cleanup did not generate new accuracy values by retraining. verify_results.py recalculates confusion-based and per-class metrics from CSV, checks group/pixel/file separation, validates CV summaries/history lengths, and loads the exact binary checkpoints on CPU.

For reproducible fresh runs, train.py uses only these two datasets and writes to a new --output directory. The scripts retain the recorded architecture and optimization recipes, while refactoring/random-number consumption and nondeterministic GPU kernels mean a rerun need not be bit-identical. Downloaded datasets and local experiments are ignored by Git; the shipped metrics remain available for report reproduction without downloading the source images.
