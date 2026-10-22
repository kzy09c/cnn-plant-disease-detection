# Custom tomato-leaf CNN experiments — 22 October 2026

Task: Healthy versus Unhealthy tomato leaves. Unhealthy includes source-labelled disease, insect damage and nutrient deficiency. The system does not identify individual diseases or verify that an input is a tomato leaf.

## Model and data

All trainable weights begin at random initialization. No ResNet, ImageNet weights, pretrained backbone, pretrained detector, teacher model or external image embedding is used. The sequential CNN has four stages, each with two 3×3 convolutions, BatchNorm and SiLU, followed by max pooling. A 2×2 adaptive average pool and dense classifier produce two logits. One recipe adds learned channel attention. Input: EXIF-corrected 160×160 RGB, bilinear resize, normalization `x/127.5 - 1`.

Experiment A combines the audited development pools of PlantVillage, Bangladesh, whole-leaf TLID and tomato-only PlantDoc. Original fold 1 supplies validation; other development folds supply training. Previous test partitions remain reserved. The existing 16 web photos are reserved. Exact duplicates, pre-existing source groups and structurally confirmed perceptual matches are grouped across sources. A group spanning partitions is promoted to the most reserved partition. Contradictory labels on identical decoded pixels cause exclusion; differing labels within a source-photo/session group are allowed.

Training uses equal draws from each source × binary-class bucket. Sampling retains the complete eligible training pool and draws with replacement. It does not create equal numbers of unique photographs in each class. Every 96-image batch contains 12 draws per bucket for four sources.

Three prospectively chosen recipes: mild augmentation, strong augmentation with mixup, strong augmentation with channel attention and mixup. AdamW, 30 maximum epochs, 64 batches per epoch, seed 20261022. Validation patience is 12 epochs after epoch 20. Recipe details, actual draws, training curves, code hashes and split hashes are stored with each run. Selection maximizes `0.6 × mean source balanced accuracy + 0.4 × minimum source balanced accuracy`. Classification uses fixed probability threshold 0.5. No test score selects epochs, hyperparameters or thresholds.

An interrupted one-epoch pilot in `runs/mixed_mild/` used an overly conservative draft audit that excluded mixed-label groups. It is excluded from all comparisons. The corrected audit only excludes contradictory identical pixels. Runs ending `_v1` use the corrected data.

## Field-data experiment B

Adding field training was specified during experiment A using development-validation limitations, before any test predictions. The chosen experiment-A recipe is retrained from random initialization on five sources, adding Tomato-Village. It uses 100-image batches, 10 draws per source × class bucket, 40 maximum epochs and 64 batches per epoch. It has its own validation-based checkpoint selection. This is a staged experiment, not an exhaustive hyperparameter search.

Tomato-Village filename stems identify original photographs shared by multiple crops. These families, exact duplicates and structurally confirmed near matches are kept together. Entire families spanning publisher folders are promoted to test > validation > train. This yields 1,379 training, 1,382 validation and 1,764 test images. The test has only 22 Healthy images. All original field photographs and dataset labels are retained in manifests; no photos are removed based on predictions.

The frozen experiment-A model is evaluated on the entire 4,525-image Tomato-Village source, unseen by that model's training and selection. Experiment B is evaluated only on its held-out 1,764-image subset; its score cannot be compared directly with experiment A's full-source score. Experiment A can additionally be scored on the same subset for a matched comparison. Experiment B is selected from its validation results; internet test performance never chooses the delivered model.

## Additional annotated-source continuation C

An additional Tomato-Disease release was investigated during experiment A and downloaded while B trained. Its archive checksum was verified. It contains 2,212 photos and 6,581 annotations; many online photos overlap earlier sources. Obvious filenames naming other crops are excluded. No new annotations or regions are labelled from model predictions.

The archive lacks a numerical class-name YAML. The binary Healthy ID is inferred as the unique class with 736 annotations, matching Healthy in the releasing authors' [Table 2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334989). The eight other annotation totals also correspond to the paper's eight damaged categories, with four fewer Leaf Mold annotations in the archive than the paper. The recovered Healthy ID is 1; class-wise training examples were visually reviewed before training. This is a documented inference from the authors' annotation counts, rather than an official supplied numerical mapping or an independent diagnosis.

Extract fixed annotation regions with 10% context margin; skip boxes with an original edge below 32 pixels. Canonical pre-Roboflow photo names, decoded image/crop duplicates, and structurally confirmed perceptual matches define groups. Source-photo/crop overlap with all earlier sources excludes the whole new group, including matches to older tests. The result is 2,630 training, 517 validation and 488 test regions. The crop test measures classification with supplied annotation regions, not automatic detection in whole field scenes. Missing web, plant and session provenance remains a limitation.

C continues the project's own validation-selected B checkpoint, with all six sources jointly sampled for retention. It does not use an externally pretrained model. Settings specified before B/C test predictions: 25 maximum epochs, 64 steps, batch 96 (eight draws per source × class bucket), learning rate multiplier 0.3, same recipe as selected in A. B and C are compared on identical six-source validation cohorts; the higher validation selection score chooses the delivered checkpoint before B/C test evaluation. Existing web and legacy tests remain exploratory and cannot certify broad internet accuracy.

Seven published GTLD example photos are separately frozen as an illustrative unseen-source check. There is only one Healthy photograph, so this is not counted as a substantial benchmark or random internet test.

## Final natural-scene continuation D

After completing and reporting A/B/C, PlantWild v1 was investigated for additional natural internet scenes. Its official tomato subset contains 1,966 photographs across eight source categories. Only original ZIP members were downloaded, with CRC verification and local SHA-256 records. No CLIP, text embeddings, pretrained features or generated class prompts are used. The original photos are not modified or redistributed; numerical input preprocessing and temporary review previews remain local research artifacts. Dataset licence: CC BY-NC-ND 4.0.

The generic `tomato leaf` category is treated as a candidate apparently Healthy pool, rather than assuming every example is healthy. All 226 candidates were screened visually before any PlantWild predictions. Ninety-six were excluded for visible damage/uncertain health, food/drink, flowers or bare stems, fruit-dominated scenes, insufficient leaf detail or an apparent other plant. Retained references are apparent visual assessments, not biological diagnoses. Published disease-category labels are retained for the other seven classes. The asymmetric reference audit and incompletely verified source labels limit interpretation.

Original source-page URLs, exact decoded duplicates and pHash<=4 + SSIM>=.97 matches define groups. Groups spanning publisher partitions are promoted to the most reserved partition. Any detected photo, crop or source-page overlap with all earlier data, including annotated parent photos and GTLD examples, excludes the entire new group. Contradictory identical-pixel labels are excluded. Final PlantWild counts: 1,004 training (77 Healthy), 205 validation (15 Healthy), 433 test (26 Healthy). Always predicting Unhealthy would score 407/433 = 94.0% accuracy and 50% balanced accuracy on this holdout.

D continues only our own selected C checkpoint, with all seven sources jointly replayed: up to 25 epochs, 64 steps per epoch, batch 112 (eight draws per source × class), same augmentation/attention recipe and LR multiplier 0.3. The starting model and D are compared using identical seven-source validation cohorts. That selection is frozen before any PlantWild holdout inference. Previous benchmark/web tests remain exploratory; the new curated PlantWild holdout is a conditional internet-scene benchmark, not a representative probability sample of the internet. No additional training or threshold choice is made from its final test predictions.

Source: [PlantWild project](https://tqwei05.github.io/PlantWild/), Wei et al., ACM MM 2024; [official archive](https://huggingface.co/datasets/uqtwei2/PlantWild). The release's own readme defines split IDs as 0=test, 1=train, 2=validation; these are preserved before group promotion. Raw source URLs, manual review decisions and exact audited manifests are retained locally.

## Reporting and limits

Report ordinary accuracy, balanced accuracy, macro F1, both class recalls, confusion matrices, category counts and group-bootstrap intervals. The primary working target is at least 90% balanced accuracy on every included source's held-out benchmark; this is a target, not a promised result. Report every failure to meet it. A high pooled score is insufficient.

Legacy benchmark tests were evaluated in prior projects, making comparisons exploratory. Parent-photo, plant, garden, camera and capture-session metadata are incomplete. Perceptual checks cannot prove independence. In particular, TLID contains processed backgrounds; greenhouse collection does not make its released images a natural-background benchmark. Bootstrap intervals do not capture label errors or undetected garden/session dependence.

The 16 web photographs are an old convenience sample with weak reference labels, not a representative random internet test. Their Healthy labels often mean apparently asymptomatic foliage. The grouping audit reserves a matching PlantDoc photo from training. Scores on these images must be called illustrative agreement. They cannot validate arbitrary internet accuracy or diagnosis. Repeatedly testing until those specific photos score highly is prohibited in these experiments.

## Sources

- [PlantVillage original repository](https://github.com/spMohanty/PlantVillage-Dataset): original RGB tomato images; existing audited groups and partitions reused.
- [Bangladesh](https://data.mendeley.com/datasets/rnbsw72zb5/1): Oni and Prama, DOI 10.17632/rnbsw72zb5.1, CC BY 4.0. Only audited label-agreement development/test images are included.
- [TLID/PTLID V2](https://data.mendeley.com/datasets/kt64b2kh89/2): Pellenz et al., DOI 10.17632/kt64b2kh89.2, CC BY 4.0. Whole leaves only; derived patches excluded.
- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset): Singh et al., CC BY 4.0. Tomato classes only, source-labelled cropped internet images.
- [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village): Gehlot, Saxena and Gandhi, DOI 10.1007/s00530-023-01158-y. Multiclass classification variant, Git tree 49da676c27fa3879ef38850a816ec74a5072f52a. Every downloaded file is verified against its Git blob SHA. The repository does not supply an explicit dataset licence in the inspected classification tree; raw images are kept locally rather than bundled for redistribution.
- Web source URLs, attribution and weak-reference notes are retained from the earlier reviewed sample in the combined manifest.
