# Model card: custom tomato-leaf CNN

Checkpoint: `best_model.pt`; SHA-256 `e905416007b09d826595af17fa4f5d5b15c103c973790853f707de73a38f1108`.

Selected run: `mixed_attention_wild_v1`. Parameters: 955,194.

All weights originate from our own randomly initialized sequential CNN. Later stages replay earlier training data while continuing only our own checkpoints. No ResNet, ImageNet weights, externally pretrained feature extractor or teacher model was used.

Input: close photograph of tomato foliage, EXIF orientation applied, RGB, bilinear resize to 160×160, normalization x/127.5−1. Output order: Healthy, Unhealthy; decision threshold 0.5. Scores are uncalibrated. CUDA uses float16 inference, CPU float32; small numerical differences are possible.

Unhealthy includes source-labelled disease, insects, nutrient deficiency and other visible damage. The model does not diagnose a particular disease, verify tomato species, locate leaves automatically or reliably reject unfamiliar inputs.

| Held-out check | N | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Bangladesh | 139 | 87.77% | 86.04% | 86.99% |
| Internet16 | 16 | 81.25% | 81.25% | 81.18% |
| PlantDoc | 69 | 86.96% | 78.33% | 74.73% |
| PlantVillage | 3632 | 94.47% | 96.97% | 86.43% |
| PlantWild | 433 | 90.99% | 88.01% | 74.02% |
| TLID | 3050 | 92.59% | 94.48% | 88.78% |
| TomatoDisease_ROI | 488 | 93.85% | 93.62% | 85.10% |
| TomatoVillage | 1764 | 95.98% | 97.96% | 68.09% |
| GTLD_examples | 7 | 85.71% | 91.67% | 78.79% |

PlantWild is a newly evaluated curated internet-scene holdout, with manually screened apparently Healthy references and source-category disease references. It is not a probability sample of the internet or independent biological diagnosis. Older datasets and the 16 web photos have been inspected in earlier experiments; their comparisons are exploratory. The annotated-region test assumes supplied bounding boxes; GTLD has only seven examples. See the full local RESULTS.md and EXPERIMENT_PROTOCOL.md for source attribution, grouping, uncertainty and failures.

Training sources: PlantVillage, Bangladesh (Oni/Prama), TLID (Pellenz et al.), PlantDoc (Singh et al.), Tomato-Village (Gehlot et al.), Tomato-Disease (Chen), and PlantWild (Wei et al.). Original photos are not included in this bundle.
