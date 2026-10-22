# PlantVillage: binary results

See ../README.md for setup, ../DATA_SOURCES.md for provenance, and ../RESULTS.md for the combined report.

## PlantVillage

The primary test recognized **318 / 318 Healthy** and **3309 / 3314 Unhealthy** leaves.

| Metric | Value |
|---|---:|
| Accuracy | 99.86% |
| Balanced accuracy | 99.92% |
| Macro F1 | 99.57% |
| Weighted F1 | 99.86% |
| Unhealthy precision | 100.00% |
| Unhealthy recall | 99.85% |
| Healthy recall | 100.00% |
| Unhealthy F1 | 99.92% |
| Healthy precision | 98.45% |
| Healthy F1 | 99.22% |
| ROC AUC | 0.9999 |
| Average precision | 1.0000 |

![Confusion counts](../figures/plantvillage_confusion_counts.png)

![Row-normalized confusion matrix](../figures/plantvillage_confusion_normalized.png)

![Grouped validation learning curves](../figures/plantvillage_cv_learning_curves.png)

![Final-fit training curves](../figures/plantvillage_final_training_curves.png)

![ROC and precision–recall](../figures/plantvillage_roc_precision_recall.png)

![Per-class precision, recall, F1](../figures/plantvillage_class_metrics.png)

![Group-bootstrap uncertainty](../figures/plantvillage_confidence_intervals.png)

![Actual learned first-layer filters](../figures/plantvillage_first_layer_filters.png)

![Fixed random test examples](../figures/plantvillage_test_examples.jpg)

![Recorded test errors](../figures/plantvillage_errors.jpg)
