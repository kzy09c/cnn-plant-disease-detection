# Bangladesh: binary results

See ../README.md for setup, ../DATA_SOURCES.md for provenance, and ../RESULTS.md for the combined report.

## Bangladesh

The primary test recognized **52 / 59 Healthy** and **80 / 80 Unhealthy** leaves.

| Metric | Value |
|---|---:|
| Accuracy | 94.96% |
| Balanced accuracy | 94.07% |
| Macro F1 | 94.75% |
| Weighted F1 | 94.91% |
| Unhealthy precision | 91.95% |
| Unhealthy recall | 100.00% |
| Healthy recall | 88.14% |
| Unhealthy F1 | 95.81% |
| Healthy precision | 100.00% |
| Healthy F1 | 93.69% |
| ROC AUC | 1.0000 |
| Average precision | 1.0000 |

![Confusion counts](../figures/bangladesh_confusion_counts.png)

![Row-normalized confusion matrix](../figures/bangladesh_confusion_normalized.png)

![Grouped validation learning curves](../figures/bangladesh_cv_learning_curves.png)

![Final-fit training curves](../figures/bangladesh_final_training_curves.png)

![ROC and precision–recall](../figures/bangladesh_roc_precision_recall.png)

![Per-class precision, recall, F1](../figures/bangladesh_class_metrics.png)

![Group-bootstrap uncertainty](../figures/bangladesh_confidence_intervals.png)

![Actual learned first-layer filters](../figures/bangladesh_first_layer_filters.png)

![Fixed random test examples](../figures/bangladesh_test_examples.jpg)

![Recorded test errors](../figures/bangladesh_errors.jpg)

![Full holdout: uncertain labels](../figures/bangladesh_full_holdout_confusion_counts.png)
