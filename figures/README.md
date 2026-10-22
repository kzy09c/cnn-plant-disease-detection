# Reusable figures and suggested captions

PNG files are rendered at 200 dpi; matching SVG files retain editable text. JPG galleries use the dataset's source images; retain dataset attribution when reusing them. All curves/counts use actual saved results, without new training or test inference.

| Figure | Suggested caption / scope |
|---|---|
| cnn_architecture | Shared binary CNN architecture and tensor dimensions; 939,874 parameters. |
| experiment_workflow | Development-only grouped validation, final fitting, and reserved-test evaluation. |
| dataset_counts | Healthy/Unhealthy image counts in each experiment; Bangladesh full holdout contains uncertain labels. |
| bangladesh_label_audit | Predetermined label-consistency filtering and released-split counts; agreement is not an expert diagnosis. |
| *_confusion_counts | Primary-test confusion counts; rows are reference classes, columns predictions. |
| *_confusion_normalized | Confusion matrix normalized by reference-class support, highlighting each class's recall. |
| *_cv_learning_curves | Mean ± SD across five grouped development folds; SD describes fold variation, not a confidence interval. |
| *_final_training_curves | Final-fit training loss and augmented training accuracy; not held-out performance. |
| *_roc_precision_recall | ROC and precision–recall curves with Unhealthy as the positive class. |
| *_class_metrics | Primary-test precision, recall, and F1 for each class. |
| *_confidence_intervals | 95% percentile group-bootstrap intervals; source/label uncertainty is not fully captured. |
| *_first_layer_filters | Actual 3×3 RGB first-convolution filters, independently normalized; not explanations of individual predictions. |
| *_test_examples.jpg | Fixed random examples selected with seed 42, not selected for correctness. |
| *_errors.jpg | First up to six recorded primary-test errors, illustrating failures rather than typical performance. |
| bangladesh_full_holdout_confusion_counts | Full 160-image publisher holdout against filename labels; 21 labels conflict with annotations. |

Reproduce figures with python make_report.py. Most figures require only the tracked results and checkpoints. Error galleries can be regenerated when source images are downloaded; existing galleries are preserved otherwise.
