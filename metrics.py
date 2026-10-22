"""Binary evaluation, group bootstrap, and conservative image-similarity helpers."""
import numpy as np
import torch
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from model import CLASSES, prepare_batch

POPCOUNT = np.array([n.bit_count() for n in range(256)], dtype=np.uint8)

def image_hash(image):
    array = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    coefficients = dctn(array, norm='ortho')[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return np.packbits(bits)

def distances(a, b):
    return POPCOUNT[a[:, None, :] ^ b[None, :, :]].sum(axis=2)

def structural_similarity(a, b):
    """Mean SSIM over non-overlapping 8x8 blocks of 64x64 grayscale thumbnails."""
    x = a.astype(np.float32).reshape(8, 8, 8, 8).swapaxes(1, 2).reshape(64, 64) / 255
    y = b.astype(np.float32).reshape(8, 8, 8, 8).swapaxes(1, 2).reshape(64, 64) / 255
    mx, my = (x.mean(1), y.mean(1))
    vx, vy = (x.var(1), y.var(1))
    covariance = ((x - mx[:, None]) * (y - my[:, None])).mean(1)
    ssim = (2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) / ((mx ** 2 + my ** 2 + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
    return float(ssim.mean())

@torch.inference_mode()
def predict(model, images, ids, batch_size):
    model.eval()
    parts = []
    for start in range(0, len(ids), batch_size):
        x = prepare_batch(images, ids[start:start + batch_size], False)
        with torch.autocast('cuda', dtype=torch.float16):
            parts.append(model(x).float().softmax(1).cpu().numpy())
    return np.concatenate(parts)

def evaluate(true, probabilities):
    predicted = probabilities.argmax(1)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(true, predicted)), 'balanced_accuracy': float(recall_score(true, predicted, average='macro', zero_division=0)), 'precision_unhealthy': float(precision_score(true, predicted, zero_division=0)), 'recall_sensitivity_unhealthy': float(recall_score(true, predicted, zero_division=0)), 'specificity_healthy': float(tn / (tn + fp)), 'f1_unhealthy': float(f1_score(true, predicted, zero_division=0)), 'macro_precision': float(precision_score(true, predicted, average='macro', zero_division=0)), 'macro_recall': float(recall_score(true, predicted, average='macro', zero_division=0)), 'macro_f1': float(f1_score(true, predicted, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(true, predicted, average='weighted', zero_division=0)), 'roc_auc': float(roc_auc_score(true, probabilities[:, 1])), 'average_precision': float(average_precision_score(true, probabilities[:, 1])), 'log_loss': float(log_loss(true, probabilities, labels=[0, 1])), 'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]], 'confusion_labels': CLASSES, 'per_class': classification_report(true, predicted, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0)}

def bootstrap(true, predicted, groups):
    matrices = np.stack([confusion_matrix(true[groups == g], predicted[groups == g], labels=[0, 1]) for g in np.unique(groups)])
    rng = np.random.default_rng(2026)
    results = []
    for _ in range(2000):
        matrix = matrices[rng.integers(len(matrices), size=len(matrices))].sum(axis=0)
        actual, predictions = (matrix.sum(axis=1), matrix.sum(axis=0))
        if np.any(actual == 0):
            continue
        tp = matrix.diagonal()
        results.append([matrix.trace() / matrix.sum(), (tp / actual).mean(), (2 * tp / (actual + predictions)).mean(), tp[1] / actual[1], tp[0] / actual[0]])
    bounds = np.quantile(results, [0.025, 0.975], axis=0)
    return {'method': '2000 percentile bootstrap draws of inferred source/similarity groups', 'valid_samples': len(results), **{name + '_95': bounds[:, i].tolist() for i, name in enumerate(['accuracy', 'balanced_accuracy', 'macro_f1', 'recall_unhealthy', 'specificity_healthy'])}}
