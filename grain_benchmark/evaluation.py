"""Common external evaluation, independent of the native training implementations."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples

def _labels(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1 or not x.size or x.dtype.kind not in "iuUSb":
        raise ValueError(f"{name} must be a nonempty one-dimensional integer/string label array")
    if x.dtype.kind in "iu" and np.any(x < 0):
        raise ValueError(f"{name} contains negative labels; unassigned/noise rows require an explicit policy")
    return x

def contingency(y_true: np.ndarray, y_pred: np.ndarray):
    y, z = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if y.shape != z.shape:
        raise ValueError("Prediction/label lengths differ")
    classes, yi = np.unique(y, return_inverse=True)
    clusters, zi = np.unique(z, return_inverse=True)
    table = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(table, (zi, yi), 1)
    return table, classes, clusters

def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    table, classes, clusters = contingency(y_true, y_pred)
    r, c = linear_sum_assignment(-table)
    return {
        "ACC": float(table[r, c].sum() / table.sum()),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "K_occupied": int(len(clusters)),
        "N": int(table.sum()),
        "C_reference": int(len(classes)),
        "acc_policy": "rectangular_one_to_one_hungarian",
        "nmi_average_method": "arithmetic",
    }

def structured_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Supplement S4: 2C x C, percent normalization over each complete column.

    Majority ties use sorted class order. Main-cluster ties use lexicographic
    full count vectors so a mere relabeling of cluster IDs cannot change the matrix.
    These tie conventions make unspecified corner cases explicit; they have not
    been verified against the original unpublished figure script.
    This function NEVER uses Hungarian alignment.
    """
    table, classes, _ = contingency(y_true, y_pred)
    c = len(classes)
    out = np.zeros((2*c, c), dtype=np.float64)
    majority = table.argmax(axis=1)
    for j in range(c):
        candidates = np.flatnonzero(majority == j)
        if not len(candidates):
            continue
        main = max(candidates, key=lambda g: (int(table[g, j]), tuple(table[g].tolist())))
        out[j] = table[main]
        out[c+j] = table[candidates].sum(axis=0) - table[main]
    out *= 100.0 / table.sum(axis=0)[None, :]
    if not np.allclose(out.sum(axis=0), 100.0):
        raise RuntimeError("Structured matrix lost probability mass")
    return out, classes

def macro_silhouette(x: np.ndarray, y_true: np.ndarray) -> dict:
    """All rows; cosine distance in original feature space; equal class weighting."""
    y = _labels(y_true, "y_true")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid features or sample correspondence")
    classes = np.unique(y)
    if not 2 <= len(classes) < len(x):
        raise ValueError("Silhouette requires 2 <= number of classes < number of samples")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine silhouette is undefined for a zero vector in this evaluator")
    x = x / norms  # evaluation copy only; stored vectors are never changed
    s = silhouette_samples(x, y, metric="cosine")
    values = {str(c): float(s[y == c].mean()) for c in classes}
    return {"macro_silhouette": float(np.mean(list(values.values()))),
            "micro_silhouette": float(s.mean()), "per_class": values,
            "N": len(y), "metric": "cosine", "space": "original_features",
            "averaging": "equal_class_weight", "self_distance": "excluded",
            "singleton_convention": "zero", "dtype": "float64"}

def align_predictions(expected_ids: np.ndarray, supplied_ids: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    expected, supplied = np.asarray(expected_ids), np.asarray(supplied_ids)
    pred = _labels(predictions, "predictions")
    if expected.ndim != 1 or supplied.ndim != 1 or len(supplied) != len(pred):
        raise ValueError("Invalid row-ID dimensions")
    if len(np.unique(expected)) != len(expected) or len(np.unique(supplied)) != len(supplied):
        raise ValueError("Duplicate row IDs")
    if expected.dtype.kind != supplied.dtype.kind:
        raise ValueError("Row-ID types differ")
    if len(expected) != len(supplied) or set(expected.tolist()) != set(supplied.tolist()):
        raise ValueError("Predictions must cover the complete collection with matching row IDs")
    pos = {v: i for i, v in enumerate(supplied.tolist())}
    return pred[np.asarray([pos[v] for v in expected.tolist()])]
