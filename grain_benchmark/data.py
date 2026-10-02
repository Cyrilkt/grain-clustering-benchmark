"""Read validated arrays without importing any model-training framework."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

REGIMES = ("grain_only", "grain_volcashdb")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _entry(root: str | Path, regime: str) -> tuple[Path, dict, dict]:
    root = Path(root).resolve()
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime!r}; expected {REGIMES}")
    manifest = json.loads((root / "manifest.json").read_text())
    return root, manifest, manifest["regimes"][regime]

def _array(root: Path, entry: dict, name: str, verify: bool) -> np.ndarray:
    info = entry["files"][name]
    # Manifest paths are repository-relative: data/embeddings/...
    rel = Path(info["path"])
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'data':
        raise ValueError(f"Unsafe manifest path: {rel}")
    path = root.joinpath(*rel.parts[1:]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Array path escapes the data directory")
    if verify and sha256_file(path) != info["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}; refusing silent dataset substitution")
    return np.load(path, allow_pickle=False)

def load_features(root: str | Path, regime: str, verify: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """No labels are read by this function."""
    root, manifest, entry = _entry(root, regime)
    x = _array(root, entry, "features", verify)
    ids = _array(root, entry, "row_ids", verify)
    if list(x.shape) != entry["shape"] or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid feature shape or non-finite features")
    if x.dtype.name != entry["dtype"]:
        raise ValueError("Feature dtype does not match the manifest")
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(x):
        raise ValueError("Invalid or duplicate row IDs")
    if np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError("Zero-norm embedding")
    return x, ids, entry

def load_labels(root: str | Path, regime: str, verify: bool = True) -> np.ndarray:
    root, _, entry = _entry(root, regime)
    y = _array(root, entry, "labels", verify)
    if y.shape != (entry["shape"][0],) or y.dtype.kind not in "iu":
        raise ValueError("Invalid evaluation labels")
    values, counts = np.unique(y, return_counts=True)
    if {str(k): int(v) for k, v in zip(values, counts)} != entry["class_counts"]:
        raise ValueError("Class counts differ from the manifest")
    return y
