from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .data import REGIMES, load_features, load_labels
from .evaluation import clustering_metrics, structured_confusion, macro_silhouette, align_predictions

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")

def provenance() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
        result[name] = metadata.version(name)
    return result

def main():
    parser = argparse.ArgumentParser(description="Grain clustering and evaluation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="Validate both exact published input arrays")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("silhouette", help="Recompute the label-based macro cosine silhouette")
    p.add_argument("--data-root", default="data")
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("kmeans", help="User-supplied PyTorchKMeans on frozen embeddings")
    p.add_argument("--data-root", default="data")
    p.add_argument("--config", type=Path)
    p.add_argument("--regime", choices=REGIMES)
    p.add_argument("--k-values", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--n-init", type=int, help="Restarts PER RANK; total = n_init * world_size")
    p.add_argument("--metric", choices=("cosine", "euclidean"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--max-iter", type=int)
    p.add_argument("--tol", type=float)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate", help="Evaluate one prediction file with explicit row IDs")
    p.add_argument("--data-root", default="data")
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--input-kind", choices=("features", "images"), default="features")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("structured-mean", help="Average S4 matrices from full-collection prediction files")
    p.add_argument("--data-root", default="data")
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--predictions", nargs="+", type=Path, required=True)
    p.add_argument("--input-kind", choices=("features", "images"), default="features")
    p.add_argument("--expected-runs", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "kmeans":
        from .config import method_settings
        args.regime, settings = method_settings("kmeans", args.config, args.regime)
        for name in ("n_init", "metric", "max_iter", "tol"):
            if getattr(args, name) is None:
                setattr(args, name, settings[name])
    if args.command == "validate":
        result = {"regimes": {}}
        for regime in REGIMES:
            x, ids, entry = load_features(args.data_root, regime)
            y = load_labels(args.data_root, regime)
            result["regimes"][regime] = {
                "shape": list(x.shape), "classes": len(np.unique(y)), "unique_row_ids": len(np.unique(ids)),
                "features_sha256": entry["files"]["features"]["sha256"], "status": "PASS"}
        print(json.dumps(result, indent=2))
        if args.out: write_json(args.out, result)
        return
    image_input = getattr(args, "input_kind", "features") == "images"
    if image_input:
        from .predictions import read_image_predictions
        first = args.predictions[0] if isinstance(args.predictions, list) else args.predictions
        y, ids, _, dataset_digest = read_image_predictions(first, args.regime)
    else:
        x, ids, entry = load_features(args.data_root, args.regime)
    if args.command == "silhouette":
        y = load_labels(args.data_root, args.regime)
        with threadpool_limits(limits=1):
            result = macro_silhouette(x, y)
        result.update(regime=args.regime, provenance=provenance(), features_sha256=entry['files']['features']['sha256'])
        write_json(args.out, result)
        print(json.dumps(result, indent=2)); return
    if args.command == "kmeans":
        from .torch_kmeans_runner import run
        run(args, x, ids, entry)
        return
    if not image_input:
        y = load_labels(args.data_root, args.regime)
    def read_predictions(path: Path):
        if image_input:
            yp, ip, pred, digest = read_image_predictions(path, args.regime)
            if digest != dataset_digest or not np.array_equal(yp, y) or not np.array_equal(ip, ids):
                raise ValueError(f"{path}: image samples or evaluation labels differ between runs")
            return pred
        with np.load(path, allow_pickle=False) as bundle:
            needed = {"row_ids", "predictions", "regime", "features_sha256"}
            if not needed.issubset(bundle.files):
                raise ValueError(f"{path}: missing {needed-set(bundle.files)}")
            if str(bundle["regime"].item()) != args.regime:
                raise ValueError(f"{path}: regime mismatch")
            if str(bundle["features_sha256"].item()) != entry['files']['features']['sha256']:
                raise ValueError(f"{path}: embedding provenance mismatch")
            return align_predictions(ids, bundle["row_ids"], bundle["predictions"])
    if args.command == "evaluate":
        pred = read_predictions(args.predictions)
        result = clustering_metrics(y, pred)
        result.update(regime=args.regime, input_kind=args.input_kind, input=str(args.predictions), provenance=provenance())
        write_json(args.out, result); print(json.dumps(result, indent=2)); return
    if args.command == "structured-mean":
        if args.expected_runs < 1 or len(args.predictions) != args.expected_runs:
            parser.error("Prediction count differs from --expected-runs")
        if len(set(p.resolve() for p in args.predictions)) != len(args.predictions):
            parser.error("Same prediction file listed more than once")
        if args.out.exists():raise FileExistsError(args.out)
        matrices=[]
        for p in args.predictions:
            matrix, classes = structured_confusion(y, read_predictions(p));matrices.append(matrix)
        args.out.mkdir(parents=True)
        np.savez_compressed(args.out/"structured_confusion.npz", mean=np.mean(matrices,axis=0),
                            per_run=np.stack(matrices), class_labels=classes)
        write_json(args.out/"metadata.json", dict(regime=args.regime, input_kind=args.input_kind, runs=len(matrices),
                    inputs=[str(p) for p in args.predictions], normalization="full_column_100_percent", alignment="majority_main_extra_not_Hungarian"))
        print(args.out)

if __name__ == "__main__":
    main()
