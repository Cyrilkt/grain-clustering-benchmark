from pathlib import Path
import numpy as np
from .cli import write_json, provenance
from .data import load_labels
from .evaluation import clustering_metrics

def save_feature_result(out, pred, ids, entry, *, data_root, regime, method, config, fit_seconds):
    out=Path(out)
    pred=np.asarray(pred,dtype=np.int64)
    if pred.shape != (len(ids),) or np.any(pred<0):
        raise ValueError('Final partition must cover every observation; no unassigned rows')
    digest=entry['files']['features']['sha256']
    np.savez_compressed(out/'predictions.npz',predictions=pred,row_ids=ids,
        regime=np.asarray(regime),features_sha256=np.asarray(digest))
    # Reference labels are read here, after all optimization and assignments.
    metrics=clustering_metrics(load_labels(data_root,regime),pred)
    metrics.update(method=method,regime=regime,seed=int(config['seed']),
        experiment_status=config.get('status','new_run'),fit_seconds=float(fit_seconds),
        features_sha256=digest,selection='final_state',config=config)
    write_json(out/'metrics.json',metrics);write_json(out/'provenance.json',provenance())
    return metrics
