"""Final-epoch export: no class-based model selection, no implicit overwrites."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np


def write_final_artifacts(output_dir, features, labels=None, *, epoch, row_ids=None, metadata=None):
    """Write arrays atomically. Labels are metadata only and never change features.

    The caller must extract from the state AFTER the last optimizer step. This
    writer does not authenticate provenance of older/pre-existing embeddings.
    """
    x = np.asarray(features)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or x.dtype.kind != 'f':
        raise ValueError('Features must be a nonempty finite floating-point matrix')
    if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 1:
        raise ValueError('Final epoch must be a positive integer')
    ids = np.arange(len(x), dtype=np.int64) if row_ids is None else np.asarray(row_ids)
    if ids.shape != (len(x),) or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(x):
        raise ValueError('Unique non-object row IDs are required')
    y = None if labels is None else np.asarray(labels)
    if y is not None and (y.shape != (len(x),) or y.dtype.kind not in 'iuUS'):
        raise ValueError('Invalid optional evaluation labels')
    meta = dict(metadata or {})
    reserved = {'selection', 'epoch', 'shape', 'dtype', 'files'}
    if reserved.intersection(meta):
        raise ValueError('Metadata cannot override final-export provenance')
    # Ensure metadata is serializable before touching any output.
    json.dumps(meta, allow_nan=False)
    dest = Path(output_dir).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.final-export-', dir=dest.parent))
    try:
        arrays = {'features': x, 'row_ids': ids}
        if y is not None:
            arrays['labels'] = y
        files = {}
        for name, array in arrays.items():
            path = temp / f'{name}.npy'
            np.save(path, array, allow_pickle=False)
            files[name] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        meta.update(selection='final_epoch_after_last_optimizer_step', epoch=int(epoch),
                    shape=list(x.shape), dtype=x.dtype.name, files=files)
        (temp/'provenance.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        if dest.exists():
            raise FileExistsError(dest)
        os.rename(temp, dest)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return meta
