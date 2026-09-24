"""Read image-run evaluation artifacts without requiring the private RGB images."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .evaluation import align_predictions


def image_dataset_digest(sample_ids, labels):
    samples, y = np.asarray(sample_ids), np.asarray(labels)
    if samples.ndim != 1 or y.shape != samples.shape or samples.dtype.kind not in 'US':
        raise ValueError('Image sample IDs and labels must be aligned one-dimensional arrays')
    if len(np.unique(samples)) != len(samples) or y.dtype.kind not in 'iu':
        raise ValueError('Unique image IDs and integer evaluation labels are required')
    payload = json.dumps({'sample_ids': samples.tolist(), 'labels': y.tolist()},
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def read_image_predictions(path, regime):
    with np.load(Path(path), allow_pickle=False) as data:
        required = {'row_ids','predictions','labels','sample_ids','regime','dataset_sha256'}
        if not required.issubset(data.files):
            raise ValueError(f'{path}: missing image-run fields {required-set(data.files)}')
        if str(data['regime'].item()) != regime:
            raise ValueError(f'{path}: regime mismatch')
        y, ids, samples = data['labels'], data['row_ids'], data['sample_ids']
        if ids.shape != y.shape or not np.array_equal(ids, np.arange(len(y))):
            raise ValueError(f'{path}: invalid sequential image row IDs')
        digest = image_dataset_digest(samples,y)
        if digest != str(data['dataset_sha256'].item()):
            raise ValueError(f'{path}: sample/label integrity mismatch')
        pred = align_predictions(ids,ids,data['predictions'])
        return y, ids, pred, digest
