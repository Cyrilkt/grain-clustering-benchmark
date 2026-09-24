# Prediction artifacts

All NumPy bundles are read with `allow_pickle=False`. A partition must contain one nonnegative integer cluster ID per observation; unassigned rows and duplicate IDs are rejected. Arbitrary numeric cluster IDs are allowed.

## Feature-model predictions

`predictions.npz` contains:

| Key | Meaning |
|---|---|
| `row_ids` | IDs corresponding to the input feature rows |
| `predictions` | One cluster assignment per ID |
| `regime` | Scalar `grain_only` or `grain_volcashdb` string |
| `features_sha256` | SHA256 of the exact input `features.npy` |

The evaluator loads labels from the matching data manifest, validates the embedding digest and aligns predictions by row ID before computing metrics. Predictions from another regime or another feature export are rejected.

## Image-model predictions

Image runs instead include `row_ids`, `sample_ids`, `predictions`, `labels`, `regime`, `dataset_sha256`, `partition_epoch` and `input_kind`.

`sample_ids` are image paths relative to the Grain root, in the loader's deterministic order. `dataset_sha256` covers the ordered sample IDs and evaluation labels; it is an identity/metadata check, not a checksum of RGB image bytes. Reading an image prediction bundle recomputes this digest. Mean matrices require the same identities and labels across all supplied runs.

Use `--input-kind images` for `evaluate` and `structured-mean`. These commands do not need the private images or the published frozen BYOL arrays. Evaluation metadata is included in the prediction bundle only after the partition has been produced.

The partition is the last native pseudo-label update before the final Grain optimization pass. The separate `final/features.npy` is extracted after the last optimizer step. Do not conflate these two readouts.

## Metrics and aggregation

ACC uses one-to-one Hungarian matching on the rectangular contingency matrix. NMI uses arithmetic normalization. ARI operates directly on the partitions. `K_occupied` counts distinct cluster IDs present in the evaluated observations.

`structured_confusion` associates each cluster with its majority reference class, chooses one main cluster per such class and aggregates the remaining clusters in the extra block. It is not a Hungarian alignment. Main and extra blocks jointly normalize every true-class column to 100 percent.

The summary tool groups compatible configurations and requires explicit expected run counts and a standard-deviation convention. Feature-model summaries check the input feature digest; image-model summaries check the dataset-identity digest. Preserve individual run artifacts alongside any aggregate.
