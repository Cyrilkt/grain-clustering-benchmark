# Grain representations and evaluation metadata

The two regime directories contain the frozen BYOL arrays used as inputs to the feature-level clustering pipelines. Each `features.npy` has shape `(5240, 256)`, dtype `float32` and approximately unit L2 norms. The exporter did not alter the values, normalize them again or reorder observations.

`labels.npy` contains 21 integer reference labels and is evaluation metadata. `row_ids.npy` contains positions in the corresponding feature matrix. IDs are local to a regime; without an original filename manifest they do not certify cross-regime image correspondence.

Both directories represent Grain only. No VolcAshDB observation is appended to the target partition. The collection is clustered as a whole; there is no independent train/test split. Older input archives contained duplicate train/test tensors, not two independent image collections.

`manifest.json` records checksums, shapes and class counts. Run `python -m grain_benchmark validate --data-root data` from the repository root after extracting the code-and-data archive or companion embeddings archive. Validation refuses silent replacement of a file whose checksum differs.

Historical BYOL seed/checkpoint-selection information was not supplied with these arrays. New fixed-final-epoch exports do not retroactively establish the provenance of those earlier representations.

Grain RGB images are not included. Official VolcAshDB download links and the RGB layout for image-training users are documented in the root README. Original images, generated checkpoints and local output directories must not be added to source version control.
