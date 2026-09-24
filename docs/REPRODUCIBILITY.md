# Grain Clustering Benchmark

Code and frozen representations for **Practical Nonparametric Deep Clustering for Grain Microscopy: A 21-Class Benchmark and the Benefit of VolcAshDB Pretraining**.

The benchmark compares DIVA, DeepDPM, DDPM and AutoProPos on a collection of **5,240 individual mineral-grain images and 21 reference classes**. KMeans and fixed-K ProPos provide complementary baselines. The two representation-learning regimes are **Grain-only** and **Grain + VolcAshDB**.

**The original Grain RGB images are not distributed with this study.** The code-and-data bundle includes the two frozen BYOL feature sets and their evaluation labels. DIVA, DeepDPM, DDPM and KMeans can therefore be run without the original images. Training BYOL, ProPos or AutoProPos requires access to Grain RGB images; those methods do not train from the published embeddings.

## Contents

- [Installation](#installation)
- [Data](#data)
- [Two configurations](#two-configurations)
- [Run KMeans](#run-kmeans)
- [Run DIVA](#run-diva)
- [Run DeepDPM](#run-deepdpm)
- [Run DDPM](#run-ddpm)
- [Image training: BYOL, ProPos and AutoProPos](#image-training-byol-propos-and-autopropos)
- [Evaluation and multiple runs](#evaluation-and-multiple-runs)
- [Outputs and implementation details](#outputs-and-implementation-details)
- [Tests](#tests)
- [References and acknowledgements](#references-and-acknowledgements)

Run every command below **from the repository root**. Examples use Bash on Linux. Seeds in the commands are explicit examples for new runs, not a claim about the identifiers of historical paper runs. Existing run directories are not overwritten.

## Installation

Use a separate environment for each training implementation. DIVA and DeepDPM retain different Lightning 1.x interfaces; installing a single recent Lightning version for both is not supported.

For evaluation and CPU KMeans, start with Python 3.10:

```bash
python3.10 -m venv .venv-kmeans
source .venv-kmeans/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environments/requirements-kmeans.txt
```

For evaluation only, install `requirements-evaluation.txt`; PyTorch is not required to recompute metrics from prediction files.

| Pipeline | Python | Requirements file |
|---|---|---|
| Evaluation / KMeans | 3.10 | `requirements-evaluation.txt` / `environments/requirements-kmeans.txt` |
| DIVA | 3.10 | `environments/requirements-diva.txt` |
| DeepDPM | 3.9 | `environments/requirements-deepdpm.txt` |
| DDPM | 3.10 | `environments/requirements-ddpm.txt` |
| BYOL / ProPos / AutoProPos | 3.10 | `environments/requirements-image.txt` |

For example, create the two feature-model environments separately:

```bash
python3.10 -m venv .venv-diva
source .venv-diva/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environments/requirements-diva.txt

deactivate
python3.9 -m venv .venv-deepdpm
source .venv-deepdpm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environments/requirements-deepdpm.txt
```

For DDPM or image training, repeat this pattern with Python 3.10 and the corresponding requirements file. Check CUDA availability inside the selected environment:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The dependency sets target the interfaces retained by each implementation; they are not complete exports of the original training machines. See [environment notes](environments/README.md) for compatibility and validation scope.

## Data

### Published Grain embeddings

The code-and-data archive contains:

```text
data/
├── manifest.json
└── embeddings/
    ├── grain_only/
    │   ├── features.npy       # float32, shape (5240, 256)
    │   ├── labels.npy         # evaluation labels, shape (5240,)
    │   └── row_ids.npy        # unique row IDs, shape (5240,)
    └── grain_volcashdb/
        ├── features.npy
        ├── labels.npy
        └── row_ids.npy
```

For a source-only checkout, extract the companion `Grain_BYOL_Embeddings_v1.zip` at the repository root. Both bundles use the same layout. Binary embeddings are excluded from source version control by `.gitignore` and can be distributed as a companion release asset.

```bash
# Needed only when data/embeddings/ is not already present:
unzip Grain_BYOL_Embeddings_v1.zip -d .

# Check shapes, row IDs, class counts and SHA256 checksums:
python -m grain_benchmark validate --data-root data
```

Each regime contains **only Grain observations**. The enriched feature set is not a concatenation of Grain and VolcAshDB embeddings. Features and labels retain their original values and row order. No train/test split is introduced: clustering and evaluation use the complete collection once. Do not concatenate the duplicate train/test tensors from older exports.

Reference labels are loaded only after fitting by the feature-model evaluation adapters. They are not passed to DIVA, DeepDPM, DDPM or KMeans for training, initialization or cluster-count selection.

The numerical row IDs identify positions within each regime. They are not original image filenames and do not establish cross-regime image correspondence. The historical BYOL checkpoint-selection metadata is not available in the feature files; the new image-training exporter described below applies to newly trained models only. See [data notes](data/README.md).

### VolcAshDB

VolcAshDB is the auxiliary image source for the enriched image-training regime:

- **Particle images:** [Figshare image dataset](https://figshare.com/articles/dataset/Particle_images/28284980).
- **Versioned collection:** [VolcAshDB, version 2](https://doi.org/10.6084/m9.figshare.c.7644656.v2).
- **Project portal:** [volcashdb.ipgp.fr](https://volcashdb.ipgp.fr/).
- **Dataset paper:** [Benet et al., Scientific Data, 2025](https://doi.org/10.1038/s41597-025-04942-9).

Download the particle images from the official record and retain the dataset's attribution and license information. Do not use the mineral/particle labels or the feature spreadsheet as supervision. VolcAshDB contributes only to self-supervised representation learning; it is excluded from the Grain partition, prototype-scattering objective and reported Grain evaluation.

### RGB layout for image training

Users with access to Grain images should organize them for `ImageFolder`:

```text
/path/to/grain_rgb/
├── mineral_a/
│   ├── grain_001.png
│   └── ...
└── mineral_b/
    └── ...

/path/to/volcashdb_rgb/
└── particles/
    ├── particle_001.png
    └── ...
```

Existing VolcAshDB subdirectories may be retained. When its images are in a flat directory, place them under one child directory such as `particles/` so `ImageFolder` can enumerate them. No semantic subfolders are needed for the auxiliary collection.

Grain folder names supply evaluation metadata only. Image losses and the supervisor do not use those class labels. The image loader reads RGB images and checks image files for validity; review warnings about skipped corrupt files before launching a benchmark run. Paths are supplied on the command line, never hard-coded in YAML.

## Two configurations

There are exactly two YAML files:

```text
configs/grain_only.yaml
configs/grain_volcashdb.yaml
```

Each contains the sections `diva`, `deepdpm`, `ddpm`, `kmeans`, `image`, `byol`, `propos` and `autopropos`. Image methods combine the shared `image` section with the section selected by `--method`. There are no dataset-specific MNIST, CIFAR, STL or alternative Grain configuration files.

| Method | Main settings |
|---|---|
| DIVA | 150 epochs, batch 64, 10-D latent space; `sF=0.005` / `0.5` by regime |
| DeepDPM | 150 AE epochs, then 500 clustering epochs; batch 128, 20-D latent space, initial K=3 |
| DDPM | 400 / 300 epochs; batch 128, six NICE coupling layers, 512 hidden units |
| KMeans | Native `PyTorchKMeans`, cosine distance, 10 initializations per process, 300 maximum iterations |
| BYOL / ProPos / AutoProPos | ResNet-50, 256-D features, 1,000 epochs, 64 images per GPU, four GPUs |
| AutoProPos | Initial K=200; supervisor at epochs 200, 600 and 800; candidate K in `[2, 100)` |

Change the method section of the appropriate regime file to define a new experiment. Command-line overrides, where provided, take precedence and are saved with the run. Changing settings creates a different experiment; do not mix such runs in a single reported mean.

## Run KMeans

Activate the KMeans environment. The standalone baseline and the image-training assignments use the same local `torch_clustering.PyTorchKMeans` implementation.

```bash
source .venv-kmeans/bin/activate

for regime in grain_only grain_volcashdb; do
  python -m grain_benchmark kmeans \
    --config "configs/${regime}.yaml" \
    --data-root data \
    --k-values 21 \
    --seeds 0 1 2 3 4 \
    --device cpu \
    --out outputs/kmeans
done
```

Here, K=21 is an explicit fixed-K baseline, not an input to the nonparametric models. To study sensitivity, supply the desired grid, for example `--k-values 2 5 10 15 20 21 25 30 35 40`. This is an example grid, not an assertion that those were the exact historical Figure 3 evaluation points.

A four-GPU invocation using the same distributed backend as image training is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 -m grain_benchmark kmeans \
  --config configs/grain_only.yaml --data-root data \
  --k-values 21 --seeds 0 --device cuda --distributed \
  --out outputs/kmeans_4gpu
```

**`n_init` is per process.** Ten starts on one process and ten starts on four processes mean 10 and 40 total initializations, respectively. Outputs record both values. Preserve the process count and total restart budget when comparing results. The native mixture methods retain their own internal initializers; these are not replaced by the standalone baseline.

## Run DIVA

Activate the DIVA environment. Both commands use the same feature interface and the common final evaluator.

```bash
source .venv-diva/bin/activate

for regime in grain_only grain_volcashdb; do
  CUDA_VISIBLE_DEVICES=0 python methods/diva/train.py \
    --config "configs/${regime}.yaml" --data-root data \
    --seed 0 --device cuda \
    --out "outputs/diva/${regime}/seed0"
done
```

DIVA trains without reference labels. Final inference covers **all 5,240 rows**, including the incomplete final batch. There is no native many-to-one ACC, CAA or second competing score implementation. The common Hungarian evaluator is the only reported ACC. The internal `bnpy` evidence/loss diagnostics are optimization quantities, not external clustering scores.

## Run DeepDPM

Activate the separate DeepDPM environment:

```bash
source .venv-deepdpm/bin/activate

for regime in grain_only grain_volcashdb; do
  CUDA_VISIBLE_DEVICES=0 python methods/deepdpm/train.py \
    --config "configs/${regime}.yaml" --data-root data \
    --seed 0 --device cuda \
    --out "outputs/deepdpm/${regime}/seed0"
done
```

The launcher performs the 150-epoch autoencoder stage and one 500-epoch split/merge clustering stage. It does not continue into additional AE/clustering alternations. Native model defaults are in `methods/deepdpm/defaults.py`; the experiment-level overrides are in the two YAML files. Reference labels are not passed to either training stage.

## Run DDPM

Activate the DDPM environment:

```bash
source .venv-ddpm/bin/activate

for regime in grain_only grain_volcashdb; do
  CUDA_VISIBLE_DEVICES=0 python methods/ddpm/train.py \
    --config "configs/${regime}.yaml" --data-root data \
    --seed 0 --device cuda \
    --out "outputs/ddpm/${regime}/seed0"
done
```

DDPM uses the native Grain NICE/Dirichlet-mixture path, including its global scalar standardization of the input matrix. The terminal mixture assignments are evaluated without an extra label-guided selection or post-hoc reassignment stage.

Feature-model launchers also accept `--device cpu`. This can help with debugging but does not imply practical full-training runtimes comparable with a GPU.

## Image training: BYOL, ProPos and AutoProPos

These commands require the RGB layout described above. **Downloading the published embeddings does not satisfy that requirement.**

```bash
source .venv-image/bin/activate
export GRAIN_ROOT=/path/to/grain_rgb
export VOLCASHDB_ROOT=/path/to/volcashdb_rgb
```

All three methods use `methods/image_training/main.py`, with an explicit `--method`. The commands below use four GPUs, 64 images per GPU and a global batch of 256. To use a different device count, change both `--nproc_per_node` and `--num-devices`; that also changes the effective batch size and is a different training configuration.

### BYOL

```bash
# Grain-only
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method byol --config configs/grain_only.yaml \
  --grain-root "$GRAIN_ROOT" --num-devices 4 --seed 0 \
  --output-root outputs/byol --run-name grain_only_seed0

# Grain + VolcAshDB
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method byol --config configs/grain_volcashdb.yaml \
  --grain-root "$GRAIN_ROOT" --volcashdb-root "$VOLCASHDB_ROOT" \
  --num-devices 4 --seed 0 \
  --output-root outputs/byol --run-name grain_volcashdb_seed0
```

BYOL disables prototype scattering and the supervisor. Embeddings are extracted afresh **after the last optimizer step at the fixed final epoch**. There is no best-ACC model selection or export option. Evaluation labels may be recorded as metadata but cannot change the exported representation.

### ProPos: fixed K

```bash
# Grain-only, fixed K=21
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method propos --config configs/grain_only.yaml \
  --grain-root "$GRAIN_ROOT" --initial-k 21 --num-devices 4 --seed 0 \
  --output-root outputs/propos --run-name grain_only_k21_seed0

# Grain + VolcAshDB, fixed K=21
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method propos --config configs/grain_volcashdb.yaml \
  --grain-root "$GRAIN_ROOT" --volcashdb-root "$VOLCASHDB_ROOT" \
  --initial-k 21 --num-devices 4 --seed 0 \
  --output-root outputs/propos --run-name grain_volcashdb_k21_seed0
```

ProPos requires an explicit `--initial-k`; that value stays fixed because the supervisor is disabled. For a K-sensitivity experiment, retrain from initialization for each requested K and seed. Do not reuse a trained state across K values.

### AutoProPos: adaptive K

```bash
# Grain-only, default initial K=200
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method autopropos --config configs/grain_only.yaml \
  --grain-root "$GRAIN_ROOT" --num-devices 4 --seed 0 \
  --output-root outputs/autopropos --run-name grain_only_seed0

# Grain + VolcAshDB, default initial K=200
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  methods/image_training/main.py \
  --method autopropos --config configs/grain_volcashdb.yaml \
  --grain-root "$GRAIN_ROOT" --volcashdb-root "$VOLCASHDB_ROOT" \
  --num-devices 4 --seed 0 \
  --output-root outputs/autopropos --run-name grain_volcashdb_seed0
```

For an initialization-sensitivity experiment, add `--initial-k K` and use a distinct run directory. Unlike ProPos, AutoProPos can revise K at the configured supervisor epochs.

There is one active Grain training loop. Prototype scattering uses the **ordinary 50-epoch warmup**; there is no alternative branch that postpones scattering to epoch 601. In the enriched regime, the auxiliary SSL pass may include VolcAshDB, but the scattering pass and inferred partition use Grain only.

The supervisor's silhouette scores are calculated from candidate cluster labels and remain part of the unsupervised algorithm. They are different from the ground-truth, class-balanced macro-silhouette used for post-hoc evaluation.

### Validate an image configuration before training

Use the same arguments with `--check-config`, without `torchrun`:

```bash
python methods/image_training/main.py \
  --method autopropos --config configs/grain_volcashdb.yaml \
  --grain-root "$GRAIN_ROOT" --volcashdb-root "$VOLCASHDB_ROOT" \
  --num-devices 4 --seed 0 --run-name check --check-config
```

This prints resolved options and checks directory/configuration consistency; it does not inspect an entire dataset or execute a training run. The enriched configuration requires `--volcashdb-root`, and the Grain-only configuration rejects it.

## Evaluation and multiple runs

### Metrics

All external metrics are implemented in [`grain_benchmark/evaluation.py`](grain_benchmark/evaluation.py):

- **ACC:** rectangular one-to-one Hungarian alignment; no replication of reference-class columns.
- **NMI / ARI:** computed directly from the two partitions; NMI uses arithmetic normalization.
- **K_occupied:** number of distinct predicted cluster IDs in the evaluated collection.

Scores are fractions, not percentages. Labels are accessed only for evaluation; they are never used to choose an exported model. The internal losses, mixture evidence and unsupervised supervisor criteria are retained where required for training.

Re-evaluate a feature-model output:

```bash
python -m grain_benchmark evaluate \
  --data-root data --regime grain_only \
  --predictions outputs/diva/grain_only/seed0/predictions.npz \
  --out outputs/diva/grain_only/seed0/metrics_recomputed.json
```

For a newly generated image-run output, use `--input-kind images`. That artifact includes its own evaluation labels and sample-identity metadata; neither the original RGB files nor frozen BYOL inputs are needed to recompute its metrics:

```bash
python -m grain_benchmark evaluate --input-kind images \
  --regime grain_only \
  --predictions outputs/autopropos/grain_only_seed0/predictions.npz \
  --out outputs/autopropos/grain_only_seed0/metrics_recomputed.json
```

Historical AutoProPos predictions are not included. These commands work on prediction files generated by this repository or files explicitly converted to its documented schema.

### Five independent runs

For example, in the DIVA environment:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python methods/diva/train.py \
    --config configs/grain_only.yaml --data-root data \
    --seed "$seed" --device cuda \
    --out "outputs/diva_5runs/grain_only/seed${seed}"
done

python -m grain_benchmark.summarize \
  --metrics outputs/diva_5runs/grain_only/seed*/metrics.json \
  --expected-runs 5 --std-ddof 1 \
  --out outputs/diva_5runs/grain_only/summary.json
```

Use the same pattern for DeepDPM or DDPM in their own environments. The seeds shown are illustrative; record the chosen seeds explicitly. KMeans already accepts a list through `--seeds`. Image methods require one fresh launch and unique `--run-name` per seed.

The aggregator checks run counts, duplicate seeds and experiment metadata. `--std-ddof 1` requests the sample standard deviation; use `0` for the population convention. The choice is explicit and recorded. Do not silently mix conventions or training settings.

### Structured mean confusion matrix

The matrix has shape **2C × C**. Main rows contain each majority class's dominant cluster; extra rows aggregate other clusters with that majority class. Columns sum to 100%. This construction **does not use Hungarian alignment** and must not be interpreted as an ACC confusion matrix.

```bash
python -m grain_benchmark structured-mean \
  --data-root data --regime grain_only \
  --predictions outputs/diva_5runs/grain_only/seed*/predictions.npz \
  --expected-runs 5 --out outputs/diva_5runs/grain_only/structured_mean

# Same calculation for newly produced image-model runs:
python -m grain_benchmark structured-mean --input-kind images \
  --regime grain_only \
  --predictions outputs/autopropos/grain_only_seed*/predictions.npz \
  --expected-runs 5 --out outputs/autopropos/grain_only_structured_mean
```

Image-run matrices require matching sample IDs and evaluation labels across runs. Outputs include the mean matrix, individual matrices, reference-class indices and normalization metadata; no arbitrary matching of cluster indices between runs is performed.

### Label-balanced macro-silhouette

```bash
for regime in grain_only grain_volcashdb; do
  python -m grain_benchmark silhouette --data-root data \
    --regime "$regime" --out "outputs/silhouette_${regime}.json"
done
```

This diagnostic uses cosine distance in the original L2-normalized 256-D BYOL space. It averages within each reference class and then equally across classes. It is not a score calculated on t-SNE coordinates, and it does not participate in representation learning or model selection.

## Outputs and implementation details

Feature-model runs write `predictions.npz`, `metrics.json`, `config_resolved.json` and `provenance.json`. KMeans additionally records its restart budget and native inertia; its directory layout is `<output>/<regime>/k<K>/seed<seed>/`. Other feature launchers write directly to `--out`.

Image runs write the resolved configuration, local logs, timing, optional supervisor diagnostics and:

```text
<output-root>/<run-name>/
├── predictions.npz          # last native Grain pseudo-label partition
├── metrics.json             # shared evaluation of that partition
└── final/
    ├── features.npy         # fresh extraction after the final optimizer step
    ├── labels.npy
    ├── row_ids.npy
    ├── image_paths.json     # paths relative to the Grain image root
    └── provenance.json
```

**The two final artifacts have different timing.** The native loop updates its pseudo-label partition before the Grain optimization pass. `predictions.npz` preserves that last native partition; `final/features.npy` is extracted after the last update. The code does not silently re-cluster the final features and report that different readout as the native result. Selection and partition epochs are recorded.

The Grain image implementation uses the native direct resize to 224 × 224 for deterministic extraction. The published frozen feature matrices are loaded without altering their stored values; each downstream method then applies its own documented native input processing. Image pseudo-label KMeans uses ten starts per process, while the supervisor uses the configured `supervisor_kmeans_n_init` starts for each candidate on its executing process.

Native model code lives under `methods/`; dataset validation, all external metrics and result handling live under `grain_benchmark/`. The standalone KMeans is shared under `torch_clustering/`. The bundled `bnpy` modules are DIVA's internal inference dependency, not additional benchmark entry points.

See [prediction-file details](docs/prediction_format.md) for row alignment and integrity checks. Re-running from the same inputs does not guarantee bitwise-identical floating-point results across different hardware or software environments. Full image-level reproduction also depends on access to the original Grain image collection.

## Tests

In the KMeans environment, install pytest and run:

```bash
python -m pip install pytest
python -m pytest -q
```

The full test suite also imports the image loader and ResNet-50 interface, so install a torchvision build matching the environment's torch build to run those tests. Tests cover common metrics, incomplete/overclustered partitions, structured matrices, label-independent final exports, configuration validation, image/auxiliary membership and the native KMeans core. CPU checks are not a substitute for validating complete GPU training runs in the method-specific environments.

## References and acknowledgements

This repository adapts the following implementations for the Grain benchmark:

| Component | Upstream reference |
|---|---|
| DIVA | [Ghiara/DIVA](https://github.com/Ghiara/DIVA) |
| DeepDPM | [BGU-CS-VIL/DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM) |
| DDPM | [naiqili/DDPM](https://github.com/naiqili/DDPM) |
| BYOL / ProPos implementation | [Hzzone/ProPos](https://github.com/Hzzone/ProPos) |
| AutoProPos | [Kana Tepakbong et al., Applied Sciences, 2025](https://doi.org/10.3390/app151810052) |
| VolcAshDB | [Benet et al., Scientific Data, 2025](https://doi.org/10.1038/s41597-025-04942-9) |

Please cite the benchmark, the clustering method(s) used and the relevant representation-learning work. Cite VolcAshDB when using its images. Third-party copyright and license notices are retained; see [THIRD_PARTY.md](THIRD_PARTY.md). The benchmark's original Grain images are not part of this distribution.
