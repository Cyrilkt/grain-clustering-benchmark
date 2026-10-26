# Grain Clustering Benchmark

Code and frozen representations for **Practical Nonparametric Deep Clustering for Grain Microscopy: A 21-Class Benchmark and the Benefit of VolcAshDB Pretraining**.

The benchmark contains **5,240 mineral-grain images from 21 reference classes** and evaluates nonparametric deep clustering under two representation-learning regimes:

- **Grain-only**
- **Grain + VolcAshDB**

## Repository structure

```text
models/          DIVA, DeepDPM, DDPM, KMeans, BYOL, ProPos and AutoProPos
configs/         Grain-only and Grain + VolcAshDB configurations
data/            Frozen BYOL embeddings and evaluation labels
grain_benchmark/ Common evaluation code
```

All model implementations used in the benchmark are under `models/`.

## Data

The repository includes the frozen **256-dimensional BYOL embeddings** used by DIVA, DeepDPM, DDPM and KMeans:

```text
data/embeddings/
├── grain_only/
│   ├── features.npy
│   ├── labels.npy
│   └── row_ids.npy
└── grain_volcashdb/
    ├── features.npy
    ├── labels.npy
    └── row_ids.npy
```

`labels.npy` is used only for post-hoc evaluation.

The original Grain RGB images are not included. They are required only to train **BYOL, ProPos and AutoProPos** directly from images.

### VolcAshDB

VolcAshDB is used only as additional unlabeled data during the **Grain + VolcAshDB** representation-learning regime. VolcAshDB images are never included in the Grain clustering partition.

- Images: https://figshare.com/articles/dataset/Particle_images/28284980
- Dataset paper: https://doi.org/10.1038/s41597-025-04942-9
- Website: https://volcashdb.ipgp.fr/

## Installation

Install the dependencies for the method you want to run.

For example, for DIVA:

```bash
pip install -r environments/requirements-diva.txt
```

Available environments:

```text
environments/requirements-kmeans.txt
environments/requirements-diva.txt
environments/requirements-deepdpm.txt
environments/requirements-ddpm.txt
environments/requirements-image.txt
```

## Run the models

Run commands from the repository root.

The examples below use `configs/grain_only.yaml`. To use the enriched representation, replace it with:

```text
configs/grain_volcashdb.yaml
```

### KMeans

```bash
python models/kmeans/train.py \
  --config configs/grain_only.yaml \
  --data-root data \
  --k-values 21 \
  --seeds 0 \
  --device cpu \
  --out outputs/kmeans
```

### DIVA

```bash
CUDA_VISIBLE_DEVICES=0 python models/diva/train.py \
  --config configs/grain_only.yaml \
  --data-root data \
  --seed 0 \
  --device cuda \
  --out outputs/diva/grain_only/seed0
```

### DeepDPM

```bash
CUDA_VISIBLE_DEVICES=0 python models/deepdpm/train.py \
  --config configs/grain_only.yaml \
  --data-root data \
  --seed 0 \
  --device cuda \
  --out outputs/deepdpm/grain_only/seed0
```

### DDPM

```bash
CUDA_VISIBLE_DEVICES=0 python models/ddpm/train.py \
  --config configs/grain_only.yaml \
  --data-root data \
  --seed 0 \
  --device cuda \
  --out outputs/ddpm/grain_only/seed0
```

### BYOL

BYOL operates directly on RGB images:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 models/byol/train.py \
  --config configs/grain_only.yaml \
  --grain-root /path/to/grain_rgb \
  --num-devices 4 \
  --seed 0 \
  --output-root outputs/byol \
  --run-name grain_only_seed0
```

### ProPos

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 models/propos/train.py \
  --config configs/grain_only.yaml \
  --grain-root /path/to/grain_rgb \
  --initial-k 21 \
  --num-devices 4 \
  --seed 0 \
  --output-root outputs/propos \
  --run-name grain_only_k21_seed0
```

### AutoProPos

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 models/autopropos/train.py \
  --config configs/grain_only.yaml \
  --grain-root /path/to/grain_rgb \
  --num-devices 4 \
  --seed 0 \
  --output-root outputs/autopropos \
  --run-name grain_only_seed0
```

For **Grain + VolcAshDB** image-level training, use `configs/grain_volcashdb.yaml` and add:

```bash
--volcashdb-root /path/to/volcashdb_rgb
```

## Evaluation

All methods use the same external evaluation code for **ACC, NMI, ARI and inferred K**. ACC uses a one-to-one Hungarian assignment.

Example:

```bash
python -m grain_benchmark evaluate \
  --data-root data \
  --regime grain_only \
  --predictions outputs/diva/grain_only/seed0/predictions.npz \
  --out outputs/diva/grain_only/seed0/metrics.json
```

The common evaluation module also contains the macro-silhouette and structured mean confusion analysis used in the paper.

## Original methods and code

This repository contains benchmark-specific adaptations of existing methods. Please cite the corresponding original work when using a method.

| Method | Paper | Original code |
|---|---|---|
| DIVA | [Bing et al., 2023](https://arxiv.org/abs/2305.14067) | [Ghiara/DIVA](https://github.com/Ghiara/DIVA) |
| DeepDPM | [Ronen et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.html) | [BGU-CS-VIL/DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM) |
| DDPM | [Li et al., UAI 2022](https://proceedings.mlr.press/v180/li22c.html) | [naiqili/DDPM](https://github.com/naiqili/DDPM) |
| BYOL | [Grill et al., NeurIPS 2020](https://arxiv.org/abs/2006.07733) | implementation adapted from the ProPos codebase |
| ProPos | [Huang et al., TPAMI](https://arxiv.org/abs/2111.11821) | [Hzzone/ProPos](https://github.com/Hzzone/ProPos) |
| AutoProPos | [Kana Tepakbong et al., 2025](https://doi.org/10.3390/app151810052) | [Cyrilkt/AutoProPos](https://github.com/Cyrilkt/AutoProPos) |
| KMeans | — | [Hzzone/torch_clustering](https://github.com/Hzzone/torch_clustering) |

Third-party licenses and unresolved permissions are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Licenses and reuse

This is a **multi-license repository**. The [root MIT license](LICENSE) covers original benchmark utilities, **not** the adapted model implementations or the released embeddings. Method-specific licenses and attribution are listed in [THIRD_PARTY.md](THIRD_PARTY.md), and the exact scope is explained in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

The original Grain RGB images are not distributed. The released BYOL embeddings support reproduction of the feature-based experiments; their availability does not imply unrestricted redistribution rights. VolcAshDB images must be downloaded separately under the dataset provider's terms.

## Citation

If you use this repository, please cite the Grain clustering benchmark and the original method(s) used in your experiment. If you use VolcAshDB images, please also cite the VolcAshDB dataset paper.
