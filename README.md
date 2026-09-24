# Grain Clustering Benchmark

Code and frozen representations for **Practical Nonparametric Deep Clustering for Grain Microscopy: A 21-Class Benchmark and the Benefit of VolcAshDB Pretraining**.

The benchmark contains **5,240 mineral-grain images from 21 reference classes** and evaluates DIVA, DeepDPM, DDPM and AutoProPos under two representation-learning regimes: **Grain-only** and **Grain + VolcAshDB**.

## Methods

| Method | Input |
|---|---|
| KMeans | frozen BYOL embeddings |
| DIVA | frozen BYOL embeddings |
| DeepDPM | frozen BYOL embeddings |
| DDPM | frozen BYOL embeddings |
| BYOL | RGB images |
| ProPos | RGB images |
| AutoProPos | RGB images |

The two experiment configurations are:

```text
configs/grain_only.yaml
configs/grain_volcashdb.yaml
```

## Data

The two frozen **256-D BYOL representations** used by the feature-based methods are included in `data/embeddings/`:

```text
grain_only/{features.npy, labels.npy, row_ids.npy}
grain_volcashdb/{features.npy, labels.npy, row_ids.npy}
```

`labels.npy` is used only for post-hoc evaluation. The original Grain RGB images are not included; they are required only for BYOL, ProPos and AutoProPos training.

Check the provided embeddings with:

```bash
python -m grain_benchmark validate --data-root data
```

### VolcAshDB

VolcAshDB contributes additional unlabeled images only during the **Grain + VolcAshDB** representation-learning regime. It is never included in the Grain clustering partition.

- Images: https://figshare.com/articles/dataset/Particle_images/28284980
- Dataset paper: https://doi.org/10.1038/s41597-025-04942-9
- Website: https://volcashdb.ipgp.fr/

## Installation

Use the dependency file corresponding to the method you want to run:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r environments/requirements-kmeans.txt
```

Available files are `requirements-kmeans.txt`, `requirements-diva.txt`, `requirements-deepdpm.txt`, `requirements-ddpm.txt` and `requirements-image.txt` in `environments/`.

## Run the benchmark

Run all commands from the repository root. The examples use `grain_only.yaml`; replace it with `grain_volcashdb.yaml` for the enriched representation.

### KMeans

```bash
python -m grain_benchmark kmeans --config configs/grain_only.yaml --data-root data \
  --k-values 21 --seeds 0 --device cpu --out outputs/kmeans
```

### DIVA

```bash
CUDA_VISIBLE_DEVICES=0 python methods/diva/train.py --config configs/grain_only.yaml \
  --data-root data --seed 0 --device cuda --out outputs/diva/grain_only/seed0
```

### DeepDPM

```bash
CUDA_VISIBLE_DEVICES=0 python methods/deepdpm/train.py --config configs/grain_only.yaml \
  --data-root data --seed 0 --device cuda --out outputs/deepdpm/grain_only/seed0
```

### DDPM

```bash
CUDA_VISIBLE_DEVICES=0 python methods/ddpm/train.py --config configs/grain_only.yaml \
  --data-root data --seed 0 --device cuda --out outputs/ddpm/grain_only/seed0
```

## BYOL, ProPos and AutoProPos

These methods operate directly on RGB images:

```bash
export GRAIN_ROOT=/path/to/grain_rgb
export VOLCASHDB_ROOT=/path/to/volcashdb_rgb
```

The examples below use four GPUs.

### BYOL

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 methods/image_training/main.py \
  --method byol --config configs/grain_only.yaml --grain-root "$GRAIN_ROOT" \
  --num-devices 4 --seed 0 --output-root outputs/byol --run-name grain_only_seed0
```

### ProPos

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 methods/image_training/main.py \
  --method propos --config configs/grain_only.yaml --grain-root "$GRAIN_ROOT" --initial-k 21 \
  --num-devices 4 --seed 0 --output-root outputs/propos --run-name grain_only_k21_seed0
```

### AutoProPos

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 methods/image_training/main.py \
  --method autopropos --config configs/grain_only.yaml --grain-root "$GRAIN_ROOT" \
  --num-devices 4 --seed 0 --output-root outputs/autopropos --run-name grain_only_seed0
```

For **Grain + VolcAshDB**, use `configs/grain_volcashdb.yaml` and add `--volcashdb-root "$VOLCASHDB_ROOT"`.

## Evaluation

All methods use the same external evaluation code for **ACC, NMI, ARI and inferred K**. ACC uses a one-to-one Hungarian assignment.

```bash
python -m grain_benchmark evaluate --data-root data --regime grain_only \
  --predictions outputs/diva/grain_only/seed0/predictions.npz \
  --out outputs/diva/grain_only/seed0/metrics.json
```

The repository also includes the label-balanced macro-silhouette and structured mean confusion analysis used in the paper. Five-run aggregation and additional commands are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Original methods and code

This repository adapts existing methods to the Grain benchmark. Please cite the corresponding original paper when using a method.

| Method | Paper | Original code |
|---|---|---|
| DIVA | [Bing et al., 2023](https://arxiv.org/abs/2305.14067) | [Ghiara/DIVA](https://github.com/Ghiara/DIVA) |
| DeepDPM | [Ronen et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.html) | [BGU-CS-VIL/DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM) |
| DDPM | [Li et al., UAI 2022](https://proceedings.mlr.press/v180/li22c.html) | [naiqili/DDPM](https://github.com/naiqili/DDPM) |
| BYOL | [Grill et al., NeurIPS 2020](https://arxiv.org/abs/2006.07733) | implementation adapted from the ProPos codebase |
| ProPos | [Huang et al., TPAMI](https://arxiv.org/abs/2111.11821) | [Hzzone/ProPos](https://github.com/Hzzone/ProPos) |
| AutoProPos | [Kana Tepakbong et al., 2025](https://doi.org/10.3390/app151810052) | [Cyrilkt/AutoProPos](https://github.com/Cyrilkt/AutoProPos) |
| KMeans | — | [Hzzone/torch_clustering](https://github.com/Hzzone/torch_clustering) |

Third-party notices and code provenance are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

If you use this repository, please cite the Grain clustering benchmark and the original method(s) used in your experiment. If you use VolcAshDB images, please also cite the VolcAshDB dataset paper.
