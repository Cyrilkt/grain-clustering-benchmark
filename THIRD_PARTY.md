# Third-party code and licenses

This benchmark reuses and adapts research implementations. The project-level MIT license applies only to original benchmark utilities, **not** to all files under `models/` or to the data. The table below identifies each upstream source and its applicable or unresolved terms. Copyright headers and existing licenses in the source files have been preserved.

| Component in this repository | Original source | License or permission status |
|---|---|---|
| `models/deepdpm/` | [DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM) | **MIT**; text retained in `models/deepdpm/LICENSE` |
| `models/ddpm/` | [DDPM](https://github.com/naiqili/DDPM) | **MIT**; text retained in `models/ddpm/LICENSE` |
| `models/diva/bnpy/` | [bnpy](https://github.com/bnpy/bnpy) | **BSD-3-Clause**; upstream notice copied into `models/diva/bnpy/LICENSE.txt` |
| `models/diva/` (DIVA adaptation apart from bnpy) | [DIVA](https://github.com/Ghiara/DIVA) | **No explicit license found** in the public upstream repository (checked 2026-10-23). Attribution is not a grant of permission; redistribution rights require clarification with the upstream authors |
| `models/kmeans/torch_clustering/` | [torch_clustering](https://github.com/Hzzone/torch_clustering) | **MIT**; upstream notice copied into `models/kmeans/torch_clustering/LICENSE`. The upstream copyright line still contains `[year] [fullname]` |
| `models/autopropos/` | [AutoProPos](https://github.com/Cyrilkt/AutoProPos) | **MIT notice** in upstream repository, copied into `models/autopropos/LICENSE`. The upstream copyright line retains placeholders; this notice cannot override rights in the underlying ProPos code |
| `models/common/image_training/` and ProPos-based image training (BYOL, ProPos, AutoProPos) | [ProPos](https://github.com/Hzzone/ProPos) and [AutoProPos](https://github.com/Cyrilkt/AutoProPos) | **ProPos has no explicit upstream license found** (checked 2026-10-23). The existing `models/common/image_training/LICENSE` is a generic MIT text with placeholders and must **not** be interpreted as a verified license grant from the ProPos authors |
| Momentum/shuffle routines in `models/common/image_training/propos/momentum_base.py` | Facebook/Meta code, as indicated by the preserved source header | Original copyright notice retained. Its precise upstream provenance and applicable redistribution license should be verified before broader reuse |
| C helper `models/diva/bnpy/util/lib/sparseResp/fastexp.h` | Paul Mineiro | Its BSD-style redistribution notice is retained verbatim in the file |

### Data

- **Grain RGB images:** not distributed in this repository.
- **Frozen BYOL embeddings and reference labels:** supplied to support the reported feature-level experiments. They are not covered by the repository MIT license; broader reuse permissions should be confirmed with the relevant data rights holders.
- **VolcAshDB:** images are obtained separately from the [dataset record](https://figshare.com/articles/dataset/Particle_images/28284980); the dataset provider's own conditions apply.

### What still needs attention

The DIVA and ProPos source repositories are publicly accessible but were not accompanied by an explicit license at the time of inspection. Because parts of their code have been adapted or included, the maintainers' permission or another authoritative license statement is needed to establish redistribution rights. **Do not label the full repository MIT-licensed on the basis of the root license or the AutoProPos license.**

Upstream sources were checked on 23 October 2026. Their notices and the exact version of each source should be rechecked if model code is replaced or updated. Citation of the original papers remains necessary for scholarly attribution but does not substitute for a software license.
