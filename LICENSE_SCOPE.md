# License scope

This repository combines original benchmark code, adapted research methods, and released feature data. **It does not have a single license for every file.**

## Original benchmark utilities

The root [MIT license](LICENSE) is intended for the original benchmark utilities developed for this project, including `grain_benchmark/`, `configs/`, `environments/`, `tests/`, and the evaluation-related documentation. Its copyright notice names Cyril Kana Tepakbong and contributors. The relevant contributors or rights holders should approve this grant before it is published.

## Adapted method implementations

The root MIT license **does not apply** to `models/` as a whole. These folders contain original or modified third-party implementations. Each component retains its own copyright, attribution, and applicable terms, as documented in [THIRD_PARTY.md](THIRD_PARTY.md). A downstream adaptation cannot grant new rights over the upstream code.

In particular, the public DIVA and ProPos repositories did not show an explicit software license when checked on 23 October 2026. Permission to redistribute their code cannot be inferred from public GitHub access. Their provenance is documented, but any remaining redistribution permission must be confirmed with the upstream rights holders.

## Embeddings and images

The root MIT license **does not apply** to `data/embeddings/`, the reference labels, or any source microscopy images. Embeddings are supplied to reproduce the feature-level benchmark; this scientific availability does not itself specify broader redistribution rights. The original Grain RGB images are not included. VolcAshDB images are not redistributed here and retain their own dataset terms.

This file describes the scope of the license notice. It does not replace the separate licenses, copyright notices, or permissions of the component rights holders.
