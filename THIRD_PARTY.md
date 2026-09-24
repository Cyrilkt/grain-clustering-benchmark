# Third-party code and data

The method implementations retain their own provenance and notices. This repository does not replace those notices with a blanket license over all components.

| Component | Origin | Included notice |
|---|---|---|
| DIVA | Grain adaptation of Ghiara/DIVA | Source attribution; the provided archive did not contain a standalone DIVA license file |
| bnpy | Inference runtime included with the DIVA implementation | Existing source notices retained |
| DeepDPM | BGU-CS-VIL/DeepDPM | `methods/deepdpm/LICENSE` and source copyright headers |
| DDPM | naiqili/DDPM | `methods/ddpm/LICENSE` |
| Image training | Grain AutoProPos implementation based on ProPos/BYOL | `methods/image_training/LICENSE` and retained author/copyright headers |
| Native KMeans | `torch_clustering` implementation used by AutoProPos | Source attribution; checksums of the four numerical core files in `tests/kmeans_core_sha256.json` |

The shared image runtime retains the momentum/shuffle primitives needed by the model, including their existing copyright headers. DIVA's bundled `bnpy` is an internal dependency; removing its demonstration scripts does not change the ownership of its retained components.

Consult the respective upstream authors for components without an explicit license grant in the available archive. No additional redistribution rights are implied here. Original Grain RGB images are excluded. VolcAshDB is obtained separately from its official dataset record, whose attribution and license terms apply to those images.
