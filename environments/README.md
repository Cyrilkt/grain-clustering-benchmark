# Environment notes

Create environments from the repository root; paths inside requirements files are relative to those files. These are method-specific dependency sets, not an exact export of the historical server. Use the CUDA-enabled torch build appropriate to the machine and verify `torch.cuda.is_available()` before GPU training.

| Environment | Python | Interface constraint |
|---|---|---|
| Evaluation | 3.10 | NumPy / SciPy / scikit-learn only; no deep-learning framework required |
| KMeans | 3.10 | PyTorch; the local `torch_clustering` package supplies the algorithm |
| DIVA | 3.10 | Lightning 1.9.5 for the retained epoch hooks and Trainer interface |
| DeepDPM | 3.9 | Lightning 1.2.10 / torch 1.11.0 for the native multiple-optimizer and hyperparameter APIs |
| DDPM | 3.10 | torch 2.1.2; no Lightning dependency |
| Image training | 3.10 | torch 2.1.2 / torchvision 0.16.2; launch with `torchrun` |

Do not upgrade DIVA or DeepDPM to Lightning 2.x without an explicit port and new numerical validation. DeepDPM's internal `kmeans-pytorch` dependency is separate from the benchmark's standalone `PyTorchKMeans`; retain it for its native initialization path.

`requirements-evaluation.txt` installs the shared evaluator and configuration loader. For the complete repository tests, also install torch, a matching torchvision build and pytest. Image configuration tests use CPU and temporary synthetic images. Native DIVA/DeepDPM training requires the respective environments above.

Local CPU checks validate the evaluation code, configuration interfaces and selected numerical components. The dependency sets have not been certified by complete GPU reruns, and the test suite does not imply that such reruns were performed.
