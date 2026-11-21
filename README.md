# TAC Algorithm Reproducibility Package

Minimal source code for the **Training-Aware Controller (TAC)** used in:

> Improving Deep Neural Network Training Using Training-State-Guided Adaptive Learning Rate Control for Image Classification

This package contains **only the algorithm and a runnable experiment entry point** for peer-review reproduction. Large result archives, thesis plotting scripts, and multi-seed sweep infrastructure are intentionally excluded.

## Algorithm components

| File | Role |
|------|------|
| `src/signals.py` | PAC-Bayes-inspired PB signal, primal-dual-inspired PD signal, online z-score, alpha-fusion, beta-EMA |
| `src/controller.py` | Reliability patience `p_lr` to multiplicative LR decay (fixed epoch budget; no early stopping) |
| `src/metrics.py` | Accuracy, NLL, ECE, Brier, AURC and related evaluation utilities |
| `src/models.py` | Backbone constructors (ResNet / VGG / ConvNeXt / DeiT) |
| `src/data.py` | CIFAR-10/100 and ImageNet-100 loaders |
| `run_experiment.py` | End-to-end training loop that wires signals + controller |

Default controller hyperparameters (paper main setting):

- alpha = 0.5, beta = 0.9, sigma = 0.05, gamma = 0.5, `p_lr` = 8, `min_lr` = 1e-5
- Training budget: fixed `epochs` (default 120 in the main config)
- Checkpoint selection: best validation accuracy

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/macOS:
# .venv/bin/python -m pip install -r requirements.txt
```

Requires PyTorch with CUDA for the paper protocol (`"device": "cuda"`). CIFAR experiments download automatically into `./data`.

## Reproduce a smoke test (minutes)

```bash
python run_experiment.py --config configs/quick_cifar10_resnet18.json
```

This runs Cosine vs TAC (`reliability`) for 3 epochs on CIFAR-10 / ResNet-18 and writes artifacts under `results/quick_cifar10_resnet18/{method}/seed_1/`.

## Reproduce the paper protocol setting (CIFAR-10 / ResNet-18)

```bash
python run_experiment.py --config configs/cifar10_resnet18_main.json
```

Methods: Step, Cosine, ReduceLROnPlateau (`plateau`), TAC (`reliability`). Fixed seed `{1}`, 120 epochs.

## Unit tests (algorithm only)

```bash
python -m unittest tests.test_algorithm -v
```

## Notes for reviewers

1. **Single-seed protocol.** Configs use `"seeds": [1]` for RNG reproducibility; there is no multi-seed sweep in this package.
2. **Fixed epoch budget.** The controller adjusts the learning rate only; it does not early-stop training.
3. **Full matrix.** The paper evaluates 15 dataset-backbone settings. This package ships the reference CIFAR-10/ResNet-18 configs to keep the upload small.
4. **License.** MIT (see `LICENSE`).

## Citation

Please cite the corresponding manuscript when using this code.
