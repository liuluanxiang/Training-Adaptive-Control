# reliable-training-control (algorithm + configs)

Minimal reproducibility package for the **Training-Aware Controller (TAC)**:

- Algorithm: PAC-Bayes-inspired PB + primal–dual-inspired PD signals, online fusion/EMA, LR control
- Experiment entry: `run_experiment.py`
- Experiment configs: `configs/` (including `revision_main_matrix/`)

No training result archives are included.

## Layout

```text
src/signals.py      # PB / PD / z-score / fusion / EMA
src/controller.py   # reliability LR controller (fixed epoch budget)
src/data.py         # dataset loaders
src/models.py       # backbones
src/metrics.py      # evaluation metrics
run_experiment.py   # training entry point
configs/            # experiment JSON configs
tests/              # algorithm unit tests
```

Default controller: α=0.5, β=0.9, σ=0.05, γ=0.5, p_lr=8, min_lr=1e-5.  
Protocol: fixed `epochs`, checkpoint by validation accuracy, single seed `{1}`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Smoke test

```bash
python run_experiment.py --config configs/quick_cifar10_resnet18.json
```

## Paper main setting (CIFAR-10 / ResNet-18)

```bash
python run_experiment.py --config configs/cifar10_resnet18_main.json
```


## License

MIT

## Algorithm sketch

Each epoch after validation:

1. Compute PB from validation NLL and parameter-displacement complexity.
2. Compute PD from positive risk mismatch V, gradient norm H, and update norm U.
3. Online-standardize and fuse: r = alpha * z_PB + (1 - alpha) * z_PD.
4. Smooth: rbar = beta * rbar + (1 - beta) * r.
5. If rbar fails to improve for p_lr epochs: eta <- max(gamma * eta, min_lr).

Training uses a fixed epoch budget; checkpoint selection uses validation accuracy.
