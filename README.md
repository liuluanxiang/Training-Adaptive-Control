# reliable-training-control

Algorithm and experiment-config package for the **Training-Aware Controller (TAC)**.

## What is included

- `src/signals.py` — PB / PD diagnostics, online z-score, α-fusion, β-EMA  
- `src/controller.py` — reliability patience → learning-rate decay (fixed epoch budget)  
- `src/data.py`, `src/models.py`, `src/metrics.py` — data, backbones, evaluation  
- `run_experiment.py` — training entry point  
- `configs/` — experiment JSON configs (main matrix, sensitivity, ablation)

No result archives are included.

## Control rule (brief)

Each epoch after validation:

1. Compute PB from validation NLL and parameter-displacement complexity.  
2. Compute PD from positive risk mismatch, gradient norm, and update norm.  
3. Online-standardize and fuse: `r = α z_PB + (1-α) z_PD`.  
4. Smooth: `r̄_t = β r̄_{t-1} + (1-β) r_t`.  
5. If `r̄` fails to improve for `p_lr` epochs: `η ← max(γ η, η_min)`.

Training uses a fixed epoch budget. The reported model is the best validation-accuracy checkpoint.

Default hyperparameters: α=0.5, β=0.9, σ=0.05, γ=0.5, `p_lr`=8, `min_lr`=1e-5, seed=`{1}`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Smoke test:

```bash
python run_experiment.py --config configs/quick_cifar10_resnet18.json
```

CIFAR-10 / ResNet-18 main setting:

```bash
python run_experiment.py --config configs/cifar10_resnet18_main.json
```

Revision main-matrix configs are under `configs/revision_main_matrix/`.

## License

MIT
