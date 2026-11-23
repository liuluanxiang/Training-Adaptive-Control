# Training-Adaptive-Control (TAC)

Algorithm code for **Training-Adaptive-Control** (PB/PD signals and reliability-driven LR control).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | PB/PD signals, online Welford z-score, controller, data/models/metrics |
| `run_experiment.py` | Training entry point |
| `configs/` | Experiment JSON configs |

Experimental result archives are **not** shipped in this repository. Run locally to regenerate metrics under a local `result_TAC/` (gitignored).

## Control rule (brief)

Each epoch after validation:

1. PB from validation NLL and parameter-displacement complexity.  
2. PD from nonnegative risk mismatch, gradient norm, and update norm.  
3. Online-standardize and fuse: `r = α z_PB + (1-α) z_PD`.  
4. Smooth: `r̄_t = β r̄_{t-1} + (1-β) r_t`.  
5. If `r̄` fails to improve for `p_lr` epochs: `η ← max(γ η, η_min)`.

Fixed epoch budget; reported model is the best validation-accuracy checkpoint.  
Default hyperparameters: α=0.5, β=0.9, σ=0.05, γ=0.5, `p_lr`=8, `min_lr`=1e-5.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py --config configs/quick_cifar10_resnet18.json
python run_experiment.py --config configs/cifar10_resnet18_main.json
```

## License

MIT
