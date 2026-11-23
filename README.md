# Training-Adaptive-Control (TAC)

Algorithm code and experimental metric archive for **Training-Adaptive-Control**.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | PB/PD signals, online Welford z-score, controller, data/models/metrics |
| `run_experiment.py` | Training entry point |
| `configs/` | Experiment JSON configs |
| `result_TAC/` | Uploaded experimental **metric** archive (see below) |

## Experimental archive (`result_TAC/`)

Flat layout: `result_TAC/<experiment>/<method>/`.

Experiment name prefixes:

- `main_*` — main matrix (dataset × backbone × scheduler)
- `controller_ablation_*` / `pd_component_ablation_*` — controller ablations
- `sens_*` — α/β sensitivity
- `sigma_*` — prior-σ sensitivity

Per-run files typically include:

- `summary.json` — headline test metrics
- `history.csv` / `train_log.csv` — epoch trajectories (PB/PD, LR, losses)
- `calibration_bins.csv` — reliability-diagram bins
- `risk_coverage_curve.csv` — selective-prediction curve
- `config_used.json`, `run_metadata.json`

**Not included in git:** large binaries (`*.pt` / `*.pth`), i.e. `best_checkpoint.pt`, `test_logits.pt`, `test_targets.pt`. They are omitted because of GitHub size limits. Ablation / αβ / σ cells were archived as metrics-only.

Default branch `main` contains code **and** `result_TAC/`. Branch `result-TAC` contains the metric archive only.

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
