"""Core reliability signals PB, PD, and their online aggregation."""

from dataclasses import dataclass
import math
import time
import torch


@dataclass
class SignalConfig:
    """Signal hyperparameters matching the experiment ``controller`` block."""

    alpha: float = 0.5
    beta: float = 0.9
    delta: float = 0.05
    prior_sigma: float = 0.05
    lambda_g: float = 1e-4
    lambda_u: float = 1e-3
    snapshot_device: str = "model"


class RunningZ:
    """Online z-score via Welford updates using only past and current observations.

    Online statistics avoid future leakage from the full trajectory. The first
    sample has no estimable std, so the raw observation is returned (for PD this
    preserves the first-epoch signal, including the epoch-mean mini-batch
    gradient norm). Later samples use sample variance ``m2 / (n - 1)``.
    """

    def __init__(self, eps=1e-8):
        """Initialize empty online mean / second-moment state."""

        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def update(self, x):
        """Ingest a new observation and return the updated standardized value."""

        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if self.n < 2:
            # No sample std yet: keep the first observation instead of forcing 0.
            return x
        std = max((self.m2 / (self.n - 1)) ** 0.5, self.eps)
        return (x - self.mean) / std


class ReliabilitySignals:
    """Compute complexity, dynamic risk, and smoothed reliability scores.

    ``theta0`` is the initialization snapshot for complexity relative to init;
    ``prev_state`` is the epoch-start snapshot for per-epoch update norms.
    These semantics differ and must not share one snapshot.
    """

    def __init__(self, model, n_train: int, cfg: SignalConfig):
        """Freeze the init snapshot and create independent online stats for PB/PD/val loss."""

        self.cfg = cfg
        self.n_train = int(n_train)
        if cfg.snapshot_device not in {"model", "cpu"}:
            raise ValueError("Signal snapshot_device must be 'model' or 'cpu'.")
        self.snapshot_device = cfg.snapshot_device
        # Fixed prior center theta_0: never updated within a run.
        self.theta0 = self._snapshot(model)
        self.prev_state = None
        self.z_pb = RunningZ()
        self.z_pd = RunningZ()
        self.z_val = RunningZ()
        self.rbar = None
        self.gradient_norm_seconds = 0.0
        self.update_norm_seconds = 0.0

    def _snapshot(self, model):
        """Copy trainable parameters; large models may store snapshots on CPU."""

        return {
            name: p.detach().to("cpu", copy=True) if self.snapshot_device == "cpu" else p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def _squared_distance(parameter, reference):
        """Squared distance for one parameter tensor on the reference device."""

        current = parameter.detach()
        if current.device != reference.device:
            current = current.to(reference.device)
        return torch.sum((current - reference) ** 2)

    @staticmethod
    def _sync_model(model):
        """Synchronize CUDA at timing boundaries so async kernels are not missed."""

        parameter = next(model.parameters(), None)
        if parameter is not None and parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    @torch.no_grad()
    def begin_epoch(self, model):
        """Record epoch-start parameters and reset per-epoch controller timers."""

        self.gradient_norm_seconds = 0.0
        self.update_norm_seconds = 0.0
        self.prev_state = self._snapshot(model)

    @torch.no_grad()
    def gradient_norm(self, model):
        """Return the full-parameter L2 norm of current backprop gradients."""

        self._sync_model(model)
        started = time.perf_counter()
        terms = []
        for p in model.parameters():
            if p.grad is not None:
                terms.append(torch.sum(p.grad.detach() ** 2))
        value = torch.stack(terms).sum().sqrt().item() if terms else 0.0
        self._sync_model(model)
        self.gradient_norm_seconds += time.perf_counter() - started
        return value

    @torch.no_grad()
    def update_norm(self, model):
        """Return ``||theta_t - theta_epoch_start||_2``."""

        self._sync_model(model)
        started = time.perf_counter()
        if self.prev_state is None:
            return 0.0
        terms = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            terms.append(self._squared_distance(p, self.prev_state[name]))
        value = torch.stack(terms).sum().sqrt().item() if terms else 0.0
        self._sync_model(model)
        self.update_norm_seconds += time.perf_counter() - started
        return value

    @torch.no_grad()
    def complexity_ct(self, model):
        """Compute ``C_t = ||theta_t-theta_0||^2 / (2*sigma^2)``."""

        self._sync_model(model)
        terms = []
        sigma2 = self.cfg.prior_sigma ** 2
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            terms.append(self._squared_distance(p, self.theta0[name]))
        total = torch.stack(terms).sum().item() if terms else 0.0
        self._sync_model(model)
        return total / (2.0 * sigma2)

    def compute(self, model, train_nll, val_nll, grad_norm, update_norm, variant="reliability"):
        """Compute PB, PD, instantaneous risk ``r``, and EMA ``rbar`` for a variant.

        PB combines validation NLL with an init-relative complexity penalty; PD
        combines positive risk mismatch, gradient norm, and update norm. Each
        ``variant`` only swaps terms required by the paper ablation while keeping
        the same online stats and control interface.
        """

        pb_started = time.perf_counter()
        c_t = self.complexity_ct(model)
        n = max(1, self.n_train)

        # PAC-Bayes-style bound term: delta sets confidence; n is train subset size.
        pb = val_nll + math.sqrt(max(0.0, (c_t + math.log(2.0 * math.sqrt(n) / self.cfg.delta)) / (2.0 * n)))
        pb_seconds = time.perf_counter() - pb_started
        pd_started = time.perf_counter()
        # Penalize only val risk above train risk; negative gaps are clipped to 0.
        v_t = max(0.0, val_nll - train_nll)
        pd_terms = {
            "reliability_pd_full": v_t + self.cfg.lambda_g * grad_norm + self.cfg.lambda_u * update_norm,
            "reliability_pd_v_only": v_t,
            "reliability_pd_v_h": v_t + self.cfg.lambda_g * grad_norm,
            "reliability_pd_v_u": v_t + self.cfg.lambda_u * update_norm,
            "reliability_pd_h_u": self.cfg.lambda_g * grad_norm + self.cfg.lambda_u * update_norm,
        }
        pd = pd_terms.get(variant, pd_terms["reliability_pd_full"])
        pd_seconds = self.gradient_norm_seconds + self.update_norm_seconds + (time.perf_counter() - pd_started)

        # PB/PD have different scales: online-standardize separately, then alpha-fuse.
        standardization_started = time.perf_counter()
        pb_z = self.z_pb.update(pb)
        pd_z = self.z_pd.update(pd)
        val_z = self.z_val.update(val_nll)

        if variant == "reliability_pb_only":
            r = pb_z
        elif variant == "reliability_pd_only":
            r = pd_z
        elif variant == "reliability_val_loss_only":
            r = val_z
        else:
            r = self.cfg.alpha * pb_z + (1.0 - self.cfg.alpha) * pd_z

        # beta is cross-epoch EMA; the no-smoothing ablation uses instantaneous r.
        if variant == "reliability_no_smoothing":
            self.rbar = r
        elif self.rbar is None:
            self.rbar = r
        else:
            self.rbar = self.cfg.beta * self.rbar + (1.0 - self.cfg.beta) * r
        standardization_seconds = time.perf_counter() - standardization_started

        return {
            "complexity_ct": c_t,
            "pb_signal": pb,
            "risk_violation_vt": v_t,
            "pd_signal": pd,
            "pb_z": pb_z,
            "pd_z": pd_z,
            "val_loss_z": val_z,
            "reliability_r": r,
            "reliability_rbar": self.rbar,
            "pb_computation_seconds": pb_seconds,
            "pd_computation_seconds": pd_seconds,
            "standardization_smoothing_seconds": standardization_seconds,
        }
