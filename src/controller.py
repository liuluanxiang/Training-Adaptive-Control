"""Learning-rate controller driven by the reliability score.

Protocol
--------
Predicate (lower is better; ``epsilon`` suppresses numeric noise)::

    rel_improved := finite(rbar) ∧ (rbar < best_rbar - epsilon)

Learning-rate control::

    if epoch > control_warmup_epochs and finite(rbar):
        if best_rbar not yet set: best_rbar ← rbar   # baseline only, no stagnation
        elif rel_improved: best_rbar ← rbar; bad_rel ← 0
        else: bad_rel ← bad_rel + 1
        if bad_rel >= p_lr:
            η ← max(γη, min_lr); bad_rel ← 0     # LR drop resets reliability patience

The training budget is the configured ``epochs``; this controller never early-stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Controller hyperparameters.

    ``p_lr``: patience before an LR reduction.
    ``epsilon``: minimum improvement required for a strict update.
    ``control_warmup_epochs``: skip reliability patience early on
    (avoids locking onto the first RunningZ observation before a sample std exists).
    """

    gamma: float = 0.5
    min_lr: float = 1e-5
    p_lr: int = 8
    epsilon: float = 1e-4
    control_warmup_epochs: int = 2


class ReliabilityController:
    """Track the smoothed reliability score and emit LR-control actions."""

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.best_rbar = float("inf")
        self._rbar_initialized = False
        self.bad_rel = 0
        self.lr_reductions = 0
        self.first_lr_drop_epoch = -1

    @staticmethod
    def _finite(x: float) -> bool:
        return x is not None and math.isfinite(float(x))

    def step(self, optimizer, rbar, epoch, val_loss=None):
        """Consume one epoch reliability observation and return LR decisions/audit state.

        ``val_loss`` remains optional for call-site compatibility but is unused.
        """

        del val_loss  # early stopping removed; validation loss does not drive control
        lr_reduced = False

        eps = float(self.cfg.epsilon)
        rbar_f = float(rbar) if rbar is not None else float("nan")
        rel_active = int(epoch) > int(self.cfg.control_warmup_epochs) and self._finite(rbar_f)

        if rel_active:
            if not self._rbar_initialized:
                # Establish baseline without counting stagnation on the first valid point
                self.best_rbar = rbar_f
                self._rbar_initialized = True
                rel_improved = True
            else:
                rel_improved = rbar_f < self.best_rbar - eps
                if rel_improved:
                    self.best_rbar = rbar_f

            if rel_improved:
                self.bad_rel = 0
            else:
                self.bad_rel += 1

            if self.bad_rel >= int(self.cfg.p_lr):
                lr_reduced = self._reduce_lr(optimizer)
                self.bad_rel = 0
                if lr_reduced:
                    self.lr_reductions += 1
                    if self.first_lr_drop_epoch < 0:
                        self.first_lr_drop_epoch = int(epoch)

        return {
            "lr_reduced": lr_reduced,
            "bad_rel": self.bad_rel,
            "bad_rel_lr": self.bad_rel,
            "first_lr_drop_epoch": self.first_lr_drop_epoch,
            "lr_reductions": self.lr_reductions,
            "rel_active": rel_active,
            "rbar_initialized": self._rbar_initialized,
        }

    def _reduce_lr(self, optimizer) -> bool:
        reduced = False
        for group in optimizer.param_groups:
            old_lr = group["lr"]
            group["lr"] = max(old_lr * self.cfg.gamma, self.cfg.min_lr)
            reduced = reduced or group["lr"] < old_lr
        return reduced
