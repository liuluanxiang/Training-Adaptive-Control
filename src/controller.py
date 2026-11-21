"""基于可靠性信号的学习率调整控制器。

协议
----
谓词（越小越好；``epsilon`` 抑制数值噪声）::

    rel_improved := finite(rbar) ∧ (rbar < best_rbar - epsilon)

学习率控制::

    if epoch > control_warmup_epochs and finite(rbar):
        if best_rbar 尚未初始化: best_rbar ← rbar   # 仅建基线，不计停滞
        elif rel_improved: best_rbar ← rbar; bad_rel ← 0
        else: bad_rel ← bad_rel + 1
        if bad_rel >= p_lr:
            η ← max(γη, min_lr); bad_rel ← 0     # 降 LR 重置 reliability 耐心

训练预算固定为配置中的 ``epochs``；本控制器不触发提前停止。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """控制器超参数。

    ``p_lr``：降学习率耐心。
    ``epsilon``：判定“严格改善”的最小幅度。
    ``control_warmup_epochs``：前若干 epoch 不累计 reliability 耐心
    （避开 RunningZ 首样本强制返回 0 的无效基线）。
    """

    gamma: float = 0.5
    min_lr: float = 1e-5
    p_lr: int = 8
    epsilon: float = 1e-4
    control_warmup_epochs: int = 2


class ReliabilityController:
    """跟踪平滑可靠性指标，并给出学习率控制动作。"""

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
        """消费一个 epoch 的可靠性观测，返回学习率决策及审计状态。

        ``val_loss`` 保留为可选参数以兼容旧调用点，但不参与控制。
        """

        del val_loss  # 早停已移除；验证损失不再驱动控制器
        lr_reduced = False

        eps = float(self.cfg.epsilon)
        rbar_f = float(rbar) if rbar is not None else float("nan")
        rel_active = int(epoch) > int(self.cfg.control_warmup_epochs) and self._finite(rbar_f)

        if rel_active:
            if not self._rbar_initialized:
                # 建基线：不计停滞，避免首个有效点被当成“已停滞 1”
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
