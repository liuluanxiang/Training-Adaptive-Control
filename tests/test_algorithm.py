import unittest
import torch
from src.controller import ControllerConfig, ReliabilityController
from src.signals import SignalConfig, ReliabilitySignals, RunningZ


class ControllerTests(unittest.TestCase):
    def test_reliability_patience_reduces_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        controller = ReliabilityController(
            ControllerConfig(gamma=0.5, min_lr=1e-5, p_lr=2, epsilon=1e-4, control_warmup_epochs=0)
        )
        action = None
        for epoch in range(1, 12):
            action = controller.step(optimizer, rbar=1.0, epoch=epoch)
            self.assertNotIn("early_stop", action)
        self.assertEqual(action["lr_reductions"], 5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.003125)

    def test_warmup_skips_invalid_zero_baseline(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        controller = ReliabilityController(ControllerConfig(control_warmup_epochs=2, p_lr=8))
        for epoch in (1, 2):
            action = controller.step(optimizer, rbar=0.0, epoch=epoch)
            self.assertFalse(action["rel_active"])
            self.assertFalse(action["rbar_initialized"])
        action = controller.step(optimizer, rbar=0.5, epoch=3)
        self.assertTrue(action["rel_active"])
        self.assertTrue(action["rbar_initialized"])


class SignalTests(unittest.TestCase):
    def test_running_z_first_sample_is_zero(self):
        z = RunningZ()
        self.assertEqual(z.update(3.0), 0.0)
        second = z.update(5.0)
        self.assertNotEqual(second, 0.0)

    def test_signal_norms(self):
        model = torch.nn.Linear(2, 1, bias=False)
        signals = ReliabilitySignals(model, n_train=10, cfg=SignalConfig(prior_sigma=0.5))
        signals.begin_epoch(model)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        self.assertAlmostEqual(signals.gradient_norm(model), 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
