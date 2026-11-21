"""Centralize Python, NumPy, and PyTorch RNG setup."""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs and choose deterministic or high-performance cuDNN mode.

    Non-deterministic mode still fixes PRNG seeds, but cuDNN benchmark may pick
    different kernels; the config records this choice and the two modes are not
    treated as identical-protocol reproductions.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
