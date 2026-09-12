import random

import numpy as np
import torch

from explainable_kd.common.seed import seed_everything


def test_seed_everything_reproduces_python_numpy_and_torch():
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(2))

    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(2))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
