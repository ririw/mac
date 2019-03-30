import torch
import numpy as np
from src.mac.utils import some_fn


def test_simple() -> None:
    x = torch.ones(5, requires_grad=True)

    opt = torch.optim.LBFGS([x])

    def err() -> torch.Tensor:
        opt.zero_grad()
        r = some_fn(x)
        r.backward()
        return r

    opt.step(err)
    np.testing.assert_equal(x.detach().numpy().round(3), [0, 0, 0, 0, 0])
