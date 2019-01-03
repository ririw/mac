import torch
from src.mac.cli import some_fn


def test_simple():
    x = torch.ones(5, requires_grad=True)

    opt = torch.optim.LBFGS([x])

    def err():
        opt.zero_grad()
        r = some_fn(x)
        r.backward()
        return r

    opt.step(err)
    print('Expect: [-0. -0. -0. -0. -0.]')
    print(x.detach().numpy().round(3))