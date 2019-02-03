import nose.tools
import numpy as np
import torch
import torch.functional
import torch.nn.functional

from mac.mac_cell import OutputCell


def test_simple_case():
    batch_size, ctrl_dim = 31, 8

    cell = OutputCell(ctrl_dim)
    opt = torch.optim.Adam(cell.parameters())

    err = 10
    for i in range(1000):
        opt.zero_grad()
        mem = torch.randn(batch_size, ctrl_dim)
        control = torch.randn(batch_size, ctrl_dim)
        target = torch.zeros(batch_size, 28)
        target[np.arange(batch_size), torch.argmax(control, 1)] = 1

        out = cell(control, mem)
        err = torch.nn.MSELoss()(out, target)
        err.backward()
        opt.step()
        err = err.item()
    nose.tools.assert_less(err, 0.05)
