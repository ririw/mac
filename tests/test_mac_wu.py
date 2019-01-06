import nose.tools
import torch
import torch.functional
import torch.nn.functional

from mac.mac_cell import WUCell


def test_remember_everything():
    batch_size, ctrl_dim = 31, 7

    cell = WUCell(ctrl_dim)
    opt = torch.optim.Adam(cell.parameters())

    err = 10
    for i in range(1000):
        opt.zero_grad()
        mem = torch.randn(batch_size, ctrl_dim)
        ri = torch.randn(batch_size, ctrl_dim)
        control = torch.randn(batch_size, ctrl_dim)

        nextmem = cell(mem, ri, control)
        err = torch.nn.MSELoss()(nextmem, mem)
        err.backward()
        opt.step()
        err = err.item()

    nose.tools.assert_less(err, 0.1)


def test_remember_ri():
    batch_size, ctrl_dim = 31, 7

    cell = WUCell(ctrl_dim)
    opt = torch.optim.Adam(cell.parameters())

    err = 10
    for i in range(2000):
        opt.zero_grad()
        mem = torch.randn(batch_size, ctrl_dim)
        ri = torch.randn(batch_size, ctrl_dim, requires_grad=False)
        control = torch.randn(batch_size, ctrl_dim)

        nextmem = cell(mem, ri, control)
        err = torch.nn.MSELoss()(nextmem, ri)
        err.backward()
        opt.step()
        err = err.item()

    nose.tools.assert_less(err, 0.1)
