import torch
import torch.functional
import torch.nn.functional
from mac.mac_cell import RUCell, __debug__options__
import nose.tools
import numpy as np


def test_direct_inter():
    __debug__options__['save_locals'] = True
    batch_size, ctrl_dim = 63, 17
    a = torch.zeros(batch_size, ctrl_dim)
    b = torch.zeros(batch_size, ctrl_dim)
    c = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.LBFGS(ru.parameters(), max_iter=100)

    for i in range(batch_size):
        a[i, i % ctrl_dim] = 1
        b[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    def err_cb():
        opt.zero_grad()
        ru(a, b, c)
        v = __debug__options__['locals']['direct_inter']
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        return err

    opt.step(err_cb)
    nose.tools.assert_less(err_cb().item(), 0.1)


def test_second_inter():
    __debug__options__['save_locals'] = True
    batch_size, ctrl_dim = 31, 17
    a = torch.zeros(batch_size, ctrl_dim)
    b = torch.zeros(batch_size, ctrl_dim)
    c = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        a[i, i % ctrl_dim] = 1
        b[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(a, b, c)
        v = __debug__options__['locals']['second_inter']
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)


def test_weighted_control():
    __debug__options__['save_locals'] = True
    batch_size, ctrl_dim = 31, 17
    a = torch.zeros(batch_size, ctrl_dim)
    b = torch.zeros(batch_size, ctrl_dim)
    c = torch.ones(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        a[i, i % ctrl_dim] = 1
        b[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(a, b, c)
        v = __debug__options__['locals']['weighted_control']
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)


def test_ra():
    __debug__options__['save_locals'] = True
    batch_size, ctrl_dim = 31, 17
    a = torch.zeros(batch_size, ctrl_dim)
    b = torch.zeros(batch_size, ctrl_dim)
    c = torch.ones(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        a[i, i % ctrl_dim] = 1
        b[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(a, b, c)
        v = __debug__options__['locals']['ra']
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)