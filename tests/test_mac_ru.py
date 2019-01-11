import torch
import torch.functional
import torch.nn.functional
from mac.mac_cell import RUCell
from mac.debug_helpers import get_saved_locals, enable_debug
import nose.tools
import numpy as np


def test_direct_inter():
    enable_debug()
    batch_size, ctrl_dim = 63, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, 14, 14, ctrl_dim)
    ctrl = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.LBFGS(ru.parameters(), max_iter=100)

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    def err_cb():
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['direct_inter'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        return err

    opt.step(err_cb)
    nose.tools.assert_less(err_cb().item(), 0.1)


def test_second_inter():
    enable_debug()
    batch_size, ctrl_dim = 31, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, 14, 14, ctrl_dim)
    ctrl = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['second_inter'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)


def test_weighted_control():
    enable_debug()
    batch_size, ctrl_dim = 31, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, 14, 14, ctrl_dim)
    ctrl = torch.ones(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['weighted_control'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)


def test_ra():
    enable_debug()
    batch_size, ctrl_dim = 31, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, 14, 14, ctrl_dim)
    ctrl = torch.ones(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['ra'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)
