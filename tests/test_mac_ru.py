import torch
import torch.functional
import torch.nn.functional
from mac.mac import RUCell
from mac.debug_helpers import get_saved_locals, enable_debug
import nose.tools
import numpy as np


def est_direct_inter():
    enable_debug()
    batch_size, ctrl_dim = 63, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, ctrl_dim, 14, 14)
    ctrl = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim*2)

    ru = RUCell(ctrl_dim)

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i, :, 5, 5] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    def err_cb():
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['mem_kb_inter_cat'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        return err
    opt = torch.optim.Adam(ru.parameters(), max_iter=100)
    opt.step(err_cb)
    nose.tools.assert_less(err_cb().item(), 0.1)


def est_second_inter():
    enable_debug()
    batch_size, ctrl_dim = 31, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, ctrl_dim, 14, 14)
    ctrl = torch.zeros(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i, :, 5, 5] = torch.tensor(np.arange(ctrl_dim))
        target[i, :] = i % ctrl_dim

    for i in range(500):
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['second_inter'][:, 0, 0, :]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.1)


def est_weighted_control():
    enable_debug()
    batch_size, ctrl_dim = 31, 17
    mem = torch.zeros(batch_size, ctrl_dim)
    kb = torch.zeros(batch_size, ctrl_dim, 14, 14)
    ctrl = torch.ones(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, ctrl_dim)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, i % ctrl_dim] = 1
        kb[i, :, 5, 5] = torch.tensor(np.arange(ctrl_dim))
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
    mem = torch.rand(batch_size, ctrl_dim)
    kb = torch.rand(batch_size, ctrl_dim, 14, 14)
    ctrl = torch.rand(batch_size, ctrl_dim)
    target = torch.zeros(batch_size, 14, 14)

    ru = RUCell(ctrl_dim)
    opt = torch.optim.Adam(ru.parameters())

    for i in range(batch_size):
        mem[i, :] = kb[i, :, i % 14, i % 14]
        ctrl[i, :] = kb[i, :, i % 14, i % 14]
        target[i, i % 14, i % 14] = 1

    for i in range(500):
        opt.zero_grad()
        ru(mem, kb, ctrl)
        v = get_saved_locals()['attended'][:]
        err = torch.nn.functional.mse_loss(v, target)
        err.backward()
        opt.step()

    nose.tools.assert_less(err.item(), 0.002)
