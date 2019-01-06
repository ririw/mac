import torch
import torch.functional
import torch.nn.functional
from mac.mac_cell import CUCell, __debug__options__
import nose.tools


def test_simple_control_run():
    batch_size, seq_len, ctrl_dim = 17, 31, 7
    prev_ctrl = torch.randn(batch_size, ctrl_dim)
    context_words = torch.randn(batch_size, seq_len, ctrl_dim)
    question_words = torch.randn(batch_size, ctrl_dim)

    cell = CUCell(ctrl_dim)
    cell(prev_ctrl, context_words, question_words)


def test_simple_control_leakage():
    batch_size, seq_len, ctrl_dim = 17, 31, 7
    prev_ctrl = torch.randn(batch_size, ctrl_dim)
    context_words = torch.randn(batch_size, seq_len, ctrl_dim)
    question_words = torch.randn(batch_size, ctrl_dim)

    context_words[:, :, 0] = 0

    cell = CUCell(ctrl_dim)
    next_ctrl = cell(prev_ctrl, context_words, question_words)
    nose.tools.assert_equal(next_ctrl[:, 0].max().item(), 0)
    nose.tools.assert_equal(next_ctrl[:, 0].min().item(), 0)
    nose.tools.assert_equal(next_ctrl.shape, (batch_size, ctrl_dim))


def test_variable_batches():
    batch_size, seq_len, ctrl_dim = 11, 13, 7
    cell = CUCell(ctrl_dim)

    prev_ctrl = torch.randn(batch_size, ctrl_dim)
    context_words = torch.randn(batch_size, seq_len, ctrl_dim)
    question_words = torch.randn(batch_size, ctrl_dim)
    next_ctrl = cell(prev_ctrl, context_words, question_words)
    nose.tools.assert_equal(next_ctrl.shape, (batch_size, ctrl_dim))

    batch_size = 23
    prev_ctrl = torch.randn(batch_size, ctrl_dim)
    context_words = torch.randn(batch_size, seq_len, ctrl_dim)
    question_words = torch.randn(batch_size, ctrl_dim)
    next_ctrl = cell(prev_ctrl, context_words, question_words)
    nose.tools.assert_equal(next_ctrl.shape, (batch_size, ctrl_dim))


def test_silly_train():
    __debug__options__['save_locals'] = True
    batch_size, seq_len, ctrl_dim = 63, 13, 11
    cell = CUCell(ctrl_dim)
    opt = torch.optim.Adam(cell.parameters())

    err = 100
    for i in range(1000):
        opt.zero_grad()
        prev_ctrl = torch.randn(batch_size, ctrl_dim)
        context_words = torch.zeros(batch_size, seq_len, ctrl_dim)
        question_words = torch.zeros(batch_size, ctrl_dim)
        target = torch.zeros(batch_size, ctrl_dim, requires_grad=False)
        for j in range(batch_size):
            question_words[j, -j % ctrl_dim] = 1
            context_words[j, j % seq_len, j % ctrl_dim] = 1
            target[j, j % ctrl_dim] = 1

        next_ctrl = cell(prev_ctrl, context_words, question_words)

        assert next_ctrl.shape == target.shape
        err = torch.nn.MSELoss()(next_ctrl, target)
        err.backward()
        opt.step()
        err = err.item()
    nose.tools.assert_less(err, 0.1)


def test_multi_lin():
    batch_size, seq_len, ctrl_dim = 31, 31, 11
    a = torch.nn.Linear(ctrl_dim, 1, bias=False)
    opt = torch.optim.Adam(a.parameters())
    x = torch.randn(batch_size, seq_len, ctrl_dim)

    err = 100
    for i in range(2000):
        opt.zero_grad()
        b = a(x).squeeze(2)
        target = x[:, :, 0]
        err = torch.nn.MSELoss()(b, target)
        err.backward()
        opt.step()

        err = err.item()
    nose.tools.assert_less(err, 0.1)
