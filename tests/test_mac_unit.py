import torch
import nose.tools

import mac.mac


def test_runs():
    batch_size, recurrence_length, ctrl_dim, seq_len = 7, 11, 5, 13
    question_words = torch.rand(batch_size, ctrl_dim)
    image_vec = torch.rand(batch_size, 14, 14, ctrl_dim)
    context_words = torch.rand(batch_size, seq_len, ctrl_dim)
    cell = mac.mac.MAC(recurrence_length, ctrl_dim)
    res = cell(question_words, image_vec, context_words)

    nose.tools.assert_equal(res.shape, (7, 28))


def test_params():
    recurrence_length, ctrl_dim = 11, 5
    cell = mac.mac.MAC(recurrence_length, ctrl_dim)

    named_params = [n for n, t in cell.named_parameters()]
    nose.tools.assert_in('control_9.ca_lin.bias', named_params)
    nose.tools.assert_in('control_0.cq_lin.bias', named_params)
    nose.tools.assert_in('write_10.mem_read_int.weight', named_params)
    nose.tools.assert_in('read_0.kb1_lin.bias', named_params)
    nose.tools.assert_in('initial_control', named_params)
    nose.tools.assert_in('output_cell.layers.2.weight', named_params)
