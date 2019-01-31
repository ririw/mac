import torch
import nose.tools

import mac.mac


def test_runs():
    batch_size, recurrence_length, ctrl_dim, seq_len = 7, 11, 5, 13
    question_words = torch.rand(batch_size, ctrl_dim)
    image_vec = torch.rand(batch_size, 14, 14, ctrl_dim)
    context_words = torch.rand(batch_size, seq_len, ctrl_dim)
    cell = mac.mac.MACRec(recurrence_length, ctrl_dim)
    res = cell(question_words, image_vec, context_words)

    nose.tools.assert_equal(res.shape, (7, 28))
