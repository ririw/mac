import unittest.mock
import torch

from mac import mac


def test_can_run():
    batch_size, seq_len, ctrl_dim = 3, 11, 7
    mock_mac = unittest.mock.Mock()
    mock_mac.ctrl_dim = ctrl_dim

    macnet = mac.MACNet(mock_mac)
    kb_image = torch.randn(batch_size, 1024, 14, 14)
    questions = torch.randn(batch_size, seq_len, 256)

    macnet(kb_image, questions)
