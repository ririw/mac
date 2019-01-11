import unittest.mock
import torch

from mac import mac


def test_can_run():
    batch_size = 3
    mock_mac = unittest.mock.Mock()
    mock_mac.ctrl_dim = 7

    macnet = mac.MACNet(mock_mac)
    kb_image = torch.randn(batch_size, 1024, 14, 14)

    macnet(kb_image, [
        'who dat boi'.split(' '),
        'golf wang'.split(' '),
        'Boredom Boredom Boredom Boredom Boredom'.split(' ')])
