import unittest.mock
import torch
import numpy as np

from mac import mac


def test_can_run():
    batch_size, seq_len, ctrl_dim = 3, 11, 8
    mock_mac = unittest.mock.Mock()
    mock_mac.ctrl_dim = ctrl_dim

    macnet = mac.MACNet(mock_mac)
    kb_image = torch.randn(batch_size, 1024, 14, 14)
    questions = torch.randn(batch_size, seq_len, 256)

    macnet(kb_image, questions)


def test_nan_poison_question():
    batch_size, seq_len, ctrl_dim = 3, 11, 8

    mac_cell = mac.MACRec(4, ctrl_dim)
    macnet = mac.MACNet(mac_cell)
    kb_image = torch.randn(batch_size, 1024, 14, 14)
    questions = torch.randn(batch_size, seq_len, 256)

    questions[0] = np.NaN
    with torch.no_grad():
        res = macnet(kb_image, questions).cpu().numpy()
    is_nan = np.isnan(res)
    assert np.all(is_nan, 1)[0]
    assert ~np.any(is_nan, 1)[1]
    assert ~np.any(is_nan, 1)[2]


def test_nan_poison_image():
    batch_size, seq_len, ctrl_dim = 3, 11, 8

    mac_cell = mac.MACRec(4, ctrl_dim)
    macnet = mac.MACNet(mac_cell)
    kb_image = torch.randn(batch_size, 1024, 14, 14)
    questions = torch.randn(batch_size, seq_len, 256)

    kb_image[0] = np.NaN
    with torch.no_grad():
        res = macnet(kb_image, questions).cpu().numpy()
    is_nan = np.isnan(res)
    assert np.all(is_nan, 1)[0]
    assert ~np.any(is_nan, 1)[1]
    assert ~np.any(is_nan, 1)[2]
