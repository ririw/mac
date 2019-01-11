import torch

from mac import mac


def test_can_run():
    batch_size = 8
    macnet = mac.MACNet()
    kb_image = torch.randn(batch_size, 1024, 14, 14)

    macnet(kb_image)
