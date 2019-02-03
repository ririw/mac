import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def test_embed():
    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 0, 0]
    ]).T
    x = torch.from_numpy(x)
    a = pack_padded_sequence(x, [4, 2])
    print(len(a))
    b, lens = pad_packed_sequence(a, batch_first=True)

    emb = torch.nn.Embedding(11, 7)
    c = emb(b)
    print(c.shape)
