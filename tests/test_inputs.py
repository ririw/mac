from contextlib import closing

import h5py
import numpy as np
import torch
import torch.utils.data
import fs.tempfs

from mac import inputs
from mac.config import getconfig


def test_input_trf():
    config = getconfig()
    config['progress'] = False
    batch_size = 9
    input_data = torch.randn(batch_size, 3, 224, 224)
    input_dataset = ConcatDataset(input_data)

    with fs.tempfs.TempFS() as output_fs:
        inputs.image_preprocess(
            'dummy', input_dataset, output_fs,
            batch_size=4)

        assert output_fs.exists('data.h5')
        f = output_fs.getsyspath('data.h5')
        with closing(h5py.File(f)) as f5:
            res = f5['dummy']['images']
            assert res.size == np.product([9, 1024, 14, 14])


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_ds):
        self.underlying_ds = underlying_ds

    def __getitem__(self, index):
        res = self.underlying_ds[index]
        if isinstance(res, tuple):
            res = torch.stack(res)
        return res

    def __len__(self):
        return len(self.underlying_ds)
