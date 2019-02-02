import fs.tempfs
import numpy as np
import torch
import torch.utils.data

from mac import preprocessing
from mac.config import getconfig


def test_input_trf():
    config = getconfig()
    config['progress'] = False
    batch_size = 9
    input_data = torch.randn(batch_size, 3, 224, 224)
    input_dataset = ConcatDataset(input_data)

    with fs.tempfs.TempFS() as output_fs:
        preprocessing.image_preprocess(
            'dummy', input_dataset, output_fs,
            batch_size=4)

        assert output_fs.exists('dummy/images')
        f = output_fs.getsyspath('dummy/images')
        ds = np.memmap(f, 'float32', 'r')
        assert ds.size == np.product([9, 1024, 14, 14])


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
