import numpy as np
import torch
import torch.utils.data
import fs.tempfs

from mac import inputs


def test_input_trf():
    batch_size = 9
    input_data = torch.randn(batch_size, 3, 224, 224)
    input_dataset = ConcatDataset(input_data)

    with fs.tempfs.TempFS() as output_fs:
        inputs.image_preprocess(
            'dummy', input_dataset, output_fs,
            batch_size=4, progress=False)

        assert output_fs.exists('dummy-mmap.dat')
        dummy_path = output_fs.getsyspath('dummy-mmap.dat')
        res = np.memmap(dummy_path, mode='r', dtype=np.float32)
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
