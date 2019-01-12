import fs
import h5py
import numpy as np
from torch.utils.data import Dataset, BatchSampler, RandomSampler


class MAC_H5_Dataset(Dataset):
    def __init__(self, hdf5_filename, group_name):
        self.hdf5_filename = hdf5_filename
        self.group_name = group_name

        self.file = None
        self.group = None

        self.answer = None
        self.question = None
        self.image_ix = None
        self.image = None

    def __enter__(self):
        self.file = h5py.File(self.hdf5_filename, 'r')
        self.group = self.file[self.group_name]

        self.answer = self.group['answer']
        self.question = self.group['question']
        self.image_ix = self.group['img_ix']
        self.image = self.group['images']
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __getitem__(self, index):
        index = sorted(index)
        answer = self.answer[index]
        question = self.question[index]
        image_ix = self.image_ix[index]
        image = self.image[image_ix]

        return answer, question, image_ix, image

    def __len__(self):
        return len(self.answer)


class MAC_NP_Dataset(Dataset):
    def __getitem__(self, index):
        answer = self.answer[index]
        image_ix = self.image_ix[index]
        question = self.question[index]
        image = self.image[image_ix]

        return answer, question, image_ix, image

    def __len__(self):
        return self.answer.size

    def __init__(self, dataset_fs_url, group_name):
        self.dataset_fs_url = dataset_fs_url
        self.group_name = group_name

        self.fs = None

        self.image = None
        self.image_ix = None
        self.answer = None
        self.question_ix = None

        self.open_handles = []

    def __enter__(self):
        self.fs = fs.open_fs(self.dataset_fs_url)

        image_handle = self.fs.open('{}/images'.format(self.group_name))
        image_ix_handle = self.fs.open('{}/img_ix'.format(self.group_name))
        question_handle = self.fs.open('{}/question'.format(self.group_name))
        answer_handle = self.fs.open('{}/answer'.format(self.group_name))

        self.image_ix = np.memmap(
            image_ix_handle, dtype='int32', mode='r')
        self.answer = np.memmap(
            answer_handle, dtype='int32', mode='r')
        dataset_size = self.image_ix.size

        self.image = np.memmap(
            image_handle, dtype='float16', mode='r',
            shape=(dataset_size, 1024, 14, 14))
        self.question = np.memmap(
            question_handle, dtype='float16', mode='r',
            shape=(dataset_size, 160, 256))

        self.open_handles = [image_handle, image_ix_handle,
                             question_handle, answer_handle]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.image_ix = None
        self.answer = None
        self.image = None
        self.question = None
        for handle in self.open_handles:
            handle.close()
        self.fs.close()


if __name__ == '__main__':
    from tqdm import tqdm
    hdf5_filename = '/Users/richardweiss/Datasets/results/decompressed_data'
    with MAC_NP_Dataset(hdf5_filename, 'val') as kb:
        for ixs in tqdm(BatchSampler(RandomSampler(kb), 32, True)):
            _ = [v.shape for v in kb[ixs]]
