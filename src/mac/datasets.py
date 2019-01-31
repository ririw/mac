import json

import fs
import numpy as np
from torch.utils.data import Dataset, BatchSampler, RandomSampler

from mac import config


class MAC_JSON_Dataset(Dataset):
    def __init__(self, dataset_fs_url, clevr_fs_url, group_name):
        self.clevr_fs_url = clevr_fs_url
        self.group_name = group_name
        self.dataset_fs_url = dataset_fs_url

        self.image_handle = None
        self.images = None
        self.dataset_entries = None
        self.answer_mapping = config.getconfig()['answer_mapping']

    def __enter__(self):
        ds_fs = fs.open_fs(self.dataset_fs_url)
        clvr_fs = fs.open_fs(self.clevr_fs_url)

        size = 70000
        if self.group_name != 'train':
            size = 1000

        image_handle = ds_fs.open('{}/images'.format(self.group_name))
        self.images = np.memmap(
            image_handle, dtype='float32', mode='r',
            shape=(size, 1024, 14, 14))
        json_file_name = 'questions/CLEVR_{}_questions.json'.format(
            self.group_name)
        with clvr_fs.open(json_file_name) as f:
            self.dataset_entries = json.load(f)['questions']

        ds_fs.close()
        clvr_fs.close()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.images = None

    def __getitem__(self, index):
        try:
            iterator = iter(index)
        except TypeError:
            return self._get_single(index)

        answers, questions, images = [], [], []
        for ix in iterator:
            answer, question, image = self._get_single(ix)
            answers.append(answer)
            questions.append(question)
            images.append(image[None, :, :])

        return np.asanyarray(answers), questions, np.concatenate(images, 0)

    def _get_single(self, ix):
        entry = self.dataset_entries[ix]
        answer = self.answer_mapping[entry['answer']]
        image = self.images[entry['image_index']]
        question = entry['question']

        return answer, question, image

    def __len__(self):
        if self.dataset_entries is None:
            raise TypeError('Failed to use NP dataset with `with` context')

        return len(self.dataset_entries)


class MAC_NP_Dataset(Dataset):
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
        msg = 'Expected image min IX 0, got {}'.format(self.image_ix.min())
        assert self.image_ix.min() == 0, msg

        self.image = np.memmap(
            image_handle, dtype='float32', mode='r',
        ).reshape(-1, 1024, 14, 14)
        self.question = np.memmap(
            question_handle, dtype='float32', mode='r',
        ).rehsape(-1, 160, 256)

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

    def __getitem__(self, index):
        if self.answer is None:
            raise TypeError('Failed to use NP dataset with `with` context')
        answer = self.answer[index]
        image_ix = self.image_ix[index]
        question = self.question[index]
        image = self.image[image_ix]

        return answer, question, image_ix, image

    def __len__(self):
        if self.answer is None:
            raise TypeError('Failed to use NP dataset with `with` context')

        return self.answer.size


if __name__ == '__main__':
    from tqdm import tqdm

    hdf5_filename = '/Users/richardweiss/Datasets/results/decompressed_data'
    with MAC_NP_Dataset(hdf5_filename, 'val') as kb:
        for ixs in tqdm(BatchSampler(RandomSampler(kb), 32, True)):
            _ = [v.shape for v in kb[ixs]]
