import torch
from torch.nn.utils.rnn import pack_sequence

from torch.utils import data

from mac import config
import numpy as np
from mac.datasets import image_preprocess, qn_preprocess


class TaskDataset(data.Dataset):
    def __init__(self, clevr_fs, output_fs, split):
        self.images = image_preprocess.get_preprocess_images(clevr_fs, output_fs, split)
        if split == 'train':
            self.questions, word_ix = qn_preprocess.get_preprocess_questions(
                clevr_fs, output_fs, split)
        else:
            _, word_ix = qn_preprocess.get_preprocess_questions(clevr_fs, output_fs, 'train')
            self.questions, _ = qn_preprocess.get_preprocess_questions(
                clevr_fs, output_fs, split, word_ix)
        self.word_ix = word_ix

    def __getitem__(self, index):
        td = config.torch_device()
        img_ixs, answers, qns_ixs = [], [], []
        for ix in index:
            qn = self.questions[ix]
            img_ixs.append(qn.image_ix)
            answers.append(qn.answer_ix)
            qns_ixs.append(torch.from_numpy(np.array(qn.qn_ixs)).to(td))
        len_ord = np.argsort([len(q) for q in qns_ixs])[::-1]
        answers = np.asarray(answers)
        img_ixs = np.asarray(img_ixs)[len_ord]
        images = self.images[img_ixs]
        packed_qns = pack_sequence([qns_ixs[i] for i in len_ord])

        answers = torch.from_numpy(answers).to(td)
        images = torch.from_numpy(images).to(td)

        return images, packed_qns, answers

    def __len__(self):
        return self.images.shape[0]
