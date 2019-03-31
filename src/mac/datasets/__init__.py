import numpy as np
import torch
from torch.utils import data

from mac import config
from mac.datasets import image_preprocess, qn_preprocess


# noinspection PyUnresolvedReferences
class TaskDataset(data.Dataset):
    def __init__(self, clevr_fs, output_fs, split):
        self.images = image_preprocess.get_preprocess_images(clevr_fs, output_fs, split)
        if split == "train":
            self.questions, word_ix = qn_preprocess.get_preprocess_questions(
                clevr_fs, output_fs, split
            )
        else:
            _, word_ix = qn_preprocess.get_preprocess_questions(
                clevr_fs, output_fs, split
            )
            self.questions, _ = qn_preprocess.get_preprocess_questions(
                clevr_fs, output_fs, split, word_ix
            )
        self.word_ix = word_ix

    def __getitem__(self, index):
        td = config.torch_device()
        img_ixs, answers, qns_ixs = [], [], []
        for ix in index:
            qn = self.questions[ix]
            img_ixs.append(qn.image_ix)
            answers.append(qn.answer_ix)
            qns_ixs.append(qn.qn_ixs)

        len_ord = np.argsort([len(q) for q in qns_ixs])[::-1]
        answers = np.asarray(answers)[len_ord]
        img_ixs = np.asarray(img_ixs)[len_ord]
        images = self.images[img_ixs]
        ordered_qns = [qns_ixs[i] for i in len_ord]

        qn_lengths = np.asarray([len(q) for q in ordered_qns], np.int64)
        qn_words = np.zeros((len(answers), max(qn_lengths)), np.int64)
        for ix, q in enumerate(ordered_qns):
            qn_words[ix, : qn_lengths[ix]] = q

        answers = torch.from_numpy(answers).to(td)
        images = torch.from_numpy(images).to(td)
        qns = torch.from_numpy(qn_words).to(td)
        qn_lens = torch.from_numpy(qn_lengths).to(td)

        return images, qns, qn_lens, answers

    def __len__(self):
        return len(self.questions)
