"""
MAC input unit.
"""
import json
import os

import numpy as np
from fs.base import FS
from fs.walk import Walker

from mac import debug_helpers
from mac.config import getconfig
import skimage.io
import h5py
import skimage.transform
from allennlp.modules import elmo

import torch
from torch import nn
from torch.utils import data
from torchvision.models import resnet101
from tqdm import tqdm

D = 512
DEFAULT_BATCH_SIZE = 128


def image_preprocess(name, dataset, output_fs, batch_size=DEFAULT_BATCH_SIZE):
    progress = getconfig()['progress']
    resnet = resnet101(True)
    preproc_net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )

    sample = dataset[0:3]
    if getconfig()['use_cuda']:
        preproc_net = preproc_net.cuda()
        sample = sample.cuda()

    result_size = (len(dataset),) + preproc_net(sample).shape[1:]
    output_file = output_fs.getsyspath('data.h5')
    output_h5 = h5py.File(output_file)
    group = output_h5.require_group(name)
    res_file = group.require_dataset(
        'images', result_size, dtype='float32', compression="gzip")
    bar = data.SequentialSampler(dataset)
    batched = data.BatchSampler(bar, batch_size, False)
    bar = tqdm(batched, disable=not progress, desc='Processing images')
    for ixs in bar:
        batch = dataset[ixs]
        with torch.no_grad():
            if getconfig()['use_cuda']:
                preprocessed = preproc_net(batch.cuda()).cpu()
            else:
                preprocessed = preproc_net(batch)
            res_file[ixs] = preprocessed.numpy()
    output_h5.close()


def extract_qn_dataset(name, dataset_fs):
    progress = getconfig()['progress']
    im_ixs, qn_texts, answer_texts = [], [], []

    json_file_name = 'questions/CLEVR_{}_questions.json'.format(name)
    with dataset_fs.open(json_file_name) as f:
        f_data = json.load(f)
        prog_bar = tqdm(f_data['questions'],
                        disable=not progress,
                        desc='Reading text data')
        for question in prog_bar:
            im_index = question['image_index']
            qn_text = question['question']
            answer_text = question['answer']
            im_ixs.append(im_index)
            qn_texts.append(qn_text)
            answer_texts.append(answer_text)
    return im_ixs, qn_texts, answer_texts


def lang_preprocess(name, dataset_fs, output_fs,
                    batch_size=DEFAULT_BATCH_SIZE, max_len=160):
    im_ixs, qn_texts, answer_texts = extract_qn_dataset(name, dataset_fs)
    result_size = (len(qn_texts), max_len, 256)

    output_file = output_fs.getsyspath('data.h5')
    output_h5 = h5py.File(output_file)
    group = output_h5.require_group(name)

    save_im_ix(group, im_ixs)
    save_answers(answer_texts, group)
    save_questions(group, qn_texts, max_len, result_size, batch_size)

    output_h5.close()
    output_file.close()


def save_questions(group, qn_texts, max_len, result_size, batch_size):
    progress = getconfig()['progress']
    encoded_qn_ds = group.require_dataset(
        'question', result_size,
        dtype='float32', compression="gzip")
    elmo_options_file = os.path.expanduser(
        '~/Datasets/elmo_small_options.json')
    elmo_weights_file = os.path.expanduser(
        '~/Datasets/elmo_small_weights.hdf5')
    elmo_mdl = elmo.Elmo(elmo_options_file, elmo_weights_file, 2)
    use_cuda = getconfig()['use_cuda']
    if use_cuda:
        elmo_mdl = elmo_mdl.cuda()
    qn_dataset = CLEVRQuestionData(qn_texts)
    qn_sampler = data.BatchSampler(
        data.SequentialSampler(qn_dataset), batch_size, False)
    for batch_ix in tqdm(qn_sampler,
                         disable=not progress,
                         desc='Processing questions'):
        with torch.no_grad():
            batch_qns = [q[:max_len] for q in qn_dataset[batch_ix]]
            batch_ids = elmo.batch_to_ids(batch_qns)
            if use_cuda:
                batch_ids = batch_ids.cuda()
            mdl_res = elmo_mdl(batch_ids)
            batch_encoded = mdl_res['elmo_representations'][1].cpu()
            debug_helpers.check_shape(batch_encoded, (batch_size, None, 256))
            seq_len = batch_encoded.shape[1]

            encoded_qn_ds[batch_ix, :seq_len] = batch_encoded.numpy()


def save_answers(answer_texts, group):
    answer_mapping = getconfig()['answer_mapping']
    answer_ixs = [answer_mapping[ans] for ans in answer_texts]
    del answer_texts
    answer_ixs_ds = group.require_dataset(
        'answer', (len(answer_ixs),),
        dtype='int32', compression="gzip")
    answer_ixs_ds[:] = answer_ixs


def save_im_ix(group, im_ixs):
    img_ix_ds = group.require_dataset(
        'img_ix', (len(im_ixs),),
        dtype='int32', compression="gzip")
    img_ix_ds[:] = im_ixs
    del im_ixs


class CLEVRQuestionData(data.Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __getitem__(self, index):
        try:
            res = []
            for i in index:
                res.append(self.questions[i])
            return res
        except TypeError:
            return self.questions[index]

    def __len__(self):
        return len(self.questions)


class CLEVRImageData(data.Dataset):
    def __init__(self, clevr_fs: FS):
        self.clevr_fs = clevr_fs
        self.images = None
        self.img_size = (224, 224)
        self._crawl()

    def _crawl(self):
        self.images = {}

        walker = Walker(filter=['*.png'])
        for path in tqdm(walker.files(self.clevr_fs), 'Crawling...'):
            idx = int(os.path.splitext(path)[0].split('_')[-1])
            self.images[idx] = path

    def __getitem__(self, index):
        if isinstance(index, list):
            res = []
            for ix in index:
                res.append(self._read_image(ix)[None, :, :, :])
            res = np.concatenate(res, 0)
            return torch.from_numpy(res)
        if isinstance(index, slice):
            if index.start is None:
                iterator = range(index.stop)
            elif index.step is None:
                iterator = range(index.start, index.stop)
            else:
                iterator = range(index.start, index.step, index.stop)

            res = []
            for ix in iterator:
                res.append(self._read_image(ix)[None, :, :, :])
            res = np.concatenate(res, 0)
            return torch.from_numpy(res)
        else:
            return torch.from_numpy(self._read_image(index))

    def _read_image(self, ix):
        path = self.images[ix]
        with self.clevr_fs.open(path, 'rb') as f:
            img = skimage.io.imread(f)
            # RGBA -> remove the A
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = skimage.transform.resize(img, self.img_size)
            return img.transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.images)
