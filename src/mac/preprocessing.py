"""
MAC input unit.
"""
import json
import os
import pickle
import re

import numpy as np
import skimage.io
import skimage.transform
import torch
from allennlp.modules import elmo
from fs.base import FS
from fs.walk import Walker
from torch import nn
from torch.utils import data
from torchvision.models import resnet101
from tqdm import tqdm

from mac import debug_helpers
from mac.config import getconfig

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
    output_fs.makedirs(name, recreate=True)
    output_file = output_fs.getsyspath('{}/images'.format(name))
    output_mat = np.memmap(
        output_file,
        np.float32,
        'w+',
        shape=result_size)
    bar = data.SequentialSampler(dataset)
    batched = data.BatchSampler(bar, batch_size, False)
    bar = tqdm(batched, disable=not progress,
               desc='Processing images - {}'.format(name))
    for ixs in bar:
        batch = dataset[ixs]
        with torch.no_grad():
            if getconfig()['use_cuda']:
                preprocessed = preproc_net(batch.cuda()).cpu()
            else:
                preprocessed = preproc_net(batch)
            output_mat[ixs] = preprocessed.numpy()


def extract_qn_dataset(name, dataset_fs):
    progress = getconfig()['progress']
    im_ixs, qn_texts, answer_texts = [], [], []

    json_file_name = 'questions/CLEVR_{}_questions.json'.format(name)
    work_limit = getconfig()['work_limit']
    with dataset_fs.open(json_file_name) as f:
        f_data = json.load(f)
        prog_bar = tqdm(f_data['questions'],
                        disable=not progress,
                        desc='Reading text data - {}'.format(name))
        for question in prog_bar:
            im_index = question['image_index']
            qn_text = question['question']
            answer_text = question['answer']
            im_ixs.append(im_index)
            qn_texts.append(qn_text)
            answer_texts.append(answer_text)
            if work_limit is not None and len(answer_texts) >= work_limit:
                break

    return im_ixs, qn_texts, answer_texts


def lang_preprocess(name, dataset_fs, output_fs,
                    batch_size=DEFAULT_BATCH_SIZE, max_len=160):
    im_ixs, qn_texts, answer_texts = extract_qn_dataset(name, dataset_fs)
    result_size = (len(qn_texts), max_len, 256)

    output_fs = output_fs.makedirs(name, recreate=True)
    save_im_ix(output_fs, im_ixs)
    save_answers(output_fs, answer_texts)
    save_questions(output_fs, name, qn_texts, max_len, result_size, batch_size)


def save_questions(output_fs, name, qn_texts,
                   max_len, result_size, batch_size):
    progress = getconfig()['progress']
    encoded_qn_ds = np.memmap(
        output_fs.getsyspath('question'), np.float32,
        'w+', shape=result_size
    )
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
                         desc='Processing questions - {}'.format(name)):
        with torch.no_grad():
            batch_qns = [q[:max_len] for q in qn_dataset[batch_ix]]
            batch_ids = elmo.batch_to_ids(batch_qns)
            if use_cuda:
                batch_ids = batch_ids.cuda()
            mdl_res = elmo_mdl(batch_ids)
            batch_encoded = mdl_res['elmo_representations'][1].cpu()
            debug_helpers.check_shape(
                batch_encoded, (len(batch_ix), None, 256))
            seq_len = batch_encoded.shape[1]

            encoded_qn_ds[batch_ix, :seq_len] = batch_encoded.numpy()


def save_answers(output_fs, answer_texts):
    answer_mapping = getconfig()['answer_mapping']
    answer_ixs = [answer_mapping[ans] for ans in answer_texts]
    del answer_texts
    answer_ixs_ds = np.memmap(
        output_fs.getsyspath('answer'),
        np.int32, 'w+', shape=len(answer_ixs)
    )
    answer_ixs_ds[:] = answer_ixs


def save_im_ix(output_fs, im_ixs):
    img_ix_ds = np.memmap(
        output_fs.getsyspath('img_ix'),
        np.int32, 'w+', shape=len(im_ixs)
    )
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
    def __init__(self, clevr_fs: FS, work_limit=None):
        self.clevr_fs = clevr_fs
        self.images = None
        self.img_size = (224, 224)
        self._crawl()
        self.work_limit = work_limit

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
        if self.work_limit is not None:
            return min(self.work_limit, len(self.images))
        return len(self.images)


def preprocess_questions(dataset_name, input_fs, output_fs):
    fn = '/questions/CLEVR_{}_questions.json'.format(dataset_name)
    with input_fs.open(fn, 'r') as f:
        ds = json.load(f)

    result_data = []
    answer_map = getconfig()['answer_mapping']
    for qn in ds['questions']:
        image = qn['image_index']
        answer_text = qn['answer']
        answer = answer_map[answer_text]
        question = qn['question'].lower().replace('?', '').replace(';', '')
        assert re.match('^[a-z ]*$', question), 'Mismatch: ' + question
        result_data.append({
            'question': question,
            'image': image,
            'answer': answer,
            'answer_text': answer_text,
        })

    with output_fs.open('dataset_{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(result_data, f)
