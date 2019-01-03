"""
MAC input unit.
"""
import os

import numpy as np
from fs.base import FS
from fs.walk import Walker
from mac.config import getconfig
import skimage.io
import skimage.transform
import torch
from torch import nn
from torch.utils import data
from torchvision.models import resnet101
from tqdm import tqdm

D = 512


def image_preprocess(name, dataset, output_fs):
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
    res_file = np.memmap(
        output_fs.getsyspath('{}-mmap.dat'.format(name)),
        dtype=np.float32,
        mode='w+',
        shape=result_size
    )
    try:
        bar = data.SequentialSampler(dataset)
        batched = data.BatchSampler(bar, 32, False)
        bar = tqdm(batched, total=len(batched))
        for ixs in bar:
            batch = dataset[ixs]
            with torch.no_grad():
                if getconfig()['use_cuda']:
                    preprocessed = preproc_net(batch.cuda()).cpu()
                else:
                    preprocessed = preproc_net(batch)
                res_file[ixs] = preprocessed.numpy()
    finally:
        del res_file


class CLEVRData(data.Dataset):
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
