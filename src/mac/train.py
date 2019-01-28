import datetime
import os

import fs
import tensorboardX
import torch
import torch.nn.functional
from fs import open_fs
from plumbum import cli
import numpy as np
from torch.utils import data
from tqdm import tqdm

import mac.utils
from mac import datasets, mac, config, utils


@utils.MAC.subcommand('train')
class Train(cli.Application):
    batch_size = cli.SwitchAttr(
        ['-b', '--batch-size'], argtype=int, default=32)

    memorize = cli.Flag(
        ['-m', '--memorize'], help='Just learn one batch over and over')

    def main(self, preprocessed_loc, results_loc, log_loc):
        utils.cuda_message()
        np.printoptions(linewidth=139)

        mac_cell = mac.MACRec(6, 256)
        net = mac.MACNet(mac_cell)
        nowtime = str(datetime.datetime.now())
        writer = tensorboardX.SummaryWriter(os.path.join(log_loc, nowtime))
        results_fs = open_fs(results_loc, create=True)

        config.setconfig('summary_writer', writer)

        with datasets.MAC_NP_Dataset(preprocessed_loc, 'train') as train_ds:
            try:
                if self.memorize:
                    self.train_memorize(net, train_ds, writer, results_fs)
                else:
                    self.train(net, train_ds, writer, results_fs)
            except KeyboardInterrupt:
                pass

        preprocessed_fs = fs.open_fs(preprocessed_loc)
        with preprocessed_fs.open('net.pkl', 'wb') as f:
            torch.save(net, f)
        writer.close()

    def train_memorize(self, net, train_dataset, writer, results_fs):
        use_cuda = config.getconfig()['use_cuda']

        sampler = data.BatchSampler(
            data.RandomSampler(train_dataset), 32, False)
        opt = torch.optim.Adam(net.parameters())

        if use_cuda:
            net = net.cuda()

        ix = next(iter(sampler))
        for step in tqdm(range(1000)):
            self.train_step(ix, opt, use_cuda,
                            train_dataset, net, writer, step, results_fs)

    def train(self, net, train_dataset, writer, results_fs):
        use_cuda = config.getconfig()['use_cuda']

        sampler = data.BatchSampler(
            data.RandomSampler(train_dataset), self.batch_size, False)
        opt = torch.optim.Adam(net.parameters())

        if use_cuda:
            net = net.cuda()

        for step, ix in enumerate(tqdm(sampler)):
            self.train_step(ix, opt, use_cuda,
                            train_dataset, net,
                            writer, step, results_fs)

    def train_step(self, ix, opt, use_cuda,
                   train_dataset, net, writer,
                   step, results_fs):
        config.setconfig('step', step)

        opt.zero_grad()
        answer, question, image_ix, image = train_dataset[ix]
        answer = torch.Tensor(answer).long()
        question = torch.Tensor(question)
        image = torch.Tensor(image)

        if use_cuda:
            answer = answer.cuda()
            question = question.cuda()
            image = image.cuda()

        result = net.forward(image, question)
        loss = torch.nn.CrossEntropyLoss()(result, answer)
        loss.backward()

        opt.step()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(result, 1) == answer).float())
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('accuracy', accuracy.item(), step)

        if step % 50 == 0:
            with results_fs.open('model.pkl', 'wb') as f:
                torch.save({'net': net, 'opt': opt}, f)
