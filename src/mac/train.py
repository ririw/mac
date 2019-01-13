import datetime

import fs
import tensorboardX
import torch
from plumbum import cli
from torch.utils import data

import mac.utils
from mac import datasets, mac, config, utils


@utils.MAC.subcommand('train')
class Train(cli.Application):
    batch_size = cli.SwitchAttr(
        ['-b', '--batch-size'], argtype=int, default=32)

    memorize = cli.Flag(
        ['-m', '--memorize'], help='Just learn one batch over and over')

    def main(self, preprocessed_loc, log_loc):
        utils.cuda_message()

        mac_cell = mac.MAC(12, 512)
        net = mac.MACNet(mac_cell)
        comment = str(datetime.datetime.now())
        writer = tensorboardX.SummaryWriter(log_loc, comment)

        with datasets.MAC_NP_Dataset(preprocessed_loc, 'train') as train_ds:
            try:
                if self.memorize:
                    self.train_memorize(net, train_ds, writer)
                else:
                    self.train(net, train_ds, writer)
            except KeyboardInterrupt:
                pass

        preprocessed_fs = fs.open_fs(preprocessed_loc)
        with preprocessed_fs.open('net.pkl', 'wb') as f:
            torch.save(net, f)
        writer.close()

    def train_memorize(self, net, train_dataset, writer):
        use_cuda = config.getconfig()['use_cuda']

        sampler = data.BatchSampler(
            data.RandomSampler(train_dataset), 32, False)
        opt = torch.optim.Adam(net.parameters())

        if use_cuda:
            net = net.cuda()

        ix = next(iter(sampler))
        for step in range(1000):
            self.train_step(ix, opt, use_cuda,
                            train_dataset, net, writer, step)

    def train(self, net, train_dataset, writer):
        use_cuda = config.getconfig()['use_cuda']

        sampler = data.BatchSampler(
            data.RandomSampler(train_dataset), self.batch_size, False)
        opt = torch.optim.Adam(net.parameters())

        if use_cuda:
            net = net.cuda()

        for step, ix in enumerate(sampler):
            self.train_step(ix, opt, use_cuda,
                            train_dataset, net, writer, step)

    def train_step(self, ix, opt, use_cuda, train_dataset, net, writer, step):
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
        writer.add_scalar('loss', loss.item(), step)
        opt.step()

        print(loss.item())
        print(result.argmax(1))
        print(answer)
