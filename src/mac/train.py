import fs
import torch
import plumbum.cli
from torch.utils import data

import mac.utils
from mac import datasets, mac, cli, config


@cli.MAC.subcommand('train')
class Train(plumbum.cli.Application):
    def main(self, preprocessed_loc):
        mac.utils.cuda_message()

        mac_cell = mac.MAC(12, 512)
        net = mac.MACNet(mac_cell)
        preprocessed_fs = fs.open_fs(preprocessed_loc)

        with datasets.MAC_NP_Dataset(preprocessed_fs, 'train') as train_ds:
            try:
                self.train(net, train_ds)
            except KeyboardInterrupt:
                pass

        with preprocessed_fs.open('net.pkl', 'wb') as f:
            torch.save(net, f)

    def train(self, net, train_dataset):
        use_cuda = config.getconfig()['use_cuda']

        sampler = data.BatchSampler(
            data.RandomSampler(train_dataset), 8, False)
        opt = torch.optim.Adam(net.parameters())

        if use_cuda:
            net = net.cuda()

        for ix in sampler:
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

            print(loss.item())
            print(result.argmax(1))
            print(answer)
