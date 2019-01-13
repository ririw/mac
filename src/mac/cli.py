import os

import fs.zipfs
import fs.appfs
import fs.osfs

from mac import inputs, datasets, mac
from plumbum import cli
import allennlp.modules.elmo

import torch
import torch.optim
import torchvision.models.resnet
from torch.utils import data


from mac.config import getconfig


@torch.jit.script
def some_fn(x) -> torch.Tensor:
    return x.abs().sum()


class MAC(cli.Application):
    def main(self) -> int:  # pylint: disable=arguments-differ
        if self.nested_command:
            return 0
        print('No command given.')
        return 1


@MAC.subcommand('check')
class Check(cli.Application):
    def main(self) -> int:  # pylint: disable=arguments-differ
        x = torch.ones(5, requires_grad=True)

        opt = torch.optim.LBFGS([x])

        def err() -> torch.Tensor:
            opt.zero_grad()
            r = some_fn(x)
            r.backward()
            return r

        opt.step(err)
        print('Expect: [-0. -0. -0. -0. -0.]')
        print('Got:   ', x.detach().numpy().round(3))
        if abs(x.detach().numpy()).sum() > 0.01:
            return 1

        print('Checking can load resnet...')
        torchvision.models.resnet.resnet101(False)

        print('Checking can load elmo...')
        elmo_options_file = os.path.expanduser(
            '~/Datasets/elmo_small_options.json')
        elmo_weights_file = os.path.expanduser(
            '~/Datasets/elmo_small_weights.hdf5')
        allennlp.modules.elmo.Elmo(elmo_options_file, elmo_weights_file, 2)

        return 0


@MAC.subcommand('preprocess')
class Preprocess(cli.Application):
    limit = cli.SwitchAttr(['-l', '--limit'], argtype=int, default=None)

    def main(self, clevr_fs, preprocessed_loc):
        cuda_message()
        getconfig()['work_limit'] = self.limit

        out_fs = fs.open_fs(preprocessed_loc)
        zf = fs.open_fs(clevr_fs)

        inputs.lang_preprocess('val', zf, out_fs)
        inputs.lang_preprocess('train', zf, out_fs)
        with zf.opendir('images/train/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs, self.limit)
            inputs.image_preprocess('train', ds, out_fs)
        with zf.opendir('images/val/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs, self.limit)
            inputs.image_preprocess('val', ds, out_fs)
        zf.close()


@MAC.subcommand('train')
class Train(cli.Application):
    def main(self, preprocessed_loc):
        cuda_message()

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
        use_cuda = getconfig()['use_cuda']

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


def cuda_message():
    if getconfig()['use_cuda']:
        print('CUDA enabled')
    else:
        print('CUDA disabled, this may be very slow...')


def main() -> None:
    MAC.run()


if __name__ == '__main__':
    main()
