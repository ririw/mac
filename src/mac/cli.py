import os

import fs.zipfs
import fs.appfs
import fs.osfs

from mac import inputs
from plumbum import cli
import allennlp.modules.elmo

import torch
import torch.optim
import torchvision.models.resnet

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

    def main(self, clevr_fs, output_loc):
        if getconfig()['use_cuda']:
            print('CUDA enabled')
        else:
            print('CUDA disabled, this may be very slow...')

        getconfig()['work_limit'] = self.limit
        out_fs = fs.open_fs(output_loc)
        zf = fs.open_fs(clevr_fs)

        inputs.lang_preprocess('val', zf, out_fs)
        inputs.lang_preprocess('train', zf, out_fs)
        with zf.opendir('images/train/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs)
            inputs.image_preprocess('train', ds, out_fs)
        with zf.opendir('images/val/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs)
            inputs.image_preprocess('val', ds, out_fs)
        zf.close()


def main() -> None:
    MAC.run()


if __name__ == '__main__':
    main()
