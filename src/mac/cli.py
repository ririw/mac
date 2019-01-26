import os

import allennlp.modules.elmo
import fs
import torch
import torchvision.models.resnet
from plumbum import cli

from mac import inputs, config, utils, train

_used_no_del_by_flake8_ = [
    train
]


@torch.jit.script
def some_fn(x) -> torch.Tensor:
    return x.abs().sum()


@utils.MAC.subcommand('check')
class Check(cli.Application):
    def main(self) -> int:
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


@utils.MAC.subcommand('preprocess')
class Preprocess(cli.Application):
    limit = cli.SwitchAttr(['-l', '--limit'], argtype=int, default=None)

    def main(self, clevr_fs, preprocessed_loc):
        utils.cuda_message()
        config.getconfig()['work_limit'] = self.limit

        out_fs = fs.open_fs(preprocessed_loc)
        zf = fs.open_fs(clevr_fs)

        inputs.lang_preprocess('train', zf, out_fs)
        with zf.opendir('images/train/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs, self.limit)
            inputs.image_preprocess('train', ds, out_fs)
        inputs.lang_preprocess('val', zf, out_fs)
        with zf.opendir('images/val/') as data_fs:
            ds = inputs.CLEVRImageData(data_fs, self.limit)
            inputs.image_preprocess('val', ds, out_fs)

        zf.close()


def main() -> None:
    utils.MAC.run()


if __name__ == '__main__':
    main()
