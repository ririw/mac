import os

import fs.zipfs
import fs.appfs
from mac import inputs
from plumbum import cli
import allennlp.modules.elmo

import torch
import torch.optim
import torchvision.models.resnet


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
    def main(self, clevr_fs):
        out_fs = fs.appfs.UserCacheFS('mac')
        zf = fs.zipfs.ZipFS(clevr_fs)
        print('Need more space to do training...')
        with zf.opendir('CLEVR_v1.0/images/test/') as te:
            ds = inputs.CLEVRData(te)
            inputs.image_preprocess('test', ds, out_fs)


def main() -> None:
    MAC.run()


if __name__ == '__main__':
    main()
