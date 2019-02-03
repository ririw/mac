import os

import allennlp.modules.elmo
import torch
import torchvision

from plumbum import cli

from mac.config import getconfig


def cuda_message():
    if getconfig()['use_cuda']:
        print('CUDA enabled')
    else:
        print('CUDA disabled, this may be very slow...')


class MAC(cli.Application):
    def main(self) -> int:
        if self.nested_command:
            return 0
        print('No command given.')
        return 1


@torch.jit.script
def some_fn(x) -> torch.Tensor:
    return x.abs().sum()


@MAC.subcommand('check')
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
        elmo_options_file = os.path.expanduser('~/Datasets/elmo_small_options.json')
        elmo_weights_file = os.path.expanduser('~/Datasets/elmo_small_weights.hdf5')
        allennlp.modules.elmo.Elmo(elmo_options_file, elmo_weights_file, 2)

        return 0
