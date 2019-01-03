from plumbum import cli

import torch
import torch.optim


@torch.jit.script
def some_fn(x):
    return x.abs().sum()


class MAC(cli.Application):
    def main(self):
        if self.nested_command:
            return 0
        print('No command given.')
        return 1


@MAC.subcommand('check')
class Check(cli.Application):
    def main(self):
        x = torch.ones(5, requires_grad=True)

        opt = torch.optim.LBFGS([x])

        def err():
            opt.zero_grad()
            r = some_fn(x)
            r.backward()
            return r

        opt.step(err)
        print('Expect: [-0. -0. -0. -0. -0.]')
        print(x.detach().numpy().round(3))
        if abs(x.detach().numpy()).sum() > 0.01:
            return 1
        else:
            return 0


def main():
    MAC.run(argv=['mac', 'check'])


if __name__ == '__main__':
    main()
