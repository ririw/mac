from plumbum import cli

import torch
import torch.optim


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
        print(x.detach().numpy().round(3))
        if abs(x.detach().numpy()).sum() > 0.01:
            return 1
        return 0


def main() -> None:
    MAC.run(argv=['mac', 'check'])


if __name__ == '__main__':
    main()
