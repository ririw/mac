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
