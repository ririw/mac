import typing
from ast import literal_eval

import os
import torch
from tensorboardX import SummaryWriter

_answers = [
        '0',
        '1',
        '10',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'blue',
        'brown',
        'cube',
        'cyan',
        'cylinder',
        'gray',
        'green',
        'large',
        'metal',
        'no',
        'purple',
        'red',
        'rubber',
        'small',
        'sphere',
        'yellow',
        'yes'
    ]

_config = {
    'use_cuda': (torch.cuda.is_available() and not
                 literal_eval(os.environ.get('CUDA_DISABLED', 'False'))),
    'answer_mapping': {ans: ix for ix, ans in enumerate(_answers)},
    'progress': True,
    'work_limit': None,
    'step': 0,
    'summary_writer': None,
    'check': True,
}


def torch_device():
    if getconfig()['use_cuda']:
        return 'cuda'
    else:
        return 'cpu'

def getconfig():
    return _config


def setconfig(key, value):
    if key not in _config:
        raise KeyError('Key {} not found in config amongst {}'.format(
            key, _config.keys()))
    _config[key] = value


def get_writer_maybe() -> typing.Optional[SummaryWriter]:
    cfg = getconfig()
    if cfg['step'] % 25 == 24:
        return cfg['summary_writer']
    else:
        return None
