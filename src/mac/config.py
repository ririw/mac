from ast import literal_eval

import os
import torch

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
    'progress': True
}


def getconfig():
    return _config
