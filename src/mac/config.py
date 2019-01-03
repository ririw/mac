from ast import literal_eval

import os
import torch

_config = {
    'use_cuda': (torch.cuda.is_available() and not
                 literal_eval(os.environ.get('CUDA_DISABLED', 'False')))
}


def getconfig():
    return _config
