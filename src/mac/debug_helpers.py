import inspect
import torch
import typing

from mac import config


def check_shape(tensor: torch.Tensor,
                match: typing.Tuple[typing.Optional[int], ...]) \
        -> torch.Tensor:
    if not config.getconfig()['check']:
        return tensor
    if len(tensor.shape) != len(match):
        msg = 'Shape {} does not match expectation {}'.format(tensor.shape, match)
        raise ValueError(msg)
    for s, m in zip(tensor.shape, match):
        if m is None:
            continue
        if s != m:
            msg = 'Shape {} does not match expectation {}'.format(tensor.shape, match)
            raise ValueError(msg)

    return tensor


__debug__options__ = {
    'save_locals': False,
    'locals': None,
    'debug_stack': [],
}


def save_all_locals():
    if not __debug__options__['save_locals']:
        return
    frame = inspect.currentframe()
    __debug__options__['locals'] = frame.f_back.f_locals


def enable_debug():
    __debug__options__['save_locals'] = True


def push_debug_state(new_state):
    __debug__options__['debug_stack'].append(__debug__options__['save_locals'])
    __debug__options__['save_locals'] = new_state


def pop_debug_state():
    new_state = __debug__options__['debug_stack'].pop(-1)
    __debug__options__['save_locals'] = new_state
    return new_state


def get_saved_locals():
    assert __debug__options__['save_locals']
    res = __debug__options__['locals']
    msg = 'No locals found. Ensure your code called save_all_locals() ' \
          'and that you called enable_debug() to enable saving'
    assert res is not None, msg
    return res
