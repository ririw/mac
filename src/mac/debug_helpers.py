import inspect
import torch
import typing


def check_shape(tensor: torch.Tensor,
                match: typing.Tuple[typing.Optional[int], ...])\
        -> torch.Tensor:
    for s, m in zip(tensor.shape, match):
        if m is None:
            continue
        if s != m:
            msg = 'Shape {} does not match expectation {}'.format(
                tensor.shape, match)
            raise ValueError(msg)

    return tensor


__debug__options__ = {
    'save_locals': False,
    'locals': None
}


def save_all_locals():
    if not __debug__options__['save_locals']:
        return
    frame = inspect.currentframe()
    __debug__options__['locals'] = frame.f_back.f_locals


def enable_debug():
    __debug__options__['save_locals'] = True


def get_saved_locals():
    assert __debug__options__['save_locals']
    res = __debug__options__['locals']
    msg = 'No locals found. Ensure your code called save_all_locals() ' \
          'and that you called enable_debug() to enable saving'
    assert res is not None, msg
    return res
