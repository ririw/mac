import torch
import torch.nn
import torch.nn.functional
import typing

__debug__options__ = {
    'save_locals': False,
    'locals': None
}


class CUCell(torch.nn.Module):
    def __init__(self, ctrl_dim):
        super().__init__()
        self.ctrl_dim = ctrl_dim

        self.ca_lin = torch.nn.Linear(ctrl_dim, 1)
        self.cq_lin = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)

    def forward(self, prev_ctrl, context_words, question_words):
        batch_size, seq_len, ctrl_dim = context_words.shape
        assert ctrl_dim == self.ctrl_dim

        check_shape(prev_ctrl, (batch_size, ctrl_dim))
        check_shape(question_words, (batch_size, ctrl_dim))

        c_concat = torch.cat([prev_ctrl, question_words], 1)
        check_shape(c_concat, (batch_size, ctrl_dim * 2))
        cq = self.cq_lin(c_concat)
        check_shape(cq, (batch_size, ctrl_dim))

        cw_weighted = torch.einsum('bd,bsd->bsd', cq, context_words)
        ca = self.ca_lin(cw_weighted).squeeze(2)
        check_shape(ca, (batch_size, seq_len))

        cv = torch.nn.Softmax(dim=1)(ca)
        check_shape(cv, (batch_size, seq_len))

        next_ctrl = torch.einsum('bs,bsd->bd', cv, context_words)
        check_shape(next_ctrl, (batch_size, ctrl_dim))

        if __debug__options__['save_locals']:
            __debug__options__['locals'] = locals()
        return next_ctrl


def check_shape(tensor: torch.Tensor,
                match: typing.Tuple[int, ...]) -> torch.Tensor:
    for s, m in zip(tensor.shape, match):
        if m is None:
            continue
        if s != m:
            msg = 'Shape {} does not match expectation {}'.format(
                tensor.shape, match)
            raise ValueError(msg)

    return tensor
