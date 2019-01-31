import typing

import torch
import torch.nn
import torch.nn.functional

from mac.debug_helpers import check_shape, save_all_locals


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

        save_all_locals()
        return next_ctrl


class RUCell(torch.nn.Module):
    default_image_size = (14, 14)

    def __init__(self, ctrl_dim: int,
                 image_size: typing.Optional[typing.Tuple[int, int]] = None):
        super().__init__()
        if image_size is None:
            image_size = self.default_image_size
        self.im_w, self.im_h = image_size
        self.ctrl_dim = ctrl_dim

        self.mem_lin = torch.nn.Linear(ctrl_dim, ctrl_dim)
        self.kb2_lin = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)
        self.ctrl_lin = torch.nn.Linear(ctrl_dim, ctrl_dim)

    def forward(self, mem, kb, control):
        batch_size, ctrl_dim = mem.shape
        assert ctrl_dim == self.ctrl_dim
        kb_shape = (batch_size, self.im_w, self.im_h, ctrl_dim)
        attn_shape = (batch_size, self.im_w, self.im_h)
        check_shape(kb, kb_shape)
        check_shape(control, (batch_size, ctrl_dim))

        mem_lin_1 = self.mem_lin(mem)
        check_shape(mem_lin_1, (batch_size, ctrl_dim))
        direct_inter = torch.einsum('bc,bwhc->bwhc', mem_lin_1, kb)
        check_shape(direct_inter, kb_shape)

        second_inter = self.kb2_lin(torch.cat([direct_inter, kb], -1))
        check_shape(second_inter, kb_shape)

        weighted_control = torch.einsum('bc,bwhc->bwhc', control, second_inter)
        ra = self.ctrl_lin(weighted_control).sum(-1)
        check_shape(ra, attn_shape)
        rv = torch.nn.Softmax(dim=-1)(ra.view(batch_size, -1)).view(attn_shape)

        check_shape(rv, attn_shape)
        ri = torch.einsum('bwhc,bwh->bc', kb, rv)
        check_shape(ri, (batch_size, ctrl_dim))

        save_all_locals()
        return ri


class WUCell(torch.nn.Module):
    def __init__(self, ctrl_dim, use_prev_control=False, gate_mem=True):
        super().__init__()
        self.gate_mem = gate_mem
        self.use_prev_control = use_prev_control
        self.ctrl_dim = ctrl_dim

        self.mem_read_int = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)
        if self.use_prev_control:
            self.mem_select = torch.nn.Linear(ctrl_dim, 1)
            self.mem_merge_info = torch.nn.Linear(ctrl_dim, ctrl_dim)
            self.mem_merge_other = torch.nn.Linear(
                ctrl_dim, ctrl_dim, bias=False)
        if self.gate_mem:
            self.mem_gate = torch.nn.Linear(ctrl_dim, 1)

    def forward(self, mem, ri, control, prev_control=None):
        batch_size, ctrl_dim = mem.shape
        assert ctrl_dim == self.ctrl_dim
        assert self.use_prev_control == (prev_control is not None)
        if prev_control is not None:
            check_shape(prev_control, (batch_size, None, ctrl_dim))
        check_shape(ri, (batch_size, ctrl_dim))
        check_shape(control, (batch_size, ctrl_dim))

        m_info = self.mem_read_int(torch.cat([ri, mem], 1))
        check_shape(m_info, (batch_size, ctrl_dim))

        if prev_control is not None:
            control_similarity = torch.einsum(
                'bsd,bd->bsd', prev_control, control)
            control_expweight = self.mem_select(control_similarity)
            check_shape(control_expweight, (batch_size, None, 1))
            sa = torch.nn.functional.softmax(
                control_similarity.squeeze(2), dim=1)
            m_other = torch.einsum('bs,bsd->bd', sa, prev_control)
            m_info = (self.mem_merge_other(m_other)
                      + self.mem_merge_info(m_info))
        check_shape(m_info, (batch_size, ctrl_dim))

        if self.gate_mem:
            mem_ctrl = self.mem_gate(control).squeeze(1)
            ci = torch.sigmoid(mem_ctrl)
            check_shape(m_info, (batch_size, self.ctrl_dim))
            m_next = (torch.einsum('bd,b->bd', mem, ci)
                      + torch.einsum('bd,b->bd', m_info, 1 - ci))
        else:
            m_next = m_info

        save_all_locals()
        return m_next


class OutputCell(torch.nn.Module):
    def __init__(self, ctrl_dim, out_dim=28):
        super().__init__()
        self.ctrl_dim = ctrl_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(ctrl_dim * 2, ctrl_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(ctrl_dim, out_dim),
        )

    def forward(self, control, mem):
        check_shape(control, (None, self.ctrl_dim))
        batch_size, ctrl_dim = control.shape
        check_shape(mem, (batch_size, ctrl_dim))

        return self.layers(torch.cat([control, mem], 1))
