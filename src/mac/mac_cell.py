import typing

import torch
import torch.nn
import torch.nn.functional
import torch.nn.functional as F

from mac.debug_helpers import check_shape, save_all_locals


class CUCell(torch.nn.Module):
    def __init__(self, ctrl_dim, recurrence_length):
        super().__init__()
        self.ctrl_dim = ctrl_dim

        self.step_trf = torch.nn.ModuleList(
            torch.nn.Linear(ctrl_dim, ctrl_dim)
            for _ in range(recurrence_length)
        )
        self.ca_lin = torch.nn.Linear(ctrl_dim, 1)
        self.cq_lin = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)

    def forward(self, step, prev_ctrl, context_words, question_words):
        batch_size, seq_len, ctrl_dim = context_words.shape
        assert ctrl_dim == self.ctrl_dim
        check_shape(prev_ctrl, (batch_size, ctrl_dim))
        check_shape(question_words, (batch_size, ctrl_dim))

        question_words_localized = self.step_trf[step](question_words)

        c_concat = torch.cat([prev_ctrl, question_words_localized], 1)
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

    def __init__(self, ctrl_dim: int):
        super().__init__()
        self.ctrl_dim = ctrl_dim

        self.mem_trf = torch.nn.Linear(ctrl_dim, ctrl_dim)
        self.ctrl_lin = torch.nn.Linear(ctrl_dim*2, ctrl_dim)
        self.attn = torch.nn.Linear(ctrl_dim, 1)

    def forward(self, mem, kb, control):
        batch_size, ctrl_dim = mem.shape
        assert ctrl_dim == self.ctrl_dim

        kb_shape = (batch_size, ctrl_dim, 14, 14)
        check_shape(kb, kb_shape)
        check_shape(control, (batch_size, ctrl_dim))
        kb = kb.permute(0, 2, 3, 1)
        check_shape(kb, (batch_size, 14, 14, ctrl_dim))

        mem_trfed = self.mem_trf(mem); check_shape(mem_trfed, (batch_size, ctrl_dim))

        mem_kb_inter = torch.einsum('bc,bwhc->bwhc', mem_trfed, kb)
        mem_kb_inter_cat = torch.cat([mem_kb_inter, kb], -1)
        check_shape(mem_kb_inter_cat, (batch_size, 14, 14, ctrl_dim*2))
        mem_kb_inter_cat_trf = self.ctrl_lin(mem_kb_inter_cat)
        check_shape(mem_kb_inter_cat_trf, (batch_size, 14, 14, ctrl_dim))

        ctrled = torch.einsum('bwhc,bc->bwhc', mem_kb_inter_cat_trf, control)
        attended = self.attn(ctrled).view(batch_size, -1)
        check_shape(attended, (batch_size, 14*14))
        attended_flat = F.softmax(attended, dim=-1).view(batch_size, 14, 14)
        check_shape(attended_flat, (batch_size, 14, 14))

        retrieved = torch.einsum('bwhc,bwh->bc', kb, attended_flat)
        check_shape(retrieved, (batch_size, ctrl_dim))
        return retrieved



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
