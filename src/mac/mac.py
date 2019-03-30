import torch
import torch.nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnnutils

from mac import debug_helpers
from mac.debug_helpers import check_shape, save_all_locals


class MACRec(torch.nn.Module):
    def __init__(self, recurrence_length, ctrl_dim):
        super().__init__()
        self.recurrence_length = recurrence_length
        self.ctrl_dim = ctrl_dim

        self.cu_cells = torch.nn.ModuleList(
            [CUCell(ctrl_dim, recurrence_length) for _ in range(recurrence_length)]
        )
        self.ru_cells = torch.nn.ModuleList(
            [RUCell(ctrl_dim) for _ in range(recurrence_length)]
        )
        self.wu_cells = torch.nn.ModuleList(
            [WUCell(ctrl_dim) for _ in range(recurrence_length)]
        )

        self.initial_control = torch.nn.Parameter(torch.zeros(1, self.ctrl_dim))
        self.initial_mem = torch.nn.Parameter(torch.zeros(1, self.ctrl_dim))
        self.output_cell = OutputCell(ctrl_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.initial_control)
        torch.nn.init.xavier_normal_(self.initial_mem)

    def forward(self, question_words, image_vec, context_words):
        batch_size = context_words.shape[0]

        debug_helpers.check_shape(context_words, (batch_size, None, self.ctrl_dim))
        debug_helpers.check_shape(question_words, (batch_size, self.ctrl_dim * 2))
        debug_helpers.check_shape(image_vec, (batch_size, self.ctrl_dim, 14, 14))

        ctrl = self.initial_control.expand(batch_size, self.ctrl_dim)
        mem = self.initial_mem.expand(batch_size, self.ctrl_dim)

        for i in range(self.recurrence_length):
            cu_cell = self.cu_cells[i]
            ru_cell = self.ru_cells[i]
            wu_cell = self.wu_cells[i]
            ctrl = cu_cell(i, ctrl, context_words, question_words)
            ri = ru_cell(mem, image_vec, ctrl)
            mem = wu_cell(mem, ri, ctrl)
            debug_helpers.check_shape(mem, (batch_size, self.ctrl_dim))
            debug_helpers.check_shape(ri, (batch_size, self.ctrl_dim))
            debug_helpers.check_shape(ctrl, (batch_size, self.ctrl_dim))

        output = self.output_cell(question_words, mem)
        debug_helpers.check_shape(output, (batch_size, 28))
        return output


class MACNet(torch.nn.Module):
    def __init__(self, mac: MACRec, total_words):
        super().__init__()

        self.ctrl_dim = mac.ctrl_dim
        self.mac: MACRec = mac
        self.kb_mapper = torch.nn.Sequential(
            torch.nn.Conv2d(1024, self.ctrl_dim, 3, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.ctrl_dim, self.ctrl_dim, 3, padding=1),
            torch.nn.PReLU(),
        )

        self.embedding = torch.nn.Embedding(total_words + 1, self.ctrl_dim)
        self.lstm_processor = torch.nn.LSTM(
            self.ctrl_dim, self.ctrl_dim, bidirectional=True, batch_first=True
        )
        self.lstm_h0 = torch.nn.Parameter(torch.zeros(2, 1, self.ctrl_dim))
        self.lstm_c0 = torch.nn.Parameter(torch.zeros(2, 1, self.ctrl_dim))
        self.lstm_proj = torch.nn.Linear(self.ctrl_dim * 2, self.ctrl_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lstm_h0)
        torch.nn.init.xavier_normal_(self.lstm_c0)

    def forward(self, kb, questions, qn_lens):
        batch_size = kb.shape[0]
        kb_reduced = self.process_img(kb, batch_size)
        context, question = self.process_qn(questions, qn_lens, batch_size)

        res = self.mac.forward(question, kb_reduced, context)
        debug_helpers.save_all_locals()
        return res

    def process_img(self, kb, batch_size):
        debug_helpers.check_shape(kb, (batch_size, 1024, 14, 14))
        kb_reduced = self.kb_mapper(kb)
        debug_helpers.check_shape(kb_reduced, (batch_size, self.ctrl_dim, 14, 14))
        return kb_reduced

    def process_qn(self, questions, qn_lens, batch_size):
        debug_helpers.check_shape(questions, (batch_size, None))
        debug_helpers.check_shape(qn_lens, (batch_size,))

        qn_tensors = self.embedding(questions)
        debug_helpers.check_shape(qn_tensors, (batch_size, None, self.ctrl_dim))
        packed_embedded = rnnutils.pack_padded_sequence(qn_tensors, qn_lens)

        h0_c0_size = (2, batch_size, self.ctrl_dim)
        h0 = self.lstm_h0.expand(h0_c0_size).contiguous()
        c0 = self.lstm_c0.expand(h0_c0_size).contiguous()

        lstm_out, (hn, _) = self.lstm_processor(packed_embedded, (h0, c0))
        padded_lstm, _ = rnnutils.pad_packed_sequence(lstm_out)
        proj_lstm = self.lstm_proj(padded_lstm)

        hn_concat = torch.cat([hn[0], hn[1]], -1)
        debug_helpers.check_shape(proj_lstm, (batch_size, None, self.ctrl_dim))
        debug_helpers.check_shape(hn_concat, (batch_size, self.ctrl_dim * 2))
        return proj_lstm, hn_concat


class CUCell(torch.nn.Module):
    def __init__(self, ctrl_dim, recurrence_length):
        super().__init__()
        self.ctrl_dim = ctrl_dim

        self.step_trf = torch.nn.ModuleList(
            torch.nn.Linear(ctrl_dim * 2, ctrl_dim) for _ in range(recurrence_length)
        )
        self.ca_lin = torch.nn.Linear(ctrl_dim, 1)
        self.cq_lin = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)

    def forward(self, step, prev_ctrl, context_words, question_words):
        batch_size, seq_len, _ = context_words.shape

        check_shape(prev_ctrl, (batch_size, self.ctrl_dim))
        check_shape(question_words, (batch_size, self.ctrl_dim * 2))
        check_shape(context_words, (batch_size, seq_len, self.ctrl_dim))

        question_words_localized = self.step_trf[step](question_words)

        c_concat = torch.cat([prev_ctrl, question_words_localized], 1)
        check_shape(c_concat, (batch_size, self.ctrl_dim * 2))
        cq = self.cq_lin(c_concat)
        check_shape(cq, (batch_size, self.ctrl_dim))

        cw_weighted = torch.einsum("bd,bsd->bsd", cq, context_words)
        ca = self.ca_lin(cw_weighted).squeeze(2)
        check_shape(ca, (batch_size, seq_len))

        cv = torch.nn.Softmax(dim=1)(ca)
        check_shape(cv, (batch_size, seq_len))

        next_ctrl = torch.einsum("bs,bsd->bd", cv, context_words)
        check_shape(next_ctrl, (batch_size, self.ctrl_dim))

        save_all_locals()
        return next_ctrl


class RUCell(torch.nn.Module):
    default_image_size = (14, 14)

    def __init__(self, ctrl_dim: int):
        super().__init__()
        self.ctrl_dim = ctrl_dim

        self.mem_trf = torch.nn.Linear(ctrl_dim, ctrl_dim)
        self.ctrl_lin = torch.nn.Linear(ctrl_dim * 2, ctrl_dim)
        self.attn = torch.nn.Linear(ctrl_dim, 1)

    def forward(self, mem, kb, control):
        batch_size, ctrl_dim = mem.shape
        assert ctrl_dim == self.ctrl_dim

        kb_shape = (batch_size, ctrl_dim, 14, 14)
        check_shape(kb, kb_shape)
        check_shape(control, (batch_size, ctrl_dim))
        kb = kb.permute(0, 2, 3, 1)
        check_shape(kb, (batch_size, 14, 14, ctrl_dim))

        mem_trfed = self.mem_trf(mem)
        check_shape(mem_trfed, (batch_size, ctrl_dim))

        mem_kb_inter = torch.einsum("bc,bwhc->bwhc", mem_trfed, kb)
        mem_kb_inter_cat = torch.cat([mem_kb_inter, kb], -1)
        check_shape(mem_kb_inter_cat, (batch_size, 14, 14, ctrl_dim * 2))
        mem_kb_inter_cat_trf = self.ctrl_lin(mem_kb_inter_cat)
        check_shape(mem_kb_inter_cat_trf, (batch_size, 14, 14, ctrl_dim))

        ctrled = torch.einsum("bwhc,bc->bwhc", mem_kb_inter_cat_trf, control)
        attended_flat = self.attn(ctrled).view(batch_size, -1)
        check_shape(attended_flat, (batch_size, 14 * 14))
        attended = F.softmax(attended_flat, dim=-1).view(batch_size, 14, 14)
        check_shape(attended, (batch_size, 14, 14))

        retrieved = torch.einsum("bwhc,bwh->bc", kb, attended)
        check_shape(retrieved, (batch_size, ctrl_dim))
        save_all_locals()
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
            self.mem_merge_other = torch.nn.Linear(ctrl_dim, ctrl_dim, bias=False)
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
            control_similarity = torch.einsum("bsd,bd->bsd", prev_control, control)
            control_expweight = self.mem_select(control_similarity)
            check_shape(control_expweight, (batch_size, None, 1))
            sa = torch.nn.functional.softmax(control_similarity.squeeze(2), dim=1)
            m_other = torch.einsum("bs,bsd->bd", sa, prev_control)
            m_info = self.mem_merge_other(m_other) + self.mem_merge_info(m_info)
        check_shape(m_info, (batch_size, ctrl_dim))

        if self.gate_mem:
            mem_ctrl = self.mem_gate(control).squeeze(1)
            ci = torch.sigmoid(mem_ctrl)
            check_shape(m_info, (batch_size, self.ctrl_dim))
            m_next = torch.einsum("bd,b->bd", mem, ci) + torch.einsum(
                "bd,b->bd", m_info, 1 - ci
            )
        else:
            m_next = m_info

        save_all_locals()
        return m_next


class OutputCell(torch.nn.Module):
    def __init__(self, ctrl_dim, out_dim=28):
        super().__init__()
        self.ctrl_dim = ctrl_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(ctrl_dim * 3, ctrl_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(ctrl_dim, out_dim),
        )

    def forward(self, h, mem):
        batch_size = mem.shape[0]
        check_shape(h, (batch_size, self.ctrl_dim * 2))
        check_shape(mem, (batch_size, self.ctrl_dim))

        return self.layers(torch.cat([h, mem], 1))
