import torch.nn

from mac import mac_cell, debug_helpers


class MAC(torch.nn.Module):
    def __init__(self, recurrence_length, ctrl_dim):
        super().__init__()
        self.recurrence_length = recurrence_length
        self.ctrl_dim = ctrl_dim

        self.control_cells = []
        self.read_cells = []
        self.write_cells = []

        for i in range(recurrence_length):
            cu_cell = mac_cell.CUCell(ctrl_dim)
            ru_cell = mac_cell.RUCell(ctrl_dim)
            wu_cell = mac_cell.WUCell(ctrl_dim)

            self.add_module('control_{:d}'.format(i), cu_cell)
            self.add_module('read_{:d}'.format(i), ru_cell)
            self.add_module('write_{:d}'.format(i), wu_cell)
            self.control_cells.append(cu_cell)
            self.read_cells.append(ru_cell)
            self.write_cells.append(wu_cell)

        self.initial_control = torch.nn.Parameter(
            torch.zeros(1, self.ctrl_dim))
        self.initial_mem = torch.nn.Parameter(torch.zeros(1, self.ctrl_dim))
        self.output_cell = mac_cell.OutputCell(ctrl_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.initial_control)
        torch.nn.init.xavier_normal_(self.initial_mem)

    def forward(self, question_words, image_vec, context_words):
        batch_size, seq_len, ctrl_dim = context_words.shape
        if ctrl_dim != self.ctrl_dim:
            msg = 'Control dim mismatch, got {} but expected {}'\
                .format(ctrl_dim, self.ctrl_dim)
            raise ValueError(msg)
        debug_helpers.check_shape(question_words, (batch_size, ctrl_dim))
        debug_helpers.check_shape(image_vec, (batch_size, 14, 14, ctrl_dim))

        ctrl = self.initial_control.expand(batch_size, self.ctrl_dim)
        mem = self.initial_mem.expand(batch_size, self.ctrl_dim)
        for i in range(self.recurrence_length):
            cu_cell = self.control_cells[i]
            ru_cell = self.read_cells[i]
            wu_cell = self.write_cells[i]

            ctrl = cu_cell(ctrl, context_words, question_words)
            ri = ru_cell(mem, image_vec, ctrl)
            mem = wu_cell(mem, ri, ctrl)
            debug_helpers.check_shape(mem, (batch_size, ctrl_dim))
            debug_helpers.check_shape(ri, (batch_size, ctrl_dim))
            debug_helpers.check_shape(ctrl, (batch_size, ctrl_dim))

        output = self.output_cell(ctrl, mem)
        debug_helpers.check_shape(output, (batch_size, 28))
        return output


class MACNet(torch.nn.Module):
    def __init__(self, mac: MAC):
        super().__init__()
        self.ctrl_dim = mac.ctrl_dim
        if self.ctrl_dim % 2:
            msg = 'Control dim must be divisible by 2, ' \
                  'passed {}'.format(self.ctrl_dim)
            raise ValueError(msg)
        self.mac: MAC = mac
        self.kb_mapper = torch.nn.Conv2d(1024, self.ctrl_dim, 3, padding=1)

        self.lstm_processor = torch.nn.LSTM(
            256, self.ctrl_dim//2, bidirectional=True, batch_first=True)
        self.lstm_h0 = torch.zeros(2, 1, self.ctrl_dim//2)
        self.lstm_c0 = torch.zeros(2, 1, self.ctrl_dim//2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lstm_h0)
        torch.nn.init.xavier_normal_(self.lstm_c0)

    def forward(self, kb, questions):
        batch_size = kb.shape[0]
        if len(questions) != batch_size:
            msg = 'KB size and number of questions should be equal, ' \
                  'got {} and {} respectively'.format(
                    batch_size, len(questions))
            raise ValueError(msg)
        seq_len = questions.shape[1]

        debug_helpers.check_shape(kb, (batch_size, 1024, 14, 14))
        debug_helpers.check_shape(questions, (batch_size, seq_len, 256))

        kb_reduced = self.kb_mapper(kb)
        expected_kb_size = (batch_size, self.ctrl_dim, 14, 14)
        debug_helpers.check_shape(kb_reduced, expected_kb_size)

        h0_c0_size = (2, batch_size, self.ctrl_dim//2)
        h0 = self.lstm_h0.expand(h0_c0_size)
        c0 = self.lstm_c0.expand(h0_c0_size)

        lstm_out, (hn, cn) = self.lstm_processor(questions, (h0, c0))

        hn_concat = torch.cat([hn[0], hn[1]], -1)
        cn_concat = torch.cat([hn[0], hn[1]], -1)
        debug_helpers.check_shape(hn_concat, (batch_size, self.ctrl_dim))
        debug_helpers.check_shape(cn_concat, (batch_size, self.ctrl_dim))
        debug_helpers.check_shape(
            lstm_out, (batch_size, seq_len, self.ctrl_dim))

        # Fixme - deeply dubious code...
        kb_tf = kb_reduced.transpose(3, 1)
        res = self.mac.forward(hn_concat, kb_tf, lstm_out)
        debug_helpers.save_all_locals()
        return res
