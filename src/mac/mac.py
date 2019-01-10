import torch.nn

from mac import mac_cell


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

        self.initial_control = torch.nn.Parameter(torch.zeros(1, self.ctrl_dim))
        self.initial_mem = torch.nn.Parameter(torch.zeros(1, self.ctrl_dim))
        self.reset_parameters()

        self.output_cell = mac_cell.OutputCell(ctrl_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.initial_control)
        torch.nn.init.xavier_normal_(self.initial_mem)

    def forward(self, question_words, image_vec, context_words):
        batch_size, seq_len, ctrl_dim = context_words.shape
        if ctrl_dim != self.ctrl_dim:
            msg = 'Control dim mismatch, got {} but expected {}'\
                .format(ctrl_dim, self.ctrl_dim)
            raise ValueError(msg)
        mac_cell.check_shape(question_words, (batch_size, ctrl_dim))
        mac_cell.check_shape(image_vec, (batch_size, 14, 14, ctrl_dim))

        ctrl = self.initial_control.expand(batch_size, self.ctrl_dim)
        mem = self.initial_mem.expand(batch_size, self.ctrl_dim)
        for i in range(self.recurrence_length):
            cu_cell = self.control_cells[i]
            ru_cell = self.read_cells[i]
            wu_cell = self.write_cells[i]

            ctrl = cu_cell(ctrl, context_words, question_words)
            ri = ru_cell(mem, image_vec, ctrl)
            mem = wu_cell(mem, ri, ctrl)
            mac_cell.check_shape(mem, (batch_size, ctrl_dim))
            mac_cell.check_shape(ri, (batch_size, ctrl_dim))
            mac_cell.check_shape(ctrl, (batch_size, ctrl_dim))

        output = self.output_cell(ctrl, mem)
        mac_cell.check_shape(output, (batch_size, 28))
        return output
