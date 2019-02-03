import numpy as np
import torch.nn

from mac import mac_cell, debug_helpers


class MACRec(torch.nn.Module):
    def __init__(self, recurrence_length, ctrl_dim):
        super().__init__()
        self.recurrence_length = recurrence_length
        self.ctrl_dim = ctrl_dim

        self.cu_cell = mac_cell.CUCell(ctrl_dim, recurrence_length)
        self.ru_cell = mac_cell.RUCell(ctrl_dim)
        self.wu_cell = mac_cell.WUCell(ctrl_dim)

        self.initial_control = torch.nn.Parameter(
            torch.zeros(1, self.ctrl_dim))
        self.initial_mem = torch.nn.Parameter(
            torch.zeros(1, self.ctrl_dim), requires_grad=False)
        self.output_cell = mac_cell.OutputCell(ctrl_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.initial_control)

    def forward(self, question_words, image_vec, context_words):
        batch_size, seq_len, ctrl_dim = context_words.shape
        if ctrl_dim != self.ctrl_dim:
            msg = 'Control dim mismatch, got {} but expected {}'\
                .format(ctrl_dim, self.ctrl_dim)
            raise ValueError(msg)
        debug_helpers.check_shape(question_words, (batch_size, ctrl_dim))
        debug_helpers.check_shape(image_vec, (batch_size, ctrl_dim, 14, 14))

        ctrl = self.initial_control.expand(batch_size, self.ctrl_dim)
        mem = self.initial_mem.expand(batch_size, self.ctrl_dim)

        for i in range(self.recurrence_length):
            ctrl = self.cu_cell(i, ctrl, context_words, question_words)
            ri = self.ru_cell(mem, image_vec, ctrl)
            mem = self.wu_cell(mem, ri, ctrl)
            debug_helpers.check_shape(mem, (batch_size, ctrl_dim))
            debug_helpers.check_shape(ri, (batch_size, ctrl_dim))
            debug_helpers.check_shape(ctrl, (batch_size, ctrl_dim))

        output = self.output_cell(ctrl, mem)
        debug_helpers.check_shape(output, (batch_size, 28))
        return output


class MACNet(torch.nn.Module):
    def __init__(self, mac: MACRec, total_words):
        super().__init__()

        self.ctrl_dim = mac.ctrl_dim
        if self.ctrl_dim % 2:
            msg = 'Control dim must be divisible by 2, ' \
                  'passed {}'.format(self.ctrl_dim)
            raise ValueError(msg)
        self.mac: MACRec = mac
        self.kb_mapper = torch.nn.Conv2d(1024, self.ctrl_dim, 3, padding=1)

        self.embedder = LazyEmbedding(total_words, self.ctrl_dim)
        self.lstm_processor = torch.nn.LSTM(
            self.ctrl_dim, self.ctrl_dim//2,
            bidirectional=True, batch_first=True)
        self.lstm_h0 = torch.nn.Parameter(torch.zeros(2, 1, self.ctrl_dim//2))
        self.lstm_c0 = torch.nn.Parameter(torch.zeros(2, 1, self.ctrl_dim//2))

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

        debug_helpers.check_shape(kb, (batch_size, 1024, 14, 14))

        kb_reduced = self.kb_mapper(kb)
        expected_kb_size = (batch_size, self.ctrl_dim, 14, 14)
        debug_helpers.check_shape(kb_reduced, expected_kb_size)

        h0_c0_size = (2, batch_size, self.ctrl_dim//2)
        h0 = self.lstm_h0.expand(h0_c0_size).contiguous()
        c0 = self.lstm_c0.expand(h0_c0_size).contiguous()

        question_tensors = self.embedder(questions)
        debug_helpers.check_shape(
            question_tensors, (batch_size, None, self.ctrl_dim))
        lstm_out, (hn, _) = self.lstm_processor(question_tensors, (h0, c0))

        hn_concat = torch.cat([hn[0], hn[1]], -1)
        debug_helpers.check_shape(hn_concat, (batch_size, self.ctrl_dim))
        debug_helpers.check_shape(lstm_out, (batch_size, None, self.ctrl_dim))

        res = self.mac.forward(hn_concat, kb_reduced, lstm_out)
        debug_helpers.save_all_locals()
        return res


class LazyEmbedding(torch.nn.Module):
    def __init__(self, max_words, embedding_dim):
        super().__init__()
        self.word_ix = {0: -1}
        self.embedding = torch.nn.Embedding(max_words+1, embedding_dim)

    def forward(self, scentences):
        tensors = []
        for scent in scentences:
            scent_ix = []
            for w in scent.split(' '):
                if w not in self.word_ix:
                    self.word_ix[w] = len(self.word_ix)
                    if len(self.word_ix) > self.embedding.num_embeddings:
                        raise IndexError('Too many words!')
                scent_ix.append(self.word_ix[w])
            assert scent_ix, 'Failed to parse "{}"'.format(scent)
            tensor = torch.from_numpy(np.array(scent_ix, np.int32))
            if self.embedding.weight.is_cuda:
                tensor = tensor.cuda()
            tensors.append(tensor)

        lengths = [t.shape[0] for t in tensors]
        len_ord = np.argsort(lengths)[::-1]
        len_ordered = [tensors[i] for i in len_ord]

        packed = torch.nn.utils.rnn.pack_sequence(len_ordered)
        padded = torch.nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True)[0]
        padded_ordered = padded[np.argsort(len_ord)].long().contiguous()
        return self.embedding(padded_ordered)
