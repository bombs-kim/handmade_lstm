import math

import torch
from torch import nn


# TODO: Add dropout functionality

class LstmCell(nn.Module):
    """
    LSTMCell receives a single timestep, not a sequence as an input
    """
    def __init__(self, input_size, output_size, bias=True, dropout=0.0):
        super().__init__()
        self.output_size = output_size
        self.dropout = dropout
        self.linear1 = nn.Linear(input_size, output_size*4, bias=bias)
        self.linear2 = nn.Linear(output_size, output_size*4, bias=bias)
        self.reset_parameters()

    def reset_state(self):
        self.b_cell_prev = torch.zeros(1, self.output_size)
        self.s_cell_prev = torch.zeros(1, self.output_size)

    def reset_parameters(self):
        # TODO: revemp
        std = 1.0 / math.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        if not hasattr(self, 'b_cell_prev'):
            raise RuntimeError('Please call reset_state before giving a sequence')
        # Following the convention of Supervised Sequence Labelling by Alex Graves
        # a_[something]: tensor before activation
        # b_[something]: tensor after activation

        a = self.linear1(x) + self.linear2(self.b_cell_prev)
        a_cell, a_gates = torch.split(
            a, (self.output_size, self.output_size*3), dim=1)
        b_input, b_forget, b_output = torch.split(
            torch.sigmoid(a_gates), (self.output_size,)*3, dim=1)

        s_cell = b_input * torch.tanh(a_cell) + b_forget * self.s_cell_prev
        b_cell = b_output * torch.tanh(s_cell)

        self.s_cell_prev, self.b_cell_prev = s_cell, b_cell
        # Note that cell state is only internally stored without being returned
        return b_cell
