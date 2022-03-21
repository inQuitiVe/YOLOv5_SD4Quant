import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import math

__all__ = ['quantize_lstm']


def quantize_lstm(input_size, hidden_size, num_layers=1, bias=True,
                  batch_first=False, dropout=0, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''
    if dropout > 0:
        stack_type = StackedLSTMWithDropout
        if bidirectional:
            layer_type = BidirLSTMLayer
            dirs = 2
        else:
            layer_type = LSTMLayer
            dirs = 1
    else:
        stack_type = StackedLSTM
        if bidirectional:
            layer_type = BidirLSTMLayer
            dirs = 2
        else:
            layer_type = LSTMLayer
            dirs = 1

    return stack_type(num_layers, layer_type, dropout,
                      first_layer_args=[LSTMCell, input_size, hidden_size],
                      first_layer_kargs = {'bias':bias, 'batch_first':batch_first},
                      other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
                      other_layer_kargs = {'bias':bias, 'batch_first':batch_first})


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state): #use i6 bit for activation function
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        if (state is None):
            batch_size = input.size()[0] #(batch, input)
            hx, cx = (torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device),
                      torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device))
        else:
            hx, cx = state
            if (hx.shape[0] == 1):
                hx = hx.squeeze(0)
                cx = cx.squeeze(0)

        #print(input.shape, hx.shape, self.i2h, self.h2h)

        gates = self.i2h(input) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args, batch_first=False, **cell_kargs):
        super(LSTMLayer, self).__init__()
        self.batch_first = batch_first
        self.cell = cell(*cell_args, **cell_kargs)

    def forward(self, input, state=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        if (self.batch_first):
            inputs = input.permute(1, 0, 2).unbind(0)
        else:
            inputs = input.unbind(0)

        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out] #(seq, batch, hidden)

        #state[0] = state[0].unsqueeze(0) #(batch,hidden) -> (1, batch, hidden)
        #state[1] = state[1].unsqueeze(0)

        if (self.batch_first):
            return torch.stack(outputs).permute(1, 0, 2), (state[0].unsqueeze(0), state[1].unsqueeze(0))
        return torch.stack(outputs), (state[0].unsqueeze(0), state[1].unsqueeze(0))

class ReverseLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args, batch_first=False, **cell_kargs):
        super(ReverseLSTMLayer, self).__init__()
        self.batch_first = batch_first
        self.cell = cell(*cell_args, **cell_kargs)

    def forward(self, input, state=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        if (self.batch_first):
            inputs = reverse(input.permute(1, 0, 2).unbind(0))
        else:
            inputs = reverse(input.unbind(0))

        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]

        #state[0] = state[0].unsqueeze(0) #(batch,hidden) -> (1, batch, hidden)
        #state[1] = state[1].unsqueeze(0)

        if (self.batch_first):
            return torch.stack(outputs).permute(1, 0, 2), (state[0].unsqueeze(0), state[1].unsqueeze(0))
        return torch.stack(reverse(outputs)), (state[0].unsqueeze(0), state[1].unsqueeze(0))

class BidirLSTMLayer(nn.Module):
    #__constants__ = ['directions']

    def __init__(self, cell, *cell_args, **cell_kargs):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args, **cell_kargs),
            ReverseLSTMLayer(cell, *cell_args, **cell_kargs),
        ])

    def forward(self, input, states=None):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        if (states == None):
            states = [None, None]

        outputs = []
        h = []
        c = []
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            h += [out_state[0]]
            c += [out_state[1]]
            i += 1

        h = torch.cat(h, dim=0)
        c = torch.cat(c, dim=0)

        return torch.cat(outputs, -1), (h, c)

def init_stacked_lstm(num_layers, layer, first_layer_args, first_layer_kargs, other_layer_args, other_layer_kargs):
    layers = [layer(*first_layer_args, **first_layer_kargs)] + [layer(*other_layer_args, **other_layer_kargs)
                                                                for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedLSTM(nn.Module):
    #__constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, dropout, first_layer_args, first_layer_kargs, other_layer_args, other_layer_kargs):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, first_layer_kargs,
                                        other_layer_args, other_layer_kargs)

    def forward(self, input, states=None):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        if (states is None):
            states = [None for _ in range(self.num_layers)]
        elif (isinstance(states, tuple)): #only first layer's state
            states = [states if (i==0) else None for i in range(self.num_layers)]

        h = []
        c = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[0]
            output, out_state = rnn_layer(output, state)
            h += [out_state[0]]
            c += [out_state[1]]
            i += 1

        h = torch.cat(h, dim=0)
        c = torch.cat(c, dim=0)

        return output, (h, c)

class StackedLSTMWithDropout(nn.Module):
    # Necessary for iterating through self.layers and dropout support
    #__constants__ = ['layers', 'num_layers']

    def __init__(self, num_layers, layer, dropout, first_layer_args, first_layer_kargs, other_layer_args, other_layer_kargs):
        super(StackedLSTMWithDropout, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, first_layer_kargs,
                                        other_layer_args, other_layer_kargs)
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if (num_layers == 1):
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input, states=None):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        if (states == None):
            states = [None for _ in range(self.num_layers)]

        h = []
        c = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            h += [out_state[0]]
            c += [out_state[1]]
            i += 1

        h = torch.cat(h, dim=0)
        c = torch.cat(c, dim=0)

        return output, (h, c)