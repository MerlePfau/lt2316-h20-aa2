import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM_fixed_len(nn.Module) :
    def __init__(self, input_size, n_layers, hidden_dim, output_size, bidirectional=0) :
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
    
    def forward(self, batch):
        lstm_out, (h, c) = self.lstm(batch)    
        o = self.non_lin(h)
        output = self.linear(o)
        return output
    
