import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM_fixed_len(nn.Module) :
    def __init__(self, input_size, n_layers, hidden_dim, output_size, device, bidirectional=0) :
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)
    
    
#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
#         h0 = h0.to(self.device)

#         # Initialize cell state
#         c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
#         c0 = c0.to(self.device)

#         # 28 time steps
#         # We need to detach as we are doing truncated backpropagation through time (BPTT)
#         # If we don't, we'll backprop all the way to the start even after going through another batch
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

#         # Index hidden state of last time step
#         # out.size() --> 100, 28, 100
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
#         out = self.linear(out[:, -1, :]) 
#         # out.size() --> 100, 10
#         return out
    
    def forward(self, batch):
        lstm_out, (h, c) = self.lstm(batch)
        lstm_out = self.dropout(lstm_out)
     
        output = self.linear(h[-1])
        return output, hidden
    
