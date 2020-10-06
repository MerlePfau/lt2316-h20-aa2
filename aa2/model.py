import torch.nn as nn

class LSTM_fixed_len(nn.Module) :
    def __init__(self, vocab_size, n_layers, hidden_dim, label_size, bidirectional) :
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, label_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        output = self.linear(ht[-1])
        return self.softmax(output)