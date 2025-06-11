import torch
import torch.nn as nn

class LSTM_Classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x, _ = self.lstm(x)
        x = x.mean(dim=1)          # average over time
        x = self.linear(x)         # project to classes
        return x
