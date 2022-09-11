import torch
from torch import nn
from .config import *


class LSTMModel(nn.Module):
    def __init__(self, input_size=input_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=self.num_layers,
            dropout=0.1,
            batch_first=True,
        )

        self.fc = nn.Linear(self.input_size, self.input_size)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        out = self.fc(output)

        return out, state

    def init_state(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.input_size, requires_grad=True),
                torch.zeros(self.num_layers, batch_size, self.input_size, requires_grad=True))
