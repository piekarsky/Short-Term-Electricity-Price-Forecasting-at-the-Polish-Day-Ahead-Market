import torch
import torch.nn as nn

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DNN(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        out = self.main(x)
        return out


