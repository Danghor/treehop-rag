import torch
from torch import nn as nn


class ResNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, input_size)
        self.activate = nn.ReLU()
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # post Norm
        x_norm = self.layer_norm(x)
        x = x + self.activate(self.linear(x_norm))
        return x
