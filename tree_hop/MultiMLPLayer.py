from torch import nn as nn

from tree_hop.ResNet import ResNet


class MultiMLPLayer(nn.Module):
    def __init__(
        self,
        input_size,
        mlp_size,
        num_layers: int = 1
    ):
        super(MultiMLPLayer, self).__init__()
        self.layers = nn.Sequential()

        for i in range(num_layers):
            if i == 0 and input_size != mlp_size:
                self.layers.append(nn.Linear(input_size, mlp_size))
                self.layers.append(ResNet(mlp_size))
            else:
                self.layers.append(ResNet(mlp_size))

    def forward(self, x):
        x_out = self.layers(x)
        return x_out
