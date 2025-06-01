import torch
from torch import nn as nn
from torch.nn import functional as F

from tree_hop.MultiMLPLayer import MultiMLPLayer


class AttentionHead2D(nn.Module):
    def __init__(
        self,
        input_size,
        attn_size,
        mlp_size,
        *,
        bias=True,
        num_mlp=1,
        dropout=0.1
    ):
        super(AttentionHead2D, self).__init__()
        self.W_Q = nn.Linear(input_size, attn_size, bias=bias)
        self.W_K = nn.Linear(input_size, attn_size, bias=bias)
        self.W_V = nn.Linear(input_size, attn_size, bias=bias)

        # self.activate = nn.ReLU()
        self.mlp = MultiMLPLayer(attn_size, mlp_size, num_layers=num_mlp)
        self.dropout = nn.Dropout(dropout)
        self.mlp_scale = nn.Linear(mlp_size, attn_size, bias=bias)

    def forward(self, Q, K, V):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        if Q.dim() == 3:
            QK = torch.einsum("bud,bud->bd", Q, K)
        elif Q.dim() == 2:
            QK = Q * K
        else:
            raise IndexError(f"Not a supported input dimension: {Q.dim()}")

        # instead of matmul in 3D, use elementwise mul, then normalize
        scores = QK / Q.shape[1] ** 0.5
        attn = F.softmax(scores, dim=-1)
        attn_out = self.dropout(attn) * V

        mlp_out = self.mlp(attn_out)

        return self.mlp_scale(mlp_out) + attn_out
