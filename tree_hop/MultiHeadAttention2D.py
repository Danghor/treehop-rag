import torch
from torch import nn as nn

from tree_hop.AttentionHead2D import AttentionHead2D


class MultiHeadAttention2D(nn.Module):
    def __init__(
        self,
        input_size,
        attn_size,
        mlp_size,
        num_mlp=1,
        num_heads=1,
        bias=True,
        dropout=0.1
    ):
        if not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")

        super(MultiHeadAttention2D, self).__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            AttentionHead2D(input_size, attn_size, mlp_size,
                            num_mlp=num_mlp, bias=bias, dropout=dropout)
            for _ in range(num_heads)
        ])

    def forward(self, Q, K, V):
        lst_attn_out = []
        for attn_head in self.heads:
            out = attn_head(Q, K, V)
            lst_attn_out.append(out)

        attn_out = torch.cat(lst_attn_out, dim=-1)
        return attn_out
