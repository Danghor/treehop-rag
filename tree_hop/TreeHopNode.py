from torch import nn as nn

from tree_hop.MultiHeadAttention2D import MultiHeadAttention2D


class TreeHopNode(nn.Module):
    def __init__(self, embed_size, g_size, mlp_size, n_mlp=1, n_head=1):
        super(TreeHopNode, self).__init__()
        self.update_gate = MultiHeadAttention2D(
            embed_size, g_size, mlp_size,
            num_heads=n_head,
            num_mlp=n_mlp,
            dropout=0.
        )
        self.update_attn_scale = nn.Linear(g_size * n_head, embed_size, bias=False)

    def reduce_func(self, nodes):
        # message passing
        Q = nodes.mailbox["q"].clone().squeeze(1)         # last query
        K = nodes.data["rep"]           # this ctx
        V_update = nodes.data["rep"]           # this ctx

        update_gate = self.update_gate(Q, K, V_update)

        h = Q - K + self.update_attn_scale(update_gate)
        return {"h": h}
