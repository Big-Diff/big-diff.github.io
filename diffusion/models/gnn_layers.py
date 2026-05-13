"""Reusable neural layers for assignment GNN backbones.

This module contains trainable building blocks shared by CVRP/HFVRP denoisers.
Problem-specific feature construction should live outside this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class SinTimeEmbed(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float().view(-1, 1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device, dtype=torch.float32)
            / max(1, half - 1)
        ).view(1, -1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# Backward-compatible private alias for the current code style.
_SinTimeEmbed = SinTimeEmbed


class BipartiteGraphConvolution(MessagePassing):
    """Bipartite message passing with multi-head attention and edge injection.

    Edge convention:
        ``edge_index[0]`` = left/source indices, e.g. vehicles.
        ``edge_index[1]`` = right/destination indices, e.g. customers.

    Attention is normalized per destination node:
        ``alpha_{j->i} = softmax_j(<q_i, k_j + k_e> / sqrt(d))``.
    """

    def __init__(
        self,
        embd_size: int,
        edge_dim: int = 4,
        *,
        n_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
        deg_norm_alpha: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.embd_size = int(embd_size)
        self.edge_dim = int(edge_dim)

        self.n_heads = int(max(1, n_heads))
        if head_dim is None:
            if self.embd_size % self.n_heads != 0:
                self.n_heads = 1
                self.head_dim = self.embd_size
            else:
                self.head_dim = self.embd_size // self.n_heads
        else:
            self.head_dim = int(head_dim)
            if self.n_heads * self.head_dim != self.embd_size:
                if self.embd_size % self.n_heads != 0:
                    self.n_heads = 1
                    self.head_dim = self.embd_size
                else:
                    self.head_dim = self.embd_size // self.n_heads

        self.dropout = float(dropout)
        self.deg_norm_alpha = float(deg_norm_alpha)

        self.Wq = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)

        self.Wke = nn.Linear(self.edge_dim, self.n_heads * self.head_dim, bias=False)
        self.Wve = nn.Linear(self.edge_dim, self.n_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.embd_size, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        # Kept as BatchNorm for behavioral compatibility with the current model.
        # Consider making this configurable during a later cleanup step.
        self.post_conv_module = nn.Sequential(nn.BatchNorm1d(self.embd_size))

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
        )

    def forward(
        self,
        left_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        right_features: torch.Tensor,
        edge_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            left=left_features,
            right=right_features,
            edge_features=edge_features,
            edge_logit_bias=edge_logit_bias,
        )

        out = out.reshape(right_features.size(0), self.n_heads * self.head_dim)
        out = self.out_proj(out)

        if self.deg_norm_alpha > 0:
            dst = edge_indices[1]
            deg = torch.bincount(dst, minlength=right_features.size(0)).float().clamp_min(1.0)
            out = out / (deg.unsqueeze(-1) ** self.deg_norm_alpha)

        out = self.post_conv_module(out)
        return self.output_module(torch.cat([out, right_features], dim=-1))

    def message(
        self,
        right_i: torch.Tensor,
        left_j: torch.Tensor,
        edge_features: torch.Tensor,
        edge_logit_bias: torch.Tensor | None,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        q = self.Wq(right_i).view(-1, self.n_heads, self.head_dim)
        k = self.Wk(left_j).view(-1, self.n_heads, self.head_dim)
        v = self.Wv(left_j).view(-1, self.n_heads, self.head_dim)
        ke = self.Wke(edge_features).view(-1, self.n_heads, self.head_dim)
        ve = self.Wve(edge_features).view(-1, self.n_heads, self.head_dim)

        logits = (q * (k + ke)).sum(dim=-1) / math.sqrt(float(self.head_dim))

        if edge_logit_bias is not None:
            if edge_logit_bias.dim() == 1:
                logits = logits + edge_logit_bias.unsqueeze(-1)
            elif edge_logit_bias.dim() == 2:
                logits = logits + edge_logit_bias
            else:
                raise ValueError("edge_logit_bias must have shape (E,), (E,1), or (E,H)")

        alpha = pyg_softmax(logits, index=index, ptr=ptr, num_nodes=size_i)
        alpha = self.attn_dropout(alpha)
        return (v + ve) * alpha.unsqueeze(-1)

class SparseKNNNodeAttentionLayer(nn.Module):
    """Sparse edge-aware attention over customer KNN graph.

    KNN edge convention:
        edge_index[0] = receiver/current customer node
        edge_index[1] = source/neighbor customer node

    For every receiver node i, this layer performs attention over its KNN
    neighbors j and aggregates messages j -> i.

    This convention is intentionally fixed for all n2n attention layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 7,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.n_heads = int(max(1, n_heads))

        if self.hidden_dim % self.n_heads != 0:
            self.n_heads = 1

        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.Wke = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.Wve = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.edge_bias = nn.Linear(self.edge_dim, self.n_heads, bias=False)

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)
        self.norm_e = nn.LayerNorm(self.hidden_dim)

        mid_dim = int(ffn_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, self.hidden_dim),
        )

        self.attn_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = h

        if edge_index.numel() == 0:
            zero = self.out_proj(self.Wv(h) * 0.0)
            h = self.norm_attn(h + zero)
            h = self.norm_ffn(h + self.resid_dropout(self.ffn(h)))
            e_out = edge_attr.new_zeros((0, self.hidden_dim))
            return h, e_out

        cur = edge_index[0].long()
        nbr = edge_index[1].long()

        num_edges = int(cur.numel())
        num_heads = self.n_heads
        head_dim = self.head_dim

        edge_attr = edge_attr.to(dtype=h.dtype)

        q = self.Wq(h[cur]).view(num_edges, num_heads, head_dim)
        k = self.Wk(h[nbr]).view(num_edges, num_heads, head_dim)
        v = self.Wv(h[nbr]).view(num_edges, num_heads, head_dim)

        ke = self.Wke(edge_attr).view(num_edges, num_heads, head_dim)
        ve = self.Wve(edge_attr).view(num_edges, num_heads, head_dim)
        eb = self.edge_bias(edge_attr)

        logits = (q * (k + ke)).sum(dim=-1) / math.sqrt(float(head_dim))
        logits = logits + eb

        alpha = pyg_softmax(logits, index=cur, num_nodes=h.size(0))
        alpha = self.attn_dropout(alpha)

        msg = (v + ve) * alpha.unsqueeze(-1)
        msg = msg.reshape(num_edges, self.hidden_dim)

        agg = msg.new_zeros((h.size(0), self.hidden_dim))
        agg.index_add_(0, cur, msg)

        out = self.out_proj(agg)
        h = self.norm_attn(h_in + self.resid_dropout(out))
        h = self.norm_ffn(h + self.resid_dropout(self.ffn(h)))

        e_out = self.norm_e(F.silu((ke + ve).reshape(num_edges, self.hidden_dim)))
        return h, e_out


__all__ = [
    "SinTimeEmbed",
    "_SinTimeEmbed",
    "BipartiteGraphConvolution",
    "SparseKNNNodeAttentionLayer",
]
