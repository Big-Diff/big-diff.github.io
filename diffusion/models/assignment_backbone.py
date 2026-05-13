from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_layers import (
    BipartiteGraphConvolution,
    SparseKNNNodeAttentionLayer,
    _SinTimeEmbed,
)
from .graph_ops import batch_mean
from .slot_attention import SlotSelfAttentionBlock


@dataclass
class AssignmentContext:
    """Common tensors prepared by problem-specific CVRP/HFVRP wrappers."""

    graph: Any
    xt01: torch.Tensor
    t: torch.Tensor

    B: int
    node_batch: torch.Tensor
    veh_batch: torch.Tensor
    edge_index: torch.Tensor
    rev_edge_index: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_graph: torch.Tensor

    node_xy: torch.Tensor
    node_unit: torch.Tensor
    demand: torch.Tensor

    nn_edge_index: torch.Tensor
    nn_edge_attr: torch.Tensor

    h_g_init_feat: torch.Tensor
    problem_ctx: dict[str, Any]


class AssignmentBackbone(nn.Module):
    """Shared assignment denoising backbone for CVRP/HFVRP variants.

    Problem-specific subclasses only need to implement graph/context feature
    construction and dynamic edge features.  The recurrent GNN layer loop is
    shared here to avoid duplicated CVRP/HFVRP implementations.
    """

    def __init__(
        self,
        *,
        node_in_dim: int,
        veh_in_dim: int,
        edge_in_dim: int,
        graph_in_dim: int,
        edge_dyn_dim: int,
        hidden_dim: int = 192,
        n_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        biattn_heads: int = 4,
        biattn_dropout: float = 0.0,
        biattn_head_dim: int | None = None,
        use_n2n: bool = True,
        use_v2v: bool = False,
        use_global: bool = True,
        use_adaln: bool = False,
        n2n_knn_k: int = 8,
        dyn_refresh_every: int = 2,
        intra_every: int = 2,
        n2n_attn_heads: int = 4,
        n2n_attn_dropout: float = 0.05,
        n2n_attn_ffn_mult: int = 2,
        v2v_every: int = 2,
        v2v_heads: int = 4,
        v2v_dropout: float = 0.05,
        v2v_ffn_mult: int = 2,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.graph_in_dim = int(graph_in_dim)
        self.edge_dyn_dim = int(edge_dyn_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)

        self.use_n2n = bool(use_n2n)
        self.use_v2v = bool(use_v2v)
        self.use_global = bool(use_global)
        self.use_adaln = bool(use_adaln)
        self.use_edge_state_update = False
        self.n2n_knn_k = int(n2n_knn_k)

        self.dyn_refresh_every = max(1, int(dyn_refresh_every))
        self.intra_every = max(1, int(intra_every))
        self.v2v_every = max(1, int(v2v_every))

        self.n2n_attn_heads = int(n2n_attn_heads)
        self.n2n_attn_dropout = float(n2n_attn_dropout)
        self.n2n_attn_ffn_mult = int(n2n_attn_ffn_mult)

        self.v2v_heads = int(v2v_heads)
        self.v2v_dropout = float(v2v_dropout)
        self.v2v_ffn_mult = int(v2v_ffn_mult)

        self._build_layer_schedules()
        self._build_common_modules(
            node_in_dim=node_in_dim,
            veh_in_dim=veh_in_dim,
            edge_in_dim=edge_in_dim,
            graph_in_dim=graph_in_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            biattn_heads=biattn_heads,
            biattn_dropout=biattn_dropout,
            biattn_head_dim=biattn_head_dim,
        )

    def _build_layer_schedules(self) -> None:
        refresh_layer_ids = {0, self.n_layers - 1}
        refresh_layer_ids.update(range(0, self.n_layers, self.dyn_refresh_every))
        self.refresh_layer_ids = sorted(
            int(x) for x in refresh_layer_ids if 0 <= x < self.n_layers
        )
        self.refresh_layer_to_idx = {l: i for i, l in enumerate(self.refresh_layer_ids)}

        if self.use_n2n:
            intra_layer_ids = set(range(0, self.n_layers, self.intra_every))
            self.intra_layer_ids = sorted(
                int(x) for x in intra_layer_ids if 0 <= x < self.n_layers
            )
        else:
            self.intra_layer_ids = []
        self.intra_layer_to_idx = {l: i for i, l in enumerate(self.intra_layer_ids)}

        if self.use_v2v:
            v2v_layer_ids = set(range(0, self.n_layers, self.v2v_every))
            self.v2v_layer_ids = sorted(
                int(x) for x in v2v_layer_ids if 0 <= x < self.n_layers
            )
        else:
            self.v2v_layer_ids = []
        self.v2v_layer_to_idx = {l: i for i, l in enumerate(self.v2v_layer_ids)}

    def _build_common_modules(
        self,
        *,
        node_in_dim: int,
        veh_in_dim: int,
        edge_in_dim: int,
        graph_in_dim: int,
        hidden_dim: int,
        time_dim: int,
        biattn_heads: int,
        biattn_dropout: float,
        biattn_head_dim: int | None,
    ) -> None:
        self.time_emb = _SinTimeEmbed(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.veh_proj = nn.Sequential(
            nn.Linear(veh_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(graph_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        n_heads = int(max(1, biattn_heads))
        head_dim = biattn_head_dim
        if head_dim is None:
            if hidden_dim % n_heads != 0:
                n_heads, head_dim = 1, hidden_dim
            else:
                head_dim = hidden_dim // n_heads
        else:
            head_dim = int(head_dim)
            if n_heads * head_dim != hidden_dim:
                if hidden_dim % n_heads != 0:
                    n_heads, head_dim = 1, hidden_dim
                else:
                    head_dim = hidden_dim // n_heads

        self.v2n = nn.ModuleList(
            [
                BipartiteGraphConvolution(
                    hidden_dim,
                    edge_dim=hidden_dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dropout=float(biattn_dropout),
                    deg_norm_alpha=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.n2v = nn.ModuleList(
            [
                BipartiteGraphConvolution(
                    hidden_dim,
                    edge_dim=hidden_dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dropout=float(biattn_dropout),
                    deg_norm_alpha=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_n2n:
            self.n2n = nn.ModuleList(
                [
                    SparseKNNNodeAttentionLayer(
                        hidden_dim=hidden_dim,
                        edge_dim=7,
                        n_heads=self.n2n_attn_heads,
                        dropout=self.n2n_attn_dropout,
                        ffn_mult=self.n2n_attn_ffn_mult,
                    )
                    for _ in self.intra_layer_ids
                ]
            )
        else:
            self.n2n = None

        self.v2v = self._build_v2v_layers(hidden_dim)

        self.norm_n_intra = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_n_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_global = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_msg_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_out_norm = nn.LayerNorm(hidden_dim)

        self.adaLN = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 8))
                for _ in range(self.n_layers)
            ]
        ) if self.use_adaln else None

        self.pre_score = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in self.refresh_layer_ids
            ]
        )

        self.edge_dyn_proj = nn.Sequential(
            nn.Linear(self.edge_dyn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(self.edge_dyn_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.edge_delta_bias = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3 + self.edge_dyn_dim + 1, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.vehicle_global = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.global_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_v2v_layers(self, hidden_dim: int) -> nn.ModuleList | None:
        if not self.use_v2v:
            return None
        return nn.ModuleList(
            [
                SlotSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    n_heads=self.v2v_heads,
                    dropout=self.v2v_dropout,
                    ffn_mult=self.v2v_ffn_mult,
                )
                for _ in self.v2v_layer_ids
            ]
        )

    def _apply_v2v(
        self,
        h_v: torch.Tensor,
        ctx: AssignmentContext,
        layer_idx: int,
        v2v_idx: int,
    ) -> torch.Tensor:
        if self.v2v is None:
            return h_v
        return self.v2v[v2v_idx](h_v, ctx.B)

    def _build_context(self, graph, xt_edge: torch.Tensor, t: torch.Tensor) -> AssignmentContext:
        raise NotImplementedError

    def _build_initial_dynamic_context(
        self,
        ctx: AssignmentContext,
        h_v: torch.Tensor,
        h_n: torch.Tensor,
        h_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _refresh_dynamic_context(
        self,
        refresh_idx: int,
        ctx: AssignmentContext,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_e_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _update_edge_state(
        self,
        h_e: torch.Tensor,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_g: torch.Tensor,
        ctx: AssignmentContext,
        cached_edge_dyn_h: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return h_e

    def forward(self, graph, xt_edge: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ctx = self._build_context(graph, xt_edge, t)

        h_n = self.node_proj(ctx.graph.node_features.float())
        h_v = self.veh_proj(ctx.graph.veh_features.float())
        h_e = self.edge_proj(
            torch.cat([ctx.graph.edge_attr.float(), ctx.xt01.view(-1, 1)], dim=-1)
        )

        t_emb = self.time_proj(self.time_emb(ctx.t))
        h_g = self.global_proj(ctx.h_g_init_feat)

        cached_edge_dyn_raw, cached_edge_dyn_h, cached_edge_bias = (
            self._build_initial_dynamic_context(ctx, h_v, h_n, h_e)
        )
        edge_bias_static = cached_edge_bias

        for layer_idx in range(self.n_layers):
            cond = t_emb + h_g

            if self.use_adaln:
                ns, nb, vs, vb, es, eb, gn, gv = self.adaLN[layer_idx](cond).chunk(8, dim=-1)
                h_n_in = h_n * (1.0 + ns[ctx.node_batch]) + nb[ctx.node_batch]
                h_v_in = h_v * (1.0 + vs[ctx.veh_batch]) + vb[ctx.veh_batch]
                h_e_in = h_e * (1.0 + es[ctx.edge_graph]) + eb[ctx.edge_graph]
                gate_n = torch.sigmoid(gn)
                gate_v = torch.sigmoid(gv)
            else:
                h_n_in, h_v_in, h_e_in = h_n, h_v, h_e
                gate_n = torch.ones_like(h_g)
                gate_v = torch.ones_like(h_g)

            intra_idx = self.intra_layer_to_idx.get(layer_idx, None)
            if intra_idx is not None:
                n2n_h, _ = self.n2n[intra_idx](h_n_in, ctx.nn_edge_index, ctx.nn_edge_attr)
                h_n = self.norm_n_intra[layer_idx](
                    h_n + F.dropout(n2n_h - h_n_in, p=self.dropout, training=self.training)
                )
                h_n_in = h_n

            refresh_idx = self.refresh_layer_to_idx.get(layer_idx, None)
            if refresh_idx is not None:
                cached_edge_dyn_raw, cached_edge_dyn_h, cached_edge_bias = self._refresh_dynamic_context(
                    refresh_idx, ctx, h_v_in, h_n_in, h_e_in
                )

            h_e = self._update_edge_state(
                h_e=h_e,
                h_v_in=h_v_in,
                h_n_in=h_n_in,
                h_g=h_g,
                ctx=ctx,
                cached_edge_dyn_h=cached_edge_dyn_h,
                layer_idx=layer_idx,
            )

            edge_base = h_e if self.use_edge_state_update else h_e_in
            edge_msg = self.edge_msg_norm[layer_idx](edge_base + cached_edge_dyn_h)

            edge_bias_delta = self.edge_delta_bias[layer_idx](
                torch.cat(
                    [
                        h_v_in[ctx.src],
                        h_n_in[ctx.dst],
                        h_g[ctx.edge_graph],
                        cached_edge_dyn_raw,
                        ctx.xt01.view(-1, 1),
                    ],
                    dim=-1,
                )
            )
            total_edge_bias = edge_bias_static + cached_edge_bias + edge_bias_delta

            v2n_out = self.v2n[layer_idx](
                h_v_in, ctx.edge_index, edge_msg, h_n_in, edge_logit_bias=total_edge_bias
            )
            n2v_out = self.n2v[layer_idx](
                h_n_in, ctx.rev_edge_index, edge_msg, h_v_in, edge_logit_bias=total_edge_bias
            )

            h_n = self.norm_n_cross[layer_idx](
                h_n + gate_n[ctx.node_batch] * F.dropout(v2n_out, p=self.dropout, training=self.training)
            )
            h_v = self.norm_v_cross[layer_idx](
                h_v + gate_v[ctx.veh_batch] * F.dropout(n2v_out, p=self.dropout, training=self.training)
            )

            v2v_idx = self.v2v_layer_to_idx.get(layer_idx, None)
            if v2v_idx is not None:
                h_v = self._apply_v2v(h_v, ctx, layer_idx, v2v_idx)

            node_mean = batch_mean(h_n, ctx.node_batch, ctx.B)
            veh_mean = batch_mean(h_v, ctx.veh_batch, ctx.B)

            vg_out = self.vehicle_global[layer_idx](
                torch.cat([h_v, h_g[ctx.veh_batch], veh_mean[ctx.veh_batch]], dim=-1)
            )
            vg_delta, vg_gate = vg_out.chunk(2, dim=-1)
            h_v = self.norm_v_global[layer_idx](
                h_v + torch.sigmoid(vg_gate) * F.dropout(vg_delta, p=self.dropout, training=self.training)
            )

            if self.use_global:
                h_g = h_g + self.global_update[layer_idx](torch.cat([h_g, node_mean, veh_mean], dim=-1))

        h_e_final = self.edge_out_norm(h_e + cached_edge_dyn_h)
        edge_feat = torch.cat(
            [
                h_v[ctx.src],
                h_n[ctx.dst],
                h_e_final,
                h_v[ctx.src] * h_n[ctx.dst],
                h_v[ctx.src] * h_e_final,
                h_n[ctx.dst] * h_e_final,
                h_g[ctx.edge_graph],
            ],
            dim=-1,
        )
        return self.edge_head(edge_feat).squeeze(-1)


__all__ = ["AssignmentContext", "AssignmentBackbone"]
