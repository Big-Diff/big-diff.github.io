"""HF-lite V4 backbones for HFVRP.

This file is designed to be added alongside the current project gnn.py and used
from the shared `_build_assignment_model()` factory.

Design goals:
- start from the CVRP-oriented V4 skeleton,
- keep the same light `v2n + n2v + optional n2n + global` macro-architecture,
- minimally extend it to HFVRP by consuming the current HF dataset fields:
    * graph.vehicle_capacity
    * graph.vehicle_fixed_cost
    * graph.vehicle_unit_distance_cost
    * graph.vehicle_tier
    * graph.K_used
    * graph.depot_xy
- optionally add explicit edge-state updates (`_EdgeUpd`) without bringing back
  the heavier A-full modules such as `xt_proj` or hidden-state-only edge bias.

Typical usage from the shared factory:
    from .models.gnn_hf_lite_v4 import (
        EdgeBipartiteDenoiserV4_HF_Lite,
        EdgeBipartiteDenoiserV4_HF_Lite_EdgeUpd,
    )

Notes:
- `graph_in_dim=10` is recommended for HF usage.
- `veh_in_dim=7`, `node_in_dim=10`, `edge_in_dim=8` remain compatible with the
  current HF dataset builder.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

# Reuse the stable building blocks from the project's main gnn.py.
# This keeps the new file small, readable, and easy to maintain.
from .gnn import (
    BipartiteGraphConvolution,
    LiteGatedGCNLayer,
    SparseKNNNodeAttentionLayer,
    _SinTimeEmbed,
)

class TypeAwareSlotSelfAttentionBlock(nn.Module):
    """Type-aware graph-local slot self-attention for HFVRP.

    This is v2v message passing among candidate vehicle slots. Unlike the CVRP
    version, attention logits are biased by pairwise fleet-type attributes:
        same tier, capacity gap, fixed-cost gap, unit-cost gap, tier gap.

    No per-graph Python loop is used.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 2,
        pair_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(max(1, n_heads))

        if self.hidden_dim % self.n_heads != 0:
            self.n_heads = 1

        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # pair features:
        #   same_tier,
        #   |cap_i-cap_j|,
        #   |fixed_i-fixed_j|,
        #   |unit_i-unit_j|,
        #   |tier_i-tier_j|
        pair_dim = 5
        mid = int(pair_hidden_dim) if pair_hidden_dim is not None else max(32, self.hidden_dim // 4)

        self.pair_bias_mlp = nn.Sequential(
            nn.Linear(pair_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, self.n_heads),
        )

        # Stable warm start: initially behaves like ordinary slot attention.
        nn.init.zeros_(self.pair_bias_mlp[-1].weight)
        nn.init.zeros_(self.pair_bias_mlp[-1].bias)

        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)

        ffn_dim = int(ffn_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ffn_dim, self.hidden_dim),
        )

        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    @staticmethod
    def _reshape_slot_attr(x: torch.Tensor, B: int, K: int, dtype: torch.dtype) -> torch.Tensor:
        return x.to(dtype=dtype).view(B, K)

    def forward(
        self,
        h_v: torch.Tensor,
        fleet_ctx: dict,
        veh_batch: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        del veh_batch  # fixed-size graph order is used; kept for API clarity.

        if h_v.numel() == 0:
            return h_v

        B = int(B)
        total_slots = int(h_v.size(0))

        if B <= 0:
            return h_v

        if total_slots % B != 0:
            raise ValueError(
                f"TypeAwareSlotSelfAttentionBlock expects fixed K per graph, "
                f"but got total_slots={total_slots}, B={B}."
            )

        K = total_slots // B
        if K <= 1:
            return h_v

        H = self.hidden_dim
        dtype = h_v.dtype

        hv = h_v.view(B, K, H)

        # ------------------------------------------------------------------
        # Build normalized slot attributes, all batched as [B, K].
        # ------------------------------------------------------------------
        cap = self._reshape_slot_attr(fleet_ctx["cap_v"], B, K, dtype)
        cap_rel = cap / cap.max(dim=1, keepdim=True).values.clamp_min(1e-6)

        fixed_rel = self._reshape_slot_attr(fleet_ctx["fixed_rel"], B, K, dtype)
        unit_rel = self._reshape_slot_attr(fleet_ctx["unit_rel"], B, K, dtype)
        tier_rel = self._reshape_slot_attr(fleet_ctx["tier_rel"], B, K, dtype)

        if "tier_raw" in fleet_ctx:
            tier_raw = self._reshape_slot_attr(fleet_ctx["tier_raw"], B, K, dtype)
        else:
            tier_raw = tier_rel

        # slot_attr: [B, K, 4]
        slot_attr = torch.stack(
            [cap_rel, fixed_rel, unit_rel, tier_rel],
            dim=-1,
        )

        ai = slot_attr.unsqueeze(2)  # [B, K, 1, 4]
        aj = slot_attr.unsqueeze(1)  # [B, 1, K, 4]
        abs_diff = (ai - aj).abs()  # [B, K, K, 4]

        same_tier = (
            tier_raw.unsqueeze(2).eq(tier_raw.unsqueeze(1))
            .to(dtype=dtype)
            .unsqueeze(-1)
        )  # [B, K, K, 1]

        pair_feat = torch.cat([same_tier, abs_diff], dim=-1)  # [B, K, K, 5]

        # [B, K, K, heads] -> [B, heads, K, K]
        pair_bias = self.pair_bias_mlp(pair_feat).permute(0, 3, 1, 2).contiguous()
        pair_bias = pair_bias.to(dtype=dtype)

        # ------------------------------------------------------------------
        # Batched slot self-attention.
        # ------------------------------------------------------------------
        qkv = self.qkv_proj(hv)
        qkv = qkv.view(B, K, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, heads, K, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=pair_bias,
            dropout_p=attn_dropout,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, K, H)
        out = self.out_proj(out)

        hv = self.norm_attn(hv + self.resid_dropout(out))
        hv = self.norm_ffn(hv + self.resid_dropout(self.ffn(hv)))

        return hv.view(total_slots, H)


class EdgeBipartiteDenoiserV4_HF_Lite(nn.Module):
    """B-style V4 backbone with minimal heterogeneous-fleet augmentation.

    Compared with the CVRP V4 base, this class adds only the heterogeneity that
    is necessary for HFVRP:
      - per-slot capacity
      - per-slot fixed cost
      - per-slot unit-distance cost
      - per-slot tier

    It intentionally does *not* include:
      - xt_proj
      - A-full per-layer edge_update
      - A-full hidden-state-only edge bias

    The main inductive bias remains V4-like:
      - static edge embedding + lightweight dynamic context
      - optional n2n local propagation
      - bipartite vehicle<->customer attention
      - light graph-level vehicle gating
    """

    def __init__(
        self,
        node_in_dim: int = 10,
        veh_in_dim: int = 7,
        edge_in_dim: int = 8,
        graph_in_dim: int = 10,
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

        n2n_mode: str = "gated",
        n2n_attn_heads: int = 4,
        n2n_attn_dropout: float = 0.05,
        n2n_attn_ffn_mult: int = 2,

        v2v_every: int = 2,
        v2v_heads: int = 4,
        v2v_dropout: float = 0.05,
        v2v_ffn_mult: int = 2,

        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.graph_in_dim = int(graph_in_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.use_n2n = bool(use_n2n)
        self.use_global = bool(use_global)
        self.use_adaln = bool(use_adaln)
        self.use_v2v = bool(use_v2v)
        self.n2n_knn_k = int(n2n_knn_k)

        self.v2v_every = max(1, int(v2v_every))
        self.v2v_heads = int(v2v_heads)
        self.v2v_dropout = float(v2v_dropout)
        self.v2v_ffn_mult = int(v2v_ffn_mult)

        self.n2n_mode = str(n2n_mode).lower()
        if self.n2n_mode in {"attention", "knn_attention", "knn_attn"}:
            self.n2n_mode = "attn"
        if self.n2n_mode not in {"gated", "attn"}:
            raise ValueError(f"Unsupported n2n_mode={n2n_mode!r}. Use 'gated' or 'attn'.")

        self.n2n_attn_heads = int(n2n_attn_heads)
        self.n2n_attn_dropout = float(n2n_attn_dropout)
        self.n2n_attn_ffn_mult = int(n2n_attn_ffn_mult)

        # 5 base dims from V4:
        #   dist_centroid, angle_diff, load_after, overload, radius_inc
        # + 3 heterogeneity dims:
        #   fixed_cost_rel, unit_cost_rel, tier_rel
        self.edge_dyn_dim = 8

        self.dyn_refresh_every = max(1, int(dyn_refresh_every))
        self.intra_every = max(1, int(intra_every))
        self._printed_n2n_knn = False

        refresh_layer_ids = {0, self.n_layers - 1}
        refresh_layer_ids.update(range(0, self.n_layers, self.dyn_refresh_every))
        self.refresh_layer_ids = sorted(
            int(x) for x in refresh_layer_ids if 0 <= x < self.n_layers
        )
        self.refresh_layer_to_idx = {l: i for i, l in enumerate(self.refresh_layer_ids)}

        # n2n layer schedule.
        if self.use_n2n:
            intra_layer_ids = set(range(0, self.n_layers, self.intra_every))
            self.intra_layer_ids = sorted(
                int(x) for x in intra_layer_ids if 0 <= x < self.n_layers
            )
        else:
            self.intra_layer_ids = []
        self.intra_layer_to_idx = {l: i for i, l in enumerate(self.intra_layer_ids)}

        # v2v layer schedule.
        if self.use_v2v:
            v2v_layer_ids = set(range(0, self.n_layers, self.v2v_every))
            self.v2v_layer_ids = sorted(
                int(x) for x in v2v_layer_ids if 0 <= x < self.n_layers
            )
        else:
            self.v2v_layer_ids = []
        self.v2v_layer_to_idx = {l: i for i, l in enumerate(self.v2v_layer_ids)}

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

        H = int(max(1, biattn_heads))
        hd = biattn_head_dim
        if hd is None:
            if hidden_dim % H != 0:
                H, hd = 1, hidden_dim
            else:
                hd = hidden_dim // H
        else:
            hd = int(hd)
            if H * hd != hidden_dim:
                if hidden_dim % H != 0:
                    H, hd = 1, hidden_dim
                else:
                    hd = hidden_dim // H

        self.v2n = nn.ModuleList(
            [
                BipartiteGraphConvolution(
                    hidden_dim,
                    edge_dim=hidden_dim,
                    n_heads=H,
                    head_dim=hd,
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
                    n_heads=H,
                    head_dim=hd,
                    dropout=float(biattn_dropout),
                    deg_norm_alpha=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_n2n:
            if self.n2n_mode == "attn":
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
                self.n2n = nn.ModuleList(
                    [
                        LiteGatedGCNLayer(
                            hidden_dim=hidden_dim,
                            edge_dim=7,
                            dropout=float(dropout),
                        )
                        for _ in self.intra_layer_ids
                    ]
                )
        else:
            self.n2n = None

        self.v2v = nn.ModuleList(
            [
                TypeAwareSlotSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    n_heads=self.v2v_heads,
                    dropout=self.v2v_dropout,
                    ffn_mult=self.v2v_ffn_mult,
                )
                for _ in self.v2v_layer_ids
            ]
        ) if self.use_v2v else None

        self.norm_n_intra = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_n_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_global = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_msg_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_out_norm = nn.LayerNorm(hidden_dim)

        self.adaLN = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 8)) for _ in range(self.n_layers)]
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

        print("[LOAD] EdgeBipartiteDenoiserV4_HF_Lite = B-style V4 + minimal hetero fleet attrs")

    # ------------------------------------------------------------------
    # generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_mean(x: torch.Tensor, batch: torch.Tensor, B: int):
        out = x.new_zeros((B, x.size(-1)))
        cnt = x.new_zeros((B, 1))
        out.index_add_(0, batch, x)
        cnt.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
        return out / cnt.clamp_min(1.0)

    @staticmethod
    def _scatter_add_1d(index: torch.Tensor, val: torch.Tensor, out_size: int):
        out = val.new_zeros((out_size,))
        out.index_add_(0, index, val)
        return out

    @staticmethod
    def _scatter_add_2d(index: torch.Tensor, val: torch.Tensor, out_size: int):
        out = val.new_zeros((out_size, val.size(-1)))
        out.index_add_(0, index, val)
        return out

    @staticmethod
    def _normalize_per_batch_max(x: torch.Tensor, batch_idx: torch.Tensor, B: int, eps: float = 1e-6):
        if x.numel() == 0:
            return x
        max_per = torch.full((B,), -float("inf"), device=x.device, dtype=x.dtype)
        max_per.scatter_reduce_(0, batch_idx, x, reduce="amax", include_self=True)
        max_per = max_per.clamp_min(eps)
        return x / max_per[batch_idx]

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(x, min=0.0))

    @staticmethod
    def _row_normalize_by_dst(
            x: torch.Tensor,
            dst: torch.Tensor,
            num_nodes: int,
            eps: float = 1e-12,
    ):
        """Normalize edge values per customer row.

        Args:
            x:
                Edge values, shape [E].
            dst:
                Destination customer index of each edge, shape [E].
            num_nodes:
                Number of customer nodes.
            eps:
                Numerical stability constant.

        Returns:
            Row-normalized edge values, shape [E].
        """
        x = torch.nan_to_num(
            x.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        denom = x.new_zeros((num_nodes,))
        denom.index_add_(0, dst, x)

        p = x / denom[dst].clamp_min(eps)

        # Fallback for all-zero rows.
        # Compute it unconditionally to avoid bool(tensor.any()) GPU sync.
        ones = torch.ones_like(x)
        cnt = x.new_zeros((num_nodes,))
        cnt.index_add_(0, dst, ones)
        uniform = ones / cnt[dst].clamp_min(1.0)

        bad = denom[dst] <= eps
        p = torch.where(bad, uniform, p)

        return p
    @staticmethod
    def _build_knn_edges_by_batch(
            node_xy: torch.Tensor,
            batch_ids: torch.Tensor,
            k: int,
            B: int | None = None,
    ):
        """Build node-node KNN edges for fixed-size batched CVRP instances.

        Assumption:
            Nodes are stored in PyG batch order:

                graph0 nodes, graph1 nodes, ..., graph(B-1) nodes

            and every graph in the batch has the same number of nodes N.

        Edge direction is kept identical to the old implementation:

            src = current node
            dst = one of its K nearest neighbours
        """
        device = node_xy.device
        k = int(k)

        if k <= 0 or node_xy.numel() == 0:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        total_nodes = int(node_xy.size(0))

        if B is None:
            if batch_ids.numel() == 0:
                B = 0
            else:
                # One scalar read only when B is not provided.
                # In forward(), pass B explicitly to avoid recomputing it here.
                B = int(batch_ids[-1].detach().item()) + 1

        B = int(B)

        if B <= 0:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        if total_nodes % B != 0:
            raise ValueError(
                f"Fixed-batch KNN requires total_nodes % B == 0, "
                f"got total_nodes={total_nodes}, B={B}."
            )

        N = total_nodes // B

        if N <= 1:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        kk = min(k, N - 1)

        # [B, N, 2]
        xy = node_xy.reshape(B, N, node_xy.size(-1))

        # [B, N, N]
        dist = torch.cdist(xy, xy, p=2)

        diag = torch.arange(N, device=device)
        dist[:, diag, diag] = float("inf")

        # nn_idx[b, i, :] gives nearest-neighbour local indices of node i.
        # Shape: [B, N, kk]
        nn_idx = torch.topk(dist, kk, largest=False, dim=-1).indices

        base = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * N

        # Preserve old edge direction:
        #   src = current node i
        #   dst = neighbour nn_idx[b, i, :]
        src_local = torch.arange(N, device=device, dtype=torch.long).view(1, N, 1).expand(B, N, kk)
        dst_local = nn_idx

        src = (base + src_local).reshape(-1)
        dst = (base + dst_local).reshape(-1)

        return torch.stack([src, dst], dim=0)
    @staticmethod
    def _n2n_edge_attr(node_xy: torch.Tensor, demand: torch.Tensor, nn_edge_index: torch.Tensor):
        if nn_edge_index.numel() == 0:
            return torch.empty((0, 7), device=node_xy.device, dtype=node_xy.dtype)
        src, dst = nn_edge_index[0], nn_edge_index[1]
        dxy = node_xy[src] - node_xy[dst]
        dist = torch.sqrt((dxy ** 2).sum(dim=-1) + 1e-12)
        inv = 1.0 / (dist + 1e-6)
        di = demand[src].to(node_xy.dtype)
        dj = demand[dst].to(node_xy.dtype)
        return torch.cat(
            [
                dxy,
                dist.unsqueeze(-1),
                inv.unsqueeze(-1),
                di.unsqueeze(-1),
                dj.unsqueeze(-1),
                (di - dj).abs().unsqueeze(-1),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # fleet semantics
    # ------------------------------------------------------------------
    def _build_fleet_static_context_light(self, graph, veh_batch: torch.Tensor, B: int, device) -> Dict[str, torch.Tensor]:
        if hasattr(graph, "vehicle_capacity") and graph.vehicle_capacity is not None:
            cap_v = graph.vehicle_capacity.float().to(device).view(-1)
        else:
            cap_b = graph.capacity.float().to(device).view(-1)
            if cap_b.numel() == 1 and B > 1:
                cap_b = cap_b.repeat(B)
            cap_v = cap_b[veh_batch]
        cap_v = cap_v.clamp_min(1e-6)

        if hasattr(graph, "vehicle_fixed_cost") and graph.vehicle_fixed_cost is not None:
            fixed_raw = graph.vehicle_fixed_cost.float().to(device).view(-1)
        else:
            fixed_raw = torch.zeros_like(cap_v)

        if hasattr(graph, "vehicle_unit_distance_cost") and graph.vehicle_unit_distance_cost is not None:
            unit_raw = graph.vehicle_unit_distance_cost.float().to(device).view(-1)
        else:
            unit_raw = torch.ones_like(cap_v)

        if hasattr(graph, "vehicle_tier") and graph.vehicle_tier is not None:
            tier_raw = graph.vehicle_tier.float().to(device).view(-1)
        else:
            tier_raw = torch.zeros_like(cap_v)

        fixed_rel = self._normalize_per_batch_max(fixed_raw, veh_batch, B)
        unit_rel = self._normalize_per_batch_max(unit_raw, veh_batch, B)
        tier_rel = self._normalize_per_batch_max(tier_raw.abs(), veh_batch, B)

        return {
            "cap_v": cap_v,
            "fixed_raw": fixed_raw,
            "unit_raw": unit_raw,
            "tier_raw": tier_raw,
            "fixed_rel": fixed_rel,
            "unit_rel": unit_rel,
            "tier_rel": tier_rel,
            "fixed_log": self._safe_log1p(fixed_raw),
            "unit_log": self._safe_log1p(unit_raw),
        }

    def _build_graph_feat_light(self, graph, node_batch, veh_batch, B: int, fleet_ctx: Dict[str, torch.Tensor], device):
        demand = graph.demand_linehaul.float().to(device)
        total_dem = self._scatter_add_1d(node_batch, demand, B)
        total_cap = self._scatter_add_1d(veh_batch, fleet_ctx["cap_v"], B).clamp_min(1e-6)
        node_cnt = torch.bincount(node_batch, minlength=B).float().to(device).clamp_min(1.0)

        depot_xy = graph.depot_xy[:, 0, :].float().to(device)
        ku = graph.K_used.view(-1).float().to(device) if hasattr(graph, "K_used") else torch.ones((B,), device=device)
        if ku.numel() == 1 and B > 1:
            ku = ku.repeat(B)

        avg_fixed = self._batch_mean(fleet_ctx["fixed_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        avg_unit = self._batch_mean(fleet_ctx["unit_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        avg_tier = self._batch_mean(fleet_ctx["tier_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        mean_fixed_log = self._batch_mean(fleet_ctx["fixed_log"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        mean_unit_log = self._batch_mean(fleet_ctx["unit_log"].unsqueeze(-1), veh_batch, B).squeeze(-1)

        feat = torch.stack(
            [
                depot_xy[:, 0],
                depot_xy[:, 1],
                total_dem / total_cap,
                total_dem / node_cnt,
                ku / node_cnt,
                avg_fixed,
                avg_unit,
                avg_tier,
                mean_fixed_log,
                mean_unit_log,
            ],
            dim=-1,
        )
        if feat.size(-1) < self.graph_in_dim:
            feat = F.pad(feat, (0, self.graph_in_dim - feat.size(-1)))
        elif feat.size(-1) > self.graph_in_dim:
            feat = feat[:, :self.graph_in_dim]
        return feat

    def _build_vehicle_state(
        self,
        src,
        dst,
        node_batch,
        veh_batch,
        node_xy,
        node_unit,
        demand,
        fleet_ctx,
        p,
    ) -> Dict[str, torch.Tensor]:
        Ktot = int(veh_batch.numel())
        cap_v = fleet_ctx["cap_v"]

        load = self._scatter_add_1d(src, p * demand[dst], Ktot)
        count = self._scatter_add_1d(src, p, Ktot)

        centroid_num = self._scatter_add_2d(src, p.unsqueeze(-1) * node_xy[dst], Ktot)
        centroid = centroid_num / count.unsqueeze(-1).clamp_min(1e-6)

        dir_num = self._scatter_add_2d(src, p.unsqueeze(-1) * node_unit[dst], Ktot)
        mean_dir = dir_num / count.unsqueeze(-1).clamp_min(1e-6)
        mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        dist_to_centroid = (node_xy[dst] - centroid[src]).norm(dim=-1)
        radius_num = self._scatter_add_1d(src, p * dist_to_centroid, Ktot)
        radius = radius_num / count.clamp_min(1e-6)

        load_ratio = load / cap_v
        remain_ratio = torch.clamp(1.0 - load_ratio, min=0.0)

        return {
            "load": load,
            "count": count,
            "centroid": centroid,
            "mean_dir": mean_dir,
            "radius": radius,
            "load_ratio": load_ratio,
            "remain_ratio": remain_ratio,
            "cap_v": cap_v,
            "fixed_rel": fleet_ctx["fixed_rel"],
            "unit_rel": fleet_ctx["unit_rel"],
            "tier_rel": fleet_ctx["tier_rel"],
        }

    def _build_edge_dyn(self, src, dst, node_xy, node_unit, demand, state) -> torch.Tensor:
        dist_centroid = (node_xy[dst] - state["centroid"][src]).norm(dim=-1)
        cosang = (node_unit[dst] * state["mean_dir"][src]).sum(dim=-1).clamp(-1.0, 1.0)
        angle_diff = 1.0 - cosang
        load_after = (state["load"][src] + demand[dst]) / state["cap_v"][src].clamp_min(1e-6)
        overload = torch.relu(load_after - 1.0)
        radius_inc = torch.relu(dist_centroid - state["radius"][src])

        return torch.stack(
            [
                dist_centroid,
                angle_diff,
                load_after,
                overload,
                radius_inc,
                state["fixed_rel"][src],
                state["unit_rel"][src],
                state["tier_rel"][src],
            ],
            dim=-1,
        )

    def _refresh_dynamic_context(
        self,
        refresh_idx: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_batch: torch.Tensor,
        veh_batch: torch.Tensor,
        node_xy: torch.Tensor,
        node_unit: torch.Tensor,
        demand: torch.Tensor,
        fleet_ctx: Dict[str, torch.Tensor],
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_e_in: torch.Tensor,
    ):
        score_in = torch.cat([h_v_in[src], h_n_in[dst], h_e_in], dim=-1)
        score_l = self.pre_score[refresh_idx](score_in).squeeze(-1)
        p = pyg_softmax(score_l, index=dst)

        veh_state = self._build_vehicle_state(
            src=src,
            dst=dst,
            node_batch=node_batch,
            veh_batch=veh_batch,
            node_xy=node_xy,
            node_unit=node_unit,
            demand=demand,
            fleet_ctx=fleet_ctx,
            p=p,
        )
        edge_dyn_raw = self._build_edge_dyn(src, dst, node_xy, node_unit, demand, veh_state)
        edge_dyn_h = self.edge_dyn_proj(edge_dyn_raw)
        edge_bias = self.edge_bias_mlp(edge_dyn_raw)
        return veh_state, edge_dyn_raw, edge_dyn_h, edge_bias

    def _update_edge_state(
        self,
        h_e: torch.Tensor,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_g: torch.Tensor,
        edge_graph: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        cached_edge_dyn_h: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Default V4-HF-Lite does not perform explicit per-layer edge updates."""
        return h_e

    def forward(self, graph, xt_edge: torch.Tensor, t: torch.Tensor):
        device = graph.node_features.device

        # Project convention:
        #   xt_edge is already in [0, 1].
        # Do not infer range with xt_edge.min().item().
        xt01 = xt_edge.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        t = t.long().to(device).view(-1)

        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]

        # Prefer t length to avoid node_batch.max().item() sync.
        B = int(t.numel()) if t.numel() > 0 else 1

        t_emb = self.time_proj(self.time_emb(t))
        h_n = self.node_proj(graph.node_features.float())
        h_v = self.veh_proj(graph.veh_features.float())
        h_e = self.edge_proj(torch.cat([graph.edge_attr.float(), xt01.view(-1, 1)], dim=-1))

        fleet_ctx = self._build_fleet_static_context_light(graph, veh_batch, B, device)
        graph_feat = self._build_graph_feat_light(graph, node_batch, veh_batch, B, fleet_ctx, device)
        h_g = self.global_proj(graph_feat)

        node_xy = graph.node_features[:, 0:2].float().to(device)
        node_r = graph.node_features[:, 2].float().to(device).clamp_min(1e-6)
        node_unit = node_xy / node_r.unsqueeze(-1)
        demand = graph.demand_linehaul.float().to(device)

        if self.use_n2n:
            if hasattr(graph, "node_knn_edge_index") and graph.node_knn_edge_index is not None:
                nn_edge_index = graph.node_knn_edge_index.long().to(device)
                nn_edge_attr = graph.node_knn_edge_attr.float().to(device)
            else:
                nn_edge_index = self._build_knn_edges_by_batch(
                    node_xy=node_xy,
                    batch_ids=node_batch,
                    k=self.n2n_knn_k,
                    B=B,
                )
                nn_edge_attr = self._n2n_edge_attr(node_xy, demand, nn_edge_index)

        else:
            nn_edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
            nn_edge_attr = torch.empty((0, 7), device=device, dtype=node_xy.dtype)

        rev_edge_index = torch.stack([dst, src], dim=0)

        # Static prior from the row-categorical diffusion state.
        # xt01 is an edge-form row state/probability, not a logit.
        p0 = self._row_normalize_by_dst(
            xt01.view(-1),
            dst,
            num_nodes=graph.node_features.size(0),
        )
        veh_state_static = self._build_vehicle_state(
            src=src,
            dst=dst,
            node_batch=node_batch,
            veh_batch=veh_batch,
            node_xy=node_xy,
            node_unit=node_unit,
            demand=demand,
            fleet_ctx=fleet_ctx,
            p=p0,
        )
        edge_dyn_static_raw = self._build_edge_dyn(src, dst, node_xy, node_unit, demand, veh_state_static)
        edge_dyn_static_h = self.edge_dyn_proj(edge_dyn_static_raw)
        edge_bias_static = self.edge_bias_mlp(edge_dyn_static_raw)

        cached_edge_dyn_raw = edge_dyn_static_raw
        cached_edge_dyn_h = edge_dyn_static_h
        cached_edge_bias = edge_bias_static

        for l in range(self.n_layers):
            cond = t_emb + h_g

            if self.use_adaln:
                ns, nb, vs, vb, es, eb, gn, gv = self.adaLN[l](cond).chunk(8, dim=-1)
                h_n_in = h_n * (1.0 + ns[node_batch]) + nb[node_batch]
                h_v_in = h_v * (1.0 + vs[veh_batch]) + vb[veh_batch]
                h_e_in = h_e * (1.0 + es[edge_graph]) + eb[edge_graph]
                gate_n = torch.sigmoid(gn)
                gate_v = torch.sigmoid(gv)
            else:
                h_n_in, h_v_in, h_e_in = h_n, h_v, h_e
                gate_n = torch.ones_like(h_g)
                gate_v = torch.ones_like(h_g)

            intra_idx = self.intra_layer_to_idx.get(l, None)
            if intra_idx is not None:
                n2n_h, _ = self.n2n[intra_idx](h_n_in, nn_edge_index, nn_edge_attr)
                h_n = self.norm_n_intra[l](h_n + F.dropout(n2n_h - h_n_in, p=self.dropout, training=self.training))
                h_n_in = h_n

            refresh_idx = self.refresh_layer_to_idx.get(l, None)
            if refresh_idx is not None:
                _, cached_edge_dyn_raw, cached_edge_dyn_h, cached_edge_bias = self._refresh_dynamic_context(
                    refresh_idx,
                    src,
                    dst,
                    node_batch,
                    veh_batch,
                    node_xy,
                    node_unit,
                    demand,
                    fleet_ctx,
                    h_v_in,
                    h_n_in,
                    h_e_in,
                )

            h_e = self._update_edge_state(
                h_e=h_e,
                h_v_in=h_v_in,
                h_n_in=h_n_in,
                h_g=h_g,
                edge_graph=edge_graph,
                src=src,
                dst=dst,
                cached_edge_dyn_h=cached_edge_dyn_h,
                layer_idx=l,
            )

            edge_msg = self.edge_msg_norm[l](h_e + cached_edge_dyn_h)
            edge_bias_delta = self.edge_delta_bias[l](
                torch.cat(
                    [h_v_in[src], h_n_in[dst], h_g[edge_graph], cached_edge_dyn_raw, xt01.view(-1, 1)],
                    dim=-1,
                )
            )
            total_edge_bias = edge_bias_static + cached_edge_bias + edge_bias_delta

            v2n_out = self.v2n[l](h_v_in, edge_index, edge_msg, h_n_in, edge_logit_bias=total_edge_bias)
            n2v_out = self.n2v[l](h_n_in, rev_edge_index, edge_msg, h_v_in, edge_logit_bias=total_edge_bias)

            h_n = self.norm_n_cross[l](
                h_n + gate_n[node_batch] * F.dropout(v2n_out, p=self.dropout, training=self.training)
            )
            h_v = self.norm_v_cross[l](
                h_v + gate_v[veh_batch] * F.dropout(n2v_out, p=self.dropout, training=self.training)
            )

            # Type-aware v2v slot attention.
            # It is placed after n2v because h_v has already absorbed customer-side
            # assignment information at this layer.
            v2v_idx = self.v2v_layer_to_idx.get(l, None)
            if v2v_idx is not None:
                h_v = self.v2v[v2v_idx](
                    h_v=h_v,
                    fleet_ctx=fleet_ctx,
                    veh_batch=veh_batch,
                    B=B,
                )

            veh_mean = self._batch_mean(h_v, veh_batch, B)
            vg_out = self.vehicle_global[l](torch.cat([h_v, h_g[veh_batch], veh_mean[veh_batch]], dim=-1))
            vg_delta, vg_gate = vg_out.chunk(2, dim=-1)
            h_v = self.norm_v_global[l](h_v + torch.sigmoid(vg_gate) * F.dropout(vg_delta, p=self.dropout, training=self.training))

            if self.use_global:
                h_g = h_g + self.global_update[l](
                    torch.cat([h_g, self._batch_mean(h_n, node_batch, B), self._batch_mean(h_v, veh_batch, B)], dim=-1)
                )

        h_e_final = self.edge_out_norm(h_e + cached_edge_dyn_h)
        edge_feat = torch.cat(
            [h_v[src], h_n[dst], h_e_final, h_v[src] * h_n[dst], h_v[src] * h_e_final, h_n[dst] * h_e_final, h_g[edge_graph]],
            dim=-1,
        )
        return self.edge_head(edge_feat).squeeze(-1)


class EdgeBipartiteDenoiserV4_HF_Lite_EdgeUpd(EdgeBipartiteDenoiserV4_HF_Lite):
    """HF-lite V4 with optional explicit edge-state update.

    This is the minimal enhanced variant that adds back a single A-style idea:
    explicit per-layer edge-state refinement. Everything else remains V4-like.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 5, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.edge_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)])
        print("[LOAD] EdgeBipartiteDenoiserV4_HF_Lite_EdgeUpd = HF-lite V4 + explicit edge update")

    def _update_edge_state(
        self,
        h_e: torch.Tensor,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_g: torch.Tensor,
        edge_graph: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        cached_edge_dyn_h: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        edge_upd_in = torch.cat(
            [
                h_v_in[src],
                h_n_in[dst],
                h_e,
                cached_edge_dyn_h,
                h_g[edge_graph],
            ],
            dim=-1,
        )
        delta_e = self.edge_update[layer_idx](edge_upd_in)
        return self.edge_norm[layer_idx](h_e + F.dropout(delta_e, p=self.dropout, training=self.training))


__all__ = [
    "EdgeBipartiteDenoiserV4_HF_Lite",
    "EdgeBipartiteDenoiserV4_HF_Lite_EdgeUpd",
]
