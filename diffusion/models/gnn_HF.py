from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from .assignment_backbone import AssignmentBackbone, AssignmentContext
from .graph_ops import (
    batch_mean,
    build_knn_edges_by_batch,
    n2n_edge_attr,
    normalize_per_batch_max,
    row_normalize_by_dst,
    safe_log1p,
    scatter_add_1d,
    scatter_add_2d,
)
from .slot_attention import TypeAwareSlotSelfAttentionBlock


class EdgeBipartiteDenoiser_HF(AssignmentBackbone):
    """HFVRP assignment denoiser with shared V4 backbone.

    The shared backbone handles the recurrent bipartite/n2n/v2v/global loop.
    This wrapper keeps only heterogeneous-fleet feature construction and the
    HF-specific dynamic edge context.
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
        n2n_attn_heads: int = 4,
        n2n_attn_dropout: float = 0.05,
        n2n_attn_ffn_mult: int = 2,
        v2v_every: int = 2,
        v2v_heads: int = 4,
        v2v_dropout: float = 0.05,
        v2v_ffn_mult: int = 2,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected model kwargs: {sorted(kwargs)}")

        super().__init__(
            node_in_dim=node_in_dim,
            veh_in_dim=veh_in_dim,
            edge_in_dim=edge_in_dim,
            graph_in_dim=graph_in_dim,
            edge_dyn_dim=8,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_dim=time_dim,
            dropout=dropout,
            biattn_heads=biattn_heads,
            biattn_dropout=biattn_dropout,
            biattn_head_dim=biattn_head_dim,
            use_n2n=use_n2n,
            use_v2v=use_v2v,
            use_global=use_global,
            use_adaln=use_adaln,
            n2n_knn_k=n2n_knn_k,
            dyn_refresh_every=dyn_refresh_every,
            intra_every=intra_every,
            n2n_attn_heads=n2n_attn_heads,
            n2n_attn_dropout=n2n_attn_dropout,
            n2n_attn_ffn_mult=n2n_attn_ffn_mult,
            v2v_every=v2v_every,
            v2v_heads=v2v_heads,
            v2v_dropout=v2v_dropout,
            v2v_ffn_mult=v2v_ffn_mult,
        )

    def _build_v2v_layers(self, hidden_dim: int) -> nn.ModuleList | None:
        if not self.use_v2v:
            return None
        return nn.ModuleList(
            [
                TypeAwareSlotSelfAttentionBlock(
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
        return self.v2v[v2v_idx](
            h_v=h_v,
            fleet_ctx=ctx.problem_ctx["fleet_ctx"],
            veh_batch=ctx.veh_batch,
            B=ctx.B,
        )

    def _build_context(self, graph, xt_edge: torch.Tensor, t: torch.Tensor) -> AssignmentContext:
        device = graph.node_features.device

        xt01 = xt_edge.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        t = t.long().to(device).view(-1)

        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]
        rev_edge_index = torch.stack([dst, src], dim=0)
        B = int(t.numel()) if t.numel() > 0 else 1

        fleet_ctx = self._build_fleet_static_context_light(graph, veh_batch, B, device)
        graph_feat = self._build_graph_feat_light(graph, node_batch, veh_batch, B, fleet_ctx, device)

        node_xy = graph.node_features[:, 0:2].float().to(device)
        node_r = graph.node_features[:, 2].float().to(device).clamp_min(1e-6)
        node_unit = node_xy / node_r.unsqueeze(-1)
        demand = graph.demand_linehaul.float().to(device)

        if self.use_n2n:
            if hasattr(graph, "node_knn_edge_index") and graph.node_knn_edge_index is not None:
                nn_edge_index = graph.node_knn_edge_index.long().to(device)
                nn_edge_attr = graph.node_knn_edge_attr.float().to(device)
            else:
                nn_edge_index = build_knn_edges_by_batch(
                    node_xy=node_xy,
                    batch_ids=node_batch,
                    k=self.n2n_knn_k,
                    num_graphs=B,
                )
                nn_edge_attr = n2n_edge_attr(node_xy, demand, nn_edge_index)
        else:
            nn_edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
            nn_edge_attr = torch.empty((0, 7), device=device, dtype=node_xy.dtype)

        return AssignmentContext(
            graph=graph,
            xt01=xt01,
            t=t,
            B=B,
            node_batch=node_batch,
            veh_batch=veh_batch,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            src=src,
            dst=dst,
            edge_graph=edge_graph,
            node_xy=node_xy,
            node_unit=node_unit,
            demand=demand,
            nn_edge_index=nn_edge_index,
            nn_edge_attr=nn_edge_attr,
            h_g_init_feat=graph_feat,
            problem_ctx={"fleet_ctx": fleet_ctx},
        )

    def _build_fleet_static_context_light(
        self,
        graph,
        veh_batch: torch.Tensor,
        B: int,
        device,
    ) -> dict[str, torch.Tensor]:
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

        fixed_rel = normalize_per_batch_max(fixed_raw, veh_batch, B)
        unit_rel = normalize_per_batch_max(unit_raw, veh_batch, B)
        tier_rel = normalize_per_batch_max(tier_raw.abs(), veh_batch, B)

        return {
            "cap_v": cap_v,
            "fixed_raw": fixed_raw,
            "unit_raw": unit_raw,
            "tier_raw": tier_raw,
            "fixed_rel": fixed_rel,
            "unit_rel": unit_rel,
            "tier_rel": tier_rel,
            "fixed_log": safe_log1p(fixed_raw),
            "unit_log": safe_log1p(unit_raw),
        }

    def _build_graph_feat_light(
        self,
        graph,
        node_batch: torch.Tensor,
        veh_batch: torch.Tensor,
        B: int,
        fleet_ctx: dict[str, torch.Tensor],
        device,
    ) -> torch.Tensor:
        demand = graph.demand_linehaul.float().to(device)
        total_dem = scatter_add_1d(node_batch, demand, B)
        total_cap = scatter_add_1d(veh_batch, fleet_ctx["cap_v"], B).clamp_min(1e-6)
        node_cnt = torch.bincount(node_batch, minlength=B).float().to(device).clamp_min(1.0)

        depot_xy = graph.depot_xy[:, 0, :].float().to(device)
        K_used = graph.K_used.view(-1).float().to(device) if hasattr(graph, "K_used") else torch.ones((B,), device=device)
        if K_used.numel() == 1 and B > 1:
            K_used = K_used.repeat(B)

        avg_fixed = batch_mean(fleet_ctx["fixed_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        avg_unit = batch_mean(fleet_ctx["unit_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        avg_tier = batch_mean(fleet_ctx["tier_rel"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        mean_fixed_log = batch_mean(fleet_ctx["fixed_log"].unsqueeze(-1), veh_batch, B).squeeze(-1)
        mean_unit_log = batch_mean(fleet_ctx["unit_log"].unsqueeze(-1), veh_batch, B).squeeze(-1)

        feat = torch.stack(
            [
                depot_xy[:, 0],
                depot_xy[:, 1],
                total_dem / total_cap,
                total_dem / node_cnt,
                K_used / node_cnt,
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
            feat = feat[:, : self.graph_in_dim]
        return feat

    def _build_vehicle_state(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_batch: torch.Tensor,
        veh_batch: torch.Tensor,
        node_xy: torch.Tensor,
        node_unit: torch.Tensor,
        demand: torch.Tensor,
        fleet_ctx: dict[str, torch.Tensor],
        p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_slots = int(veh_batch.numel())
        cap_v = fleet_ctx["cap_v"]

        load = scatter_add_1d(src, p * demand[dst], num_slots)
        count = scatter_add_1d(src, p, num_slots)

        centroid_num = scatter_add_2d(src, p.unsqueeze(-1) * node_xy[dst], num_slots)
        centroid = centroid_num / count.unsqueeze(-1).clamp_min(1e-6)

        dir_num = scatter_add_2d(src, p.unsqueeze(-1) * node_unit[dst], num_slots)
        mean_dir = dir_num / count.unsqueeze(-1).clamp_min(1e-6)
        mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        dist_to_centroid = (node_xy[dst] - centroid[src]).norm(dim=-1)
        radius_num = scatter_add_1d(src, p * dist_to_centroid, num_slots)
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

    def _build_edge_dyn(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_xy: torch.Tensor,
        node_unit: torch.Tensor,
        demand: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
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

    def _build_initial_dynamic_context(
        self,
        ctx: AssignmentContext,
        h_v: torch.Tensor,
        h_n: torch.Tensor,
        h_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = row_normalize_by_dst(
            ctx.xt01.view(-1),
            ctx.dst,
            num_nodes=ctx.graph.node_features.size(0),
        )
        fleet_ctx = ctx.problem_ctx["fleet_ctx"]
        state = self._build_vehicle_state(
            src=ctx.src,
            dst=ctx.dst,
            node_batch=ctx.node_batch,
            veh_batch=ctx.veh_batch,
            node_xy=ctx.node_xy,
            node_unit=ctx.node_unit,
            demand=ctx.demand,
            fleet_ctx=fleet_ctx,
            p=p0,
        )
        edge_dyn_raw = self._build_edge_dyn(ctx.src, ctx.dst, ctx.node_xy, ctx.node_unit, ctx.demand, state)
        return edge_dyn_raw, self.edge_dyn_proj(edge_dyn_raw), self.edge_bias_mlp(edge_dyn_raw)

    def _refresh_dynamic_context(
        self,
        refresh_idx: int,
        ctx: AssignmentContext,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_e_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_in = torch.cat([h_v_in[ctx.src], h_n_in[ctx.dst], h_e_in], dim=-1)
        score_l = self.pre_score[refresh_idx](score_in).squeeze(-1)
        p = pyg_softmax(score_l, index=ctx.dst)

        fleet_ctx = ctx.problem_ctx["fleet_ctx"]
        state = self._build_vehicle_state(
            src=ctx.src,
            dst=ctx.dst,
            node_batch=ctx.node_batch,
            veh_batch=ctx.veh_batch,
            node_xy=ctx.node_xy,
            node_unit=ctx.node_unit,
            demand=ctx.demand,
            fleet_ctx=fleet_ctx,
            p=p,
        )
        edge_dyn_raw = self._build_edge_dyn(ctx.src, ctx.dst, ctx.node_xy, ctx.node_unit, ctx.demand, state)
        return edge_dyn_raw, self.edge_dyn_proj(edge_dyn_raw), self.edge_bias_mlp(edge_dyn_raw)


class EdgeBipartiteDenoiser_HF_EdgeUpd(EdgeBipartiteDenoiser_HF):
    """HF-lite V4 with explicit per-layer edge-state refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_edge_state_update = True
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
        edge_upd_in = torch.cat(
            [
                h_v_in[ctx.src],
                h_n_in[ctx.dst],
                h_e,
                cached_edge_dyn_h,
                h_g[ctx.edge_graph],
            ],
            dim=-1,
        )
        delta_e = self.edge_update[layer_idx](edge_upd_in)
        return self.edge_norm[layer_idx](
            h_e + F.dropout(delta_e, p=self.dropout, training=self.training)
        )


__all__ = [
    "EdgeBipartiteDenoiser_HF",
    "EdgeBipartiteDenoiser_HF_EdgeUpd",
]
