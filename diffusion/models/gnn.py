from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax

from .assignment_backbone import AssignmentBackbone, AssignmentContext
from .graph_ops import (
    build_knn_edges_by_batch,
    n2n_edge_attr,
    row_normalize_by_dst,
    scatter_add_1d,
    scatter_add_2d,
)


class CVRPVehNodeData(Data):
    def __init__(
        self,
        veh_features=None,
        node_features=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        node_batch=None,
        veh_batch=None,
        capacity=None,
        depot_xy=None,
        actions=None,
        gt_cost=None,
        graph_feat=None,
        node_knn_edge_index=None,
        node_knn_edge_attr=None,
    ):
        super().__init__()

        if veh_features is not None:
            self.veh_features = veh_features
        if node_features is not None:
            self.node_features = node_features
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y

        if node_batch is not None:
            self.node_batch = node_batch
        if veh_batch is not None:
            self.veh_batch = veh_batch

        if capacity is not None:
            self.capacity = capacity
        if depot_xy is not None:
            self.depot_xy = depot_xy

        if actions is not None:
            self.actions = actions
        if gt_cost is not None:
            self.gt_cost = gt_cost

        if graph_feat is not None:
            self.graph_feat = graph_feat
        if node_knn_edge_index is not None:
            self.node_knn_edge_index = node_knn_edge_index
        if node_knn_edge_attr is not None:
            self.node_knn_edge_attr = node_knn_edge_attr

        self.src_count = int(veh_features.size(0)) if veh_features is not None else 0
        self.dst_count = int(node_features.size(0)) if node_features is not None else 0

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == "edge_index":
            src = self.src_count if self.src_count > 0 else int(self.veh_features.size(0))
            dst = self.dst_count if self.dst_count > 0 else int(self.node_features.size(0))
            return torch.tensor([[src], [dst]])

        if key == "node_knn_edge_index":
            n = self.dst_count if self.dst_count > 0 else int(self.node_features.size(0))
            return torch.tensor([[n], [n]])

        if key in ["node_batch", "veh_batch"]:
            return int(value.max().item()) + 1 if torch.is_tensor(value) and value.numel() > 0 else 1

        return super().__inc__(key, value, store, *args, **kwargs)


class EdgeBipartiteDenoiser(AssignmentBackbone):
    """CVRP assignment denoiser with shared V4 backbone."""

    def __init__(
        self,
        node_in_dim: int = 10,
        veh_in_dim: int = 7,
        edge_in_dim: int = 8,
        graph_in_dim: int = 6,
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
        super().__init__(
            node_in_dim=node_in_dim,
            veh_in_dim=veh_in_dim,
            edge_in_dim=edge_in_dim,
            graph_in_dim=graph_in_dim,
            edge_dyn_dim=5,
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

        capacity_b = graph.capacity.float().view(-1).to(device)
        if capacity_b.numel() == 1 and B > 1:
            capacity_b = capacity_b.repeat(B)

        if hasattr(graph, "graph_feat") and graph.graph_feat is not None:
            graph_feat = graph.graph_feat.float().to(device)
        else:
            demand = graph.demand_linehaul.float().to(device)
            total_dem = scatter_add_1d(node_batch, demand, B) / capacity_b.clamp_min(1e-6)
            node_cnt = torch.bincount(node_batch, minlength=B).float().to(device).clamp_min(1.0)
            veh_cnt = torch.bincount(veh_batch, minlength=B).float().to(device).clamp_min(1.0)

            if hasattr(graph, "K_max") and graph.K_max is not None:
                K_graph = graph.K_max.view(-1).float().to(device)
                if K_graph.numel() == 1 and B > 1:
                    K_graph = K_graph.repeat(B)
                K_graph = torch.minimum(K_graph[:B].clamp_min(1.0), veh_cnt)
            else:
                K_graph = veh_cnt

            cap_lb = torch.ceil(total_dem).clamp_min(1.0)
            cap_lb_ratio = cap_lb / K_graph.clamp_min(1.0)
            slot_density = K_graph / torch.sqrt(node_cnt.clamp_min(1.0))
            depot_xy = graph.depot_xy.float().to(device)

            graph_feat = torch.stack(
                [
                    depot_xy[:, 0, 0],
                    depot_xy[:, 0, 1],
                    total_dem,
                    total_dem / node_cnt,
                    cap_lb_ratio,
                    slot_density,
                ],
                dim=-1,
            )

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
            problem_ctx={"capacity_b": capacity_b},
        )

    def _build_vehicle_state(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_batch: torch.Tensor,
        veh_batch: torch.Tensor,
        node_xy: torch.Tensor,
        node_unit: torch.Tensor,
        demand: torch.Tensor,
        capacity_b: torch.Tensor,
        p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_slots = int(veh_batch.numel())
        cap_v = capacity_b[veh_batch].clamp_min(1e-6)

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

        node_cnt_b = torch.bincount(
            node_batch,
            minlength=int(capacity_b.numel()),
        ).float().to(node_xy.device).clamp_min(1.0)
        client_count_ratio = count / node_cnt_b[veh_batch]

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
            "client_count_ratio": client_count_ratio,
            "cap_v": cap_v,
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
        return torch.stack([dist_centroid, angle_diff, load_after, overload, radius_inc], dim=-1)

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
        state = self._build_vehicle_state(
            src=ctx.src,
            dst=ctx.dst,
            node_batch=ctx.node_batch,
            veh_batch=ctx.veh_batch,
            node_xy=ctx.node_xy,
            node_unit=ctx.node_unit,
            demand=ctx.demand,
            capacity_b=ctx.problem_ctx["capacity_b"],
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

        state = self._build_vehicle_state(
            src=ctx.src,
            dst=ctx.dst,
            node_batch=ctx.node_batch,
            veh_batch=ctx.veh_batch,
            node_xy=ctx.node_xy,
            node_unit=ctx.node_unit,
            demand=ctx.demand,
            capacity_b=ctx.problem_ctx["capacity_b"],
            p=p,
        )
        edge_dyn_raw = self._build_edge_dyn(ctx.src, ctx.dst, ctx.node_xy, ctx.node_unit, ctx.demand, state)
        return edge_dyn_raw, self.edge_dyn_proj(edge_dyn_raw), self.edge_bias_mlp(edge_dyn_raw)





__all__ = [
    "CVRPVehNodeData",
    "EdgeBipartiteDenoiser",
]
