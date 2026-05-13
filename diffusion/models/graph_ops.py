"""Reusable graph/tensor operations for assignment GNN backbones.

This module intentionally contains no ``nn.Module`` definitions and no trainable
parameters.  Keep these functions side-effect free so they can be unit-tested
independently from CVRP/HFVRP model classes.
"""

from __future__ import annotations

import torch


def batch_mean(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Compute per-graph mean features by scatter-add.

    Args:
        x: Tensor with shape ``[num_items, feat_dim]``.
        batch: Graph id for every item, shape ``[num_items]``.
        num_graphs: Number of graphs in the mini-batch.

    Returns:
        Tensor with shape ``[num_graphs, feat_dim]``.
    """
    out = x.new_zeros((int(num_graphs), x.size(-1)))
    cnt = x.new_zeros((int(num_graphs), 1))
    batch = batch.long()
    out.index_add_(0, batch, x)
    cnt.index_add_(
        0,
        batch,
        torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype),
    )
    return out / cnt.clamp_min(1.0)


def scatter_add_1d(index: torch.Tensor, value: torch.Tensor, out_size: int) -> torch.Tensor:
    """Scatter-add 1D values into an output vector."""
    out = value.new_zeros((int(out_size),))
    out.index_add_(0, index.long(), value)
    return out


def scatter_add_2d(index: torch.Tensor, value: torch.Tensor, out_size: int) -> torch.Tensor:
    """Scatter-add 2D values into an output matrix."""
    out = value.new_zeros((int(out_size), value.size(-1)))
    out.index_add_(0, index.long(), value)
    return out


def row_normalize_by_dst(
    x: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize edge values per destination/customer row.

    This is used to convert an edge-form assignment state into row-wise
    probabilities.  Invalid values are mapped to zero, negative values are
    clipped, and all-zero rows fall back to a uniform distribution over the
    existing incoming edges.

    Args:
        x: Edge values, shape ``[num_edges]``.
        dst: Destination/customer index for every edge, shape ``[num_edges]``.
        num_nodes: Number of destination/customer nodes.
        eps: Numerical-stability constant.

    Returns:
        Row-normalized edge values with shape ``[num_edges]``.
    """
    dst = dst.long()
    x = torch.nan_to_num(
        x.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)

    denom = x.new_zeros((int(num_nodes),))
    denom.index_add_(0, dst, x)

    p = x / denom[dst].clamp_min(eps)

    # Fallback for all-zero rows.  Compute unconditionally to avoid GPU sync
    # from Python-side ``if tensor.any()`` checks.
    ones = torch.ones_like(x)
    cnt = x.new_zeros((int(num_nodes),))
    cnt.index_add_(0, dst, ones)
    uniform = ones / cnt[dst].clamp_min(1.0)

    bad = denom[dst] <= eps
    return torch.where(bad, uniform, p)

def build_knn_edges_by_batch(
    node_xy: torch.Tensor,
    batch_ids: torch.Tensor,
    k: int,
    num_graphs: int | None = None,
) -> torch.Tensor:
    """Build directed KNN edges for sparse n2n attention.

    Assumption:
        Nodes are stored in PyG batch order and every graph in the batch has the
        same number of customer nodes.

    Edge convention:
        edge_index[0] = receiver/current node, i.e., the node being updated.
        edge_index[1] = source/neighbor node, i.e., the KNN neighbor providing
        key/value/message to the receiver.

    Message direction:
        source -> receiver

    Attention grouping:
        softmax is computed over all source neighbors of the same receiver.

    Args:
        node_xy:
            Node coordinates, shape [total_nodes, coord_dim].
        batch_ids:
            Graph id for every node, shape [total_nodes].
            This is only used to infer num_graphs when num_graphs is not given.
        k:
            Number of nearest neighbours per receiver node.
        num_graphs:
            Number of graphs. Pass this from t.numel() in forward to avoid
            recomputing it with a GPU-to-CPU scalar sync.

    Returns:
        edge_index with shape [2, total_nodes * min(k, nodes_per_graph - 1)].
        The first row contains receiver indices; the second row contains
        source/neighbor indices.
    """
    device = node_xy.device
    k = int(k)

    if k <= 0 or node_xy.numel() == 0:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    total_nodes = int(node_xy.size(0))

    if num_graphs is None:
        if batch_ids.numel() == 0:
            num_graphs = 0
        else:
            # Prefer passing num_graphs explicitly in performance-critical
            # forward paths to avoid this scalar read.
            num_graphs = int(batch_ids[-1].detach().item()) + 1

    B = int(num_graphs)
    if B <= 0:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    if total_nodes % B != 0:
        raise ValueError(
            "Fixed-batch KNN requires total_nodes % num_graphs == 0, "
            f"got total_nodes={total_nodes}, num_graphs={B}."
        )

    nodes_per_graph = total_nodes // B
    if nodes_per_graph <= 1:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    kk = min(k, nodes_per_graph - 1)

    # [B, N, coord_dim]
    xy = node_xy.reshape(B, nodes_per_graph, node_xy.size(-1))

    # [B, N, N], pairwise distances inside each graph.
    dist = torch.cdist(xy, xy, p=2)

    # Exclude self-neighbors.
    diag = torch.arange(nodes_per_graph, device=device)
    dist[:, diag, diag] = float("inf")

    # nearest_idx[b, i, :] gives the local source/neighbor indices attended by
    # receiver node i in graph b.
    nearest_idx = torch.topk(dist, kk, largest=False, dim=-1).indices

    base = (
        torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
        * nodes_per_graph
    )

    receiver_local = torch.arange(
        nodes_per_graph,
        device=device,
        dtype=torch.long,
    ).view(1, nodes_per_graph, 1)
    receiver_local = receiver_local.expand(B, nodes_per_graph, kk)

    source_local = nearest_idx

    receiver = (base + receiver_local).reshape(-1)
    source = (base + source_local).reshape(-1)

    return torch.stack([receiver, source], dim=0)


def n2n_edge_attr(
    node_xy: torch.Tensor,
    demand: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build 7-dimensional edge features for receiver-source KNN attention.

    Edge convention:
        edge_index[0] = receiver/current node
        edge_index[1] = source/neighbor node

    Feature layout:
        dx, dy, distance, inverse_distance,
        demand_receiver, demand_source,
        |demand_receiver - demand_source|
    """
    if edge_index.numel() == 0:
        return torch.empty((0, 7), device=node_xy.device, dtype=node_xy.dtype)

    receiver = edge_index[0].long()
    source = edge_index[1].long()

    dxy = node_xy[receiver] - node_xy[source]
    dist = torch.sqrt((dxy**2).sum(dim=-1) + 1e-12)
    inv = 1.0 / (dist + 1e-6)

    demand_receiver = demand[receiver].to(node_xy.dtype)
    demand_source = demand[source].to(node_xy.dtype)

    return torch.cat(
        [
            dxy,
            dist.unsqueeze(-1),
            inv.unsqueeze(-1),
            demand_receiver.unsqueeze(-1),
            demand_source.unsqueeze(-1),
            (demand_receiver - demand_source).abs().unsqueeze(-1),
        ],
        dim=-1,
    )


def normalize_per_batch_max(
    x: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize values by the maximum value inside each graph."""
    if x.numel() == 0:
        return x

    batch_idx = batch_idx.long()
    max_per = torch.full((int(num_graphs),), -float("inf"), device=x.device, dtype=x.dtype)
    max_per.scatter_reduce_(0, batch_idx, x, reduce="amax", include_self=True)
    max_per = max_per.clamp_min(eps)
    return x / max_per[batch_idx]


def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe ``log(1 + x)`` for non-negative feature construction."""
    return torch.log1p(torch.clamp(x, min=0.0))


__all__ = [
    "batch_mean",
    "scatter_add_1d",
    "scatter_add_2d",
    "row_normalize_by_dst",
    "build_knn_edges_by_batch",
    "n2n_edge_attr",
    "normalize_per_batch_max",
    "safe_log1p",
]
