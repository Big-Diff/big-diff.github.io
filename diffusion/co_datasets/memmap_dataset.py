# diffusion/co_datasets/memmap_dataset.py
"""
Memmap CVRP dataset -> Kmax bipartite graphs for row-categorical assignment diffusion.

Aligned semantics:
  - The GNN-visible slot space is fixed by K_max / --num_vehicles.
  - Reference routes provide only instance-local partition labels y.
  - Pairwise partition loss uses y_i == y_j, so route order is not a semantic target.
  - Row CE uses y only as a weak anchor.
  - Dataset-side KNN precomputation is disabled; the GNN builds node-node KNN on the fly.

Output dimensions match EdgeBipartiteDenoiserV4:
  node_features: (N, 10)
  veh_features : (K_max, 7)
  edge_attr    : (K_max * N, 8)
  graph_feat   : (1, 6)
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.models.gnn import CVRPVehNodeData


def decode_routes_zero_sep(actions: np.ndarray, depot: int = 0):
    """Decode 0-separated route encoding into a list of non-empty routes."""
    routes, cur = [], []
    for a in np.asarray(actions).reshape(-1).tolist():
        a = int(a)
        if a == depot:
            if cur:
                routes.append(cur)
                cur = []
        else:
            cur.append(a)
    if cur:
        routes.append(cur)
    return routes


def strip_trailing_zeros(actions_1d: np.ndarray) -> np.ndarray:
    """Strip only padding zeros at the end; keep internal zeros as route separators."""
    actions_1d = np.asarray(actions_1d, dtype=np.int64).reshape(-1)
    if actions_1d.size == 0:
        return actions_1d

    end = actions_1d.size
    while end > 0 and int(actions_1d[end - 1]) == 0:
        end -= 1

    if end <= 0:
        return actions_1d[:1]
    return actions_1d[:end]


def _infer_assignment_from_routes(routes, V: int) -> np.ndarray:
    """Return y (N,), where y[i] is an instance-local partition id for client i+1.

    The integer id is not a semantic vehicle identity. It is only used as:
      1) a weak row-CE anchor;
      2) a way to derive pairwise partition labels by y_i == y_j.
    """
    N = int(V) - 1
    y = -np.ones((N,), dtype=np.int64)

    for part_id, route in enumerate(routes):
        for cid in route:
            cid = int(cid)
            if cid <= 0 or cid >= V:
                raise ValueError(f"[memmap_dataset] illegal cid={cid}, V={V}")
            pos = cid - 1
            if y[pos] != -1:
                raise ValueError(f"[memmap_dataset] duplicate client {cid} in routes")
            y[pos] = int(part_id)

    if (y < 0).any():
        missing = int((y < 0).sum())
        raise ValueError(f"[memmap_dataset] unassigned clients: {missing}/{N}")

    return y


def _normalize_demand_array(demand_linehaul: np.ndarray, V: int) -> np.ndarray:
    """Accept demand shape (N,) or (V,), return client demand shape (N,)."""
    N = int(V) - 1
    d = np.asarray(demand_linehaul, dtype=np.float32).reshape(-1)
    if d.size == V:
        return d[1:].astype(np.float32).reshape(N)
    if d.size == N:
        return d.astype(np.float32).reshape(N)
    raise ValueError(
        f"[memmap_dataset] demand shape mismatch: got {d.shape}, expected {(N,)} or {(V,)}"
    )


def build_bipartite_edge_data(
    points,
    demand_linehaul,
    capacity,
    speed,
    actions,
    gt_cost=None,
    *,
    K_max: Optional[int] = None,
    sparse_factor: int = -1,
    seed: int = 1234,
    instance_id: int = 0,
    keep_raw: bool = False,
    dbg: bool = False,
    knn_k: int = 8,
) -> CVRPVehNodeData:
    """Build one Kmax-based bipartite CVRP graph.

    Important semantics:
      - K_ref_used is decoded from the reference tour and used only as a target/metric.
      - K_graph = K_max is the full model-visible candidate slot space.
      - data.K_used is set to K_graph for backward compatibility.
      - data.y contains instance-local partition ids in [0, K_ref_used - 1].
      - No route canonicalization is applied.
      - No dataset-side KNN graph is stored.
    """
    del seed, instance_id, knn_k  # kept in the signature for compatibility with old call sites.

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"[build_bipartite_edge_data] bad points shape: {points.shape}")

    V = int(points.shape[0])
    if V < 2:
        raise ValueError(f"[build_bipartite_edge_data] V too small: {V}")

    N = V - 1
    depot_xy = points[0].astype(np.float32)
    client_xy = points[1:].astype(np.float32)

    demand = _normalize_demand_array(demand_linehaul, V=V)
    cap = float(capacity)
    spd = float(speed)

    actions = strip_trailing_zeros(np.asarray(actions, dtype=np.int64).reshape(-1))
    if actions.size == 0 or np.all(actions == 0):
        raise ValueError("[build_bipartite_edge_data] bad actions: empty/all zeros")

    routes = decode_routes_zero_sep(actions, depot=0)
    K_ref_used = int(len(routes))
    if K_ref_used <= 0:
        raise ValueError(f"[build_bipartite_edge_data] K_ref_used=0, routes={len(routes)}")

    if K_max is None or int(K_max) <= 0:
        raise ValueError(
            "[build_bipartite_edge_data] K_max must be explicitly provided. "
            "Do not infer the model-visible slot count from GT routes."
        )

    K_graph = int(K_max)
    if K_ref_used > K_graph:
        raise ValueError(
            f"[build_bipartite_edge_data] K_ref_used={K_ref_used} exceeds K_max={K_graph}. "
            "Increase --num_vehicles / --K_max."
        )

    sparse_factor = int(sparse_factor) if sparse_factor is not None else -1
    if sparse_factor not in (-1, 0):
        raise ValueError(
            "[build_bipartite_edge_data] positive sparse_factor is disabled for Kmax graphs. "
            "Use -1 or 0 so every customer can attend to every candidate slot."
        )

    # Instance-local partition labels. Pairwise loss only uses equality y_i == y_j.
    y = _infer_assignment_from_routes(routes, V=V)

    # ============================================================
    # Node static features, 10 dims, aligned with EdgeBipartiteDenoiserV4.
    # The GNN uses node_features[:, 0:2] as relative xy and node_features[:, 2] as radius.
    # Last four dims are intentionally zero because dataset-side KNN statistics are removed.
    # ============================================================
    rel = (client_xy - depot_xy[None, :]).astype(np.float32)
    x_rel = rel[:, 0]
    y_rel = rel[:, 1]
    r = np.sqrt((rel ** 2).sum(axis=1) + 1e-12).astype(np.float32)
    theta = np.arctan2(y_rel, x_rel).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    demand_norm = (demand / max(cap, 1e-6)).astype(np.float32)

    zeros4 = np.zeros((N, 4), dtype=np.float32)
    node_feat = np.concatenate(
        [
            np.stack(
                [x_rel, y_rel, r, sin_theta, cos_theta, demand_norm],
                axis=1,
            ).astype(np.float32),
            zeros4,
        ],
        axis=1,
    ).astype(np.float32)
    assert node_feat.shape == (N, 10)

    # ============================================================
    # Vehicle static features, 7 dims.
    # ============================================================
    veh_feat = np.zeros((K_graph, 7), dtype=np.float32)
    cap_norm = np.float32(cap / max(float(demand.mean()), 1e-6))
    veh_feat[:, 0] = cap_norm
    veh_feat[:, 1] = np.float32(spd)
    veh_feat[:, 2:4] = depot_xy[None, :]

    # Scale-aware but less brittle than K_graph / N.
    # For points sampled in a fixed [0, 1]^2 region, local geometric scale changes
    # roughly with 1 / sqrt(N), so K_graph / sqrt(N) is smoother across N=100,200,500.
    slot_density = np.float32(float(K_graph) / np.sqrt(max(1.0, float(N))))
    veh_feat[:, 4] = slot_density

    ids = (np.arange(K_graph, dtype=np.float32) + 0.5) / max(1, K_graph)
    veh_feat[:, 5] = np.sin(2.0 * np.pi * ids)
    veh_feat[:, 6] = np.cos(2.0 * np.pi * ids)

    # ============================================================
    # Full bipartite edges over K_graph slots, 8-dim edge attributes.
    # The last two dims are zero placeholders to keep edge_in_dim=8.
    # ============================================================
    src = np.repeat(np.arange(K_graph, dtype=np.int64), N)
    dst = np.tile(np.arange(N, dtype=np.int64), K_graph)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    zeros2_e = np.zeros((dst.size, 2), dtype=np.float32)
    edge_attr = np.concatenate(
        [
            np.stack(
                [
                    x_rel[dst],
                    y_rel[dst],
                    r[dst],
                    sin_theta[dst],
                    cos_theta[dst],
                    demand_norm[dst],
                ],
                axis=1,
            ).astype(np.float32),
            zeros2_e,
        ],
        axis=1,
    ).astype(np.float32)
    assert edge_attr.shape == (K_graph * N, 8)

    total_dem_over_cap = np.float32(demand.sum() / max(cap, 1e-6))
    mean_dem_over_cap = np.float32(demand.mean() / max(cap, 1e-6))

    # Capacity lower bound: minimum number of routes required by capacity alone.
    cap_lb = np.float32(np.ceil(total_dem_over_cap))
    cap_lb_ratio = np.float32(cap_lb / max(1.0, float(K_graph)))

    # Scale-aware slot density. This replaces both N / 100 and K_graph / N.
    slot_density = np.float32(float(K_graph) / np.sqrt(max(1.0, float(N))))

    graph_feat = np.array(
        [
            depot_xy[0],
            depot_xy[1],
            total_dem_over_cap,
            mean_dem_over_cap,
            cap_lb_ratio,
            slot_density,
        ],
        dtype=np.float32,
    )

    data = CVRPVehNodeData(
        veh_features=torch.from_numpy(veh_feat),
        node_features=torch.from_numpy(node_feat),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        graph_feat=torch.from_numpy(graph_feat).float().unsqueeze(0),
    )

    data.capacity = torch.tensor([cap], dtype=torch.float32)
    data.vehicle_capacity = data.capacity
    data.demand_linehaul = torch.from_numpy(demand.astype(np.float32)).float()

    # K_graph / K_max: visible candidate slots. K_ref_used: GT non-empty routes only.
    data.K_used = torch.tensor([int(K_graph)], dtype=torch.long)
    data.K_max = torch.tensor([int(K_graph)], dtype=torch.long)
    data.K_ref_used = torch.tensor([int(K_ref_used)], dtype=torch.long)

    data.depot_xy = torch.from_numpy(depot_xy).float().unsqueeze(0).unsqueeze(0)

    if gt_cost is None:
        data.gt_cost = torch.tensor([float("nan")], dtype=torch.float32)
    else:
        data.gt_cost = torch.tensor([float(gt_cost)], dtype=torch.float32)

    data.src_count = int(K_graph)
    data.dst_count = int(N)
    data.node_batch = torch.zeros(N, dtype=torch.long)
    data.veh_batch = torch.zeros(K_graph, dtype=torch.long)

    if keep_raw:
        data.points = torch.from_numpy(points).float()
        data.actions = torch.from_numpy(actions.astype(np.int64))
        data.demand_full = torch.from_numpy(np.concatenate([[0.0], demand]).astype(np.float32))

    if dbg:
        print("[DBG] V,N,K_ref_used,K_graph,E:", V, N, K_ref_used, K_graph, int(edge_index.shape[1]))
        for k, v in data.__dict__.items():
            if torch.is_tensor(v):
                print("[DBG]", k, tuple(v.shape), v.dtype)

    return data


class CVRPMemmapVehNodeDataset(Dataset):
    """
    Memmap directory loader.

    Required files:
      locs.npy              (M,V,2)
      demand_linehaul.npy   (M,N) or (M,V)
      vehicle_capacity.npy  (M,1) or scalar-like per instance
      best_tour.npy         (M,L), 0-separated routes with padding zeros
      best_cost.npy         (M,)
      meta.json             with at least {"written": ...}

    Optional:
      speed.npy             (M,1), defaults to 1.0 if absent.
    """

    def __init__(
        self,
        memmap_dir: str,
        K_max: Optional[int] = None,
        *,
        sparse_factor: int = -1,
        seed: int = 1234,
        max_instances=None,
        keep_raw: bool = False,
        knn_k: int = 8,
    ):
        self.dir = str(memmap_dir)
        self.keep_raw = bool(keep_raw)
        self.knn_k = int(knn_k)  # kept for call-site compatibility; not used for precomputed KNN.

        if K_max is None or int(K_max) <= 0:
            raise ValueError(
                "[CVRPMemmapVehNodeDataset] K_max must be explicitly provided. "
                "Pass --num_vehicles / --K_max; do not infer slot count from GT/meta."
            )
        self.K_max = int(K_max)

        self.sparse_factor = int(sparse_factor) if sparse_factor is not None else -1
        if self.sparse_factor not in (-1, 0):
            raise ValueError(
                "[CVRPMemmapVehNodeDataset] positive sparse_factor is disabled for Kmax graphs. "
                "Use -1 or 0."
            )
        self.seed = int(seed)

        meta_path = os.path.join(self.dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        written = int(self.meta.get("written", 0))
        if written <= 0:
            raise ValueError(f"[CVRPMemmapVehNodeDataset] meta.json written={written} invalid: {meta_path}")

        self.locs = np.load(os.path.join(self.dir, "locs.npy"), mmap_mode="r")
        self.dem = np.load(os.path.join(self.dir, "demand_linehaul.npy"), mmap_mode="r")
        self.cap = np.load(os.path.join(self.dir, "vehicle_capacity.npy"), mmap_mode="r")
        self.tour = np.load(os.path.join(self.dir, "best_tour.npy"), mmap_mode="r")
        self.cost = np.load(os.path.join(self.dir, "best_cost.npy"), mmap_mode="r")

        speed_path = os.path.join(self.dir, "speed.npy")
        self.spd = np.load(speed_path, mmap_mode="r") if os.path.exists(speed_path) else None

        self.M = int(written)
        if max_instances is not None:
            self.M = min(self.M, int(max_instances))

        if self.locs.shape[0] < self.M or self.dem.shape[0] < self.M or self.tour.shape[0] < self.M:
            raise ValueError(
                "[CVRPMemmapVehNodeDataset] memmap arrays have fewer rows than meta['written']: "
                f"written={self.M}, locs={self.locs.shape[0]}, dem={self.dem.shape[0]}, tour={self.tour.shape[0]}"
            )

    def __len__(self):
        return self.M

    def __getitem__(self, idx: int):
        idx = int(idx) % self.M

        points = np.asarray(self.locs[idx], dtype=np.float32)
        demand = np.asarray(self.dem[idx], dtype=np.float32)
        cap = float(np.asarray(self.cap[idx]).reshape(-1)[0])
        spd = 1.0 if self.spd is None else float(np.asarray(self.spd[idx]).reshape(-1)[0])

        actions = np.asarray(self.tour[idx]).astype(np.int64).reshape(-1)
        actions = strip_trailing_zeros(actions)

        raw_cost = float(np.asarray(self.cost[idx]).reshape(-1)[0])
        gt_cost = raw_cost if np.isfinite(raw_cost) else None

        data = build_bipartite_edge_data(
            points=points,
            demand_linehaul=demand,
            capacity=cap,
            speed=spd,
            actions=actions,
            gt_cost=gt_cost,
            K_max=self.K_max,
            sparse_factor=self.sparse_factor,
            seed=self.seed,
            instance_id=idx,
            keep_raw=self.keep_raw,
            knn_k=self.knn_k,
        )

        k_ref = int(data.K_ref_used.item())
        if k_ref > self.K_max:
            raise ValueError(
                f"[CVRPMemmapVehNodeDataset] instance idx={idx} has K_ref_used={k_ref} > K_max={self.K_max}. "
                "Increase --num_vehicles / --K_max."
            )

        return data


__all__ = [
    "CVRPMemmapVehNodeDataset",
    "build_bipartite_edge_data",
    "decode_routes_zero_sep",
    "strip_trailing_zeros",
]
