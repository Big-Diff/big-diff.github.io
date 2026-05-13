# diffusion/co_datasets/cvrp_dataset.py
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.models.gnn import CVRPVehNodeData


def decode_routes_zero_sep(actions: np.ndarray, depot: int = 0):
    """Decode a zero-separated CVRP tour into a list of non-empty routes.

    Example:
        [0, 3, 5, 0, 2, 1, 0] -> [[3, 5], [2, 1]]

    Customer ids are assumed to be 1-based, with 0 reserved for the depot.
    """
    routes, cur = [], []
    for a in np.asarray(actions, dtype=np.int64).reshape(-1).tolist():
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
    """Strip only padding zeros at the end; keep internal depot separators."""
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
    """Build an instance-local partition label y for CVRP customers.

    y[i] is the reference route id of client i+1. Under pairwise-partition
    supervision, this label is primarily used through the equality relation
    y[i] == y[j]. Its numeric slot id is only a weak row-CE anchor and must not
    determine the graph slot count.
    """
    N = int(V) - 1
    y = -np.ones((N,), dtype=np.int64)

    for rid, route in enumerate(routes):
        for cid in route:
            cid = int(cid)
            if cid <= 0 or cid >= V:
                raise ValueError(f"[npz_dataset] illegal cid={cid}, V={V}")
            if y[cid - 1] != -1:
                raise ValueError(f"[npz_dataset] duplicate client {cid} in routes")
            y[cid - 1] = int(rid)

    if bool((y < 0).any()):
        miss = int((y < 0).sum())
        raise ValueError(f"[npz_dataset] unassigned clients: {miss}/{N}")
    return y


def _as_1d_demand(demand_linehaul, V: int) -> np.ndarray:
    """Return customer-only demand with shape (N,)."""
    N = int(V) - 1
    d = np.asarray(demand_linehaul, dtype=np.float32).reshape(-1)
    if d.size == V:
        return d[1:].astype(np.float32)
    if d.size == N:
        return d.astype(np.float32)
    raise ValueError(
        f"[npz_dataset] demand shape mismatch: got {demand_linehaul.shape}, "
        f"expected {(N,)} or {(V,)}"
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
    knn_k: int = 8,
) -> CVRPVehNodeData:
    """Build a Kmax-based bipartite CVRP graph aligned with EdgeBipartiteDenoiserV4.

    Required GNN shapes:
        node_features : (N, 10)
        veh_features  : (K_max, 7)
        edge_attr     : (K_max * N, 8)
        graph_feat    : (1, 6)

    Slot semantics:
        K_graph / K_max is the visible candidate-slot count used by the GNN.
        K_ref_used is the number of non-empty routes in the reference solution.
        y is an instance-local partition id. It is valid as a weak row anchor,
        while the main pairwise partition loss should use only y_i == y_j.

    No route canonicalization is performed here. In a pairwise-partition objective,
    artificial route ordering is not the main supervision target and should not be
    made part of the dataset semantics.
    """
    del seed, instance_id, knn_k  # kept for call-site compatibility

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"[build_bipartite_edge_data] bad points shape: {points.shape}")

    V = int(points.shape[0])
    if V < 2:
        raise ValueError(f"[build_bipartite_edge_data] V too small: {V}")
    N = V - 1

    if K_max is None or int(K_max) <= 0:
        raise ValueError(
            "[build_bipartite_edge_data] K_max must be explicitly provided. "
            "Do not infer candidate vehicle slots from GT routes."
        )
    K_graph = int(K_max)

    # Sparse candidate edges were previously built using GT labels. That leaks the
    # reference assignment into the candidate space, so keep the graph fully
    # bipartite under strict Kmax semantics.
    if sparse_factor is not None and int(sparse_factor) not in (-1, 0):
        raise ValueError(
            "[build_bipartite_edge_data] sparse_factor > 0 is disabled because "
            "GT labels should not determine candidate edges. Use -1 or 0."
        )

    depot_xy = points[0].astype(np.float32)
    client_xy = points[1:].astype(np.float32)
    demand = _as_1d_demand(demand_linehaul, V=V)

    cap = float(np.asarray(capacity).reshape(-1)[0])
    spd = float(np.asarray(speed).reshape(-1)[0])
    if cap <= 0:
        raise ValueError(f"[build_bipartite_edge_data] capacity must be positive, got {cap}")

    actions = strip_trailing_zeros(np.asarray(actions, dtype=np.int64).reshape(-1))
    if actions.size == 0 or bool(np.all(actions == 0)):
        raise ValueError("[build_bipartite_edge_data] bad actions: empty/all zeros")

    routes = decode_routes_zero_sep(actions, depot=0)
    K_ref_used = int(len(routes))
    if K_ref_used <= 0:
        raise ValueError(f"[build_bipartite_edge_data] K_ref_used=0, routes={len(routes)}")
    if K_ref_used > K_graph:
        raise ValueError(
            f"[build_bipartite_edge_data] K_ref_used={K_ref_used} exceeds K_max={K_graph}. "
            "Increase --K_max/--k_max/--num_vehicles."
        )

    # Instance-local GT partition id. Numeric ids follow the input solution order;
    # no artificial sorting is applied. Pairwise supervision is permutation-invariant
    # because it uses only equality between these ids.
    y = _infer_assignment_from_routes(routes, V=V)

    # ------------------------------------------------------------------
    # Customer/node features: fixed 10 dims expected by EdgeBipartiteDenoiserV4.
    # The first three columns are used explicitly by the GNN for local geometry:
    # node_xy = node_features[:, :2], node_r = node_features[:, 2].
    # ------------------------------------------------------------------
    rel = (client_xy - depot_xy[None, :]).astype(np.float32)
    x_rel = rel[:, 0]
    y_rel = rel[:, 1]
    r = np.sqrt((rel * rel).sum(axis=1) + 1e-12).astype(np.float32)
    theta = np.arctan2(y_rel, x_rel).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    demand_norm = (demand / max(cap, 1e-6)).astype(np.float32)

    # Keep node_in_dim=10 without dataset-side KNN statistics. This lets the GNN
    # build n2n edges on the fly with its own n2n_knn_k, avoiding two inconsistent
    # KNN definitions and large CPU-side caches.
    zeros4 = np.zeros((N, 4), dtype=np.float32)
    node_feat = np.concatenate(
        [
            np.stack([x_rel, y_rel, r, sin_theta, cos_theta, demand_norm], axis=1).astype(np.float32),
            zeros4,
        ],
        axis=1,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Vehicle features: fixed 7 dims expected by the GNN.
    # Do not encode K_ref_used. K_graph/Kmax is allowed because it is a physical
    # model-side candidate-space choice, not an oracle route count.
    # ------------------------------------------------------------------
    veh_feat = np.zeros((K_graph, 7), dtype=np.float32)
    cap_norm = np.float32(cap / max(float(demand.mean()), 1e-6))
    veh_feat[:, 0] = cap_norm
    veh_feat[:, 1] = np.float32(spd)
    veh_feat[:, 2:4] = depot_xy[None, :]
    veh_feat[:, 4] = np.float32(K_graph / np.sqrt(max(1.0, float(N))))

    ids = (np.arange(K_graph, dtype=np.float32) + 0.5) / max(1, K_graph)
    veh_feat[:, 5] = np.sin(2.0 * np.pi * ids)
    veh_feat[:, 6] = np.cos(2.0 * np.pi * ids)

    # ------------------------------------------------------------------
    # Full Kmax x N bipartite graph.
    # edge_index[0] is vehicle/slot, edge_index[1] is customer/node.
    # ------------------------------------------------------------------
    src = np.repeat(np.arange(K_graph, dtype=np.int64), N)
    dst = np.tile(np.arange(N, dtype=np.int64), K_graph)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # Edge features: fixed 8 dims expected by the GNN. Keep the customer-side
    # geometry and demand in the first 6 dims; set the old dataset-KNN stats to 0.
    edge_attr = np.stack(
        [
            x_rel[dst],
            y_rel[dst],
            r[dst],
            sin_theta[dst],
            cos_theta[dst],
            demand_norm[dst],
            np.zeros_like(x_rel[dst], dtype=np.float32),
            np.zeros_like(x_rel[dst], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    total_dem_over_cap = np.float32(demand.sum() / max(cap, 1e-6))
    mean_dem_over_cap = np.float32(demand.mean() / max(cap, 1e-6))

    cap_lb = np.float32(np.ceil(total_dem_over_cap))
    cap_lb_ratio = np.float32(cap_lb / max(1.0, float(K_graph)))

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

    # K_used is retained for legacy call sites, but under strict Kmax semantics it
    # means visible candidate slots, not the reference route count.
    data.K_used = torch.tensor([int(K_graph)], dtype=torch.long)
    data.K_max = torch.tensor([int(K_graph)], dtype=torch.long)
    data.K_ref_used = torch.tensor([int(K_ref_used)], dtype=torch.long)

    data.depot_xy = torch.from_numpy(depot_xy).float().unsqueeze(0).unsqueeze(0)
    data.src_count = int(K_graph)
    data.dst_count = int(N)
    data.node_batch = torch.zeros(N, dtype=torch.long)
    data.veh_batch = torch.zeros(K_graph, dtype=torch.long)

    if gt_cost is None:
        data.gt_cost = torch.tensor([float("nan")], dtype=torch.float32)
    else:
        data.gt_cost = torch.tensor([float(gt_cost)], dtype=torch.float32)

    if keep_raw:
        data.points = torch.from_numpy(points).float()
        data.actions = torch.from_numpy(actions.astype(np.int64))
        data.demand_full = torch.from_numpy(np.concatenate([[0.0], demand]).astype(np.float32))


    return data


class CVRPNPZVehNodeDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        K_max: Optional[int] = None,
        *,
        sparse_factor: int = -1,
        seed: int = 1234,
        max_instances=None,
        keep_raw: bool = False,
        knn_k: int = 8,
    ):
        self.path = npz_path
        self.data = np.load(npz_path, allow_pickle=False)

        if K_max is None or int(K_max) <= 0:
            raise ValueError(
                "[CVRPNPZVehNodeDataset] K_max must be explicitly provided. "
                "Refusing to infer slot count from actions/best_tour."
            )
        self.K_max = int(K_max)

        self.sparse_factor = int(sparse_factor) if sparse_factor is not None else -1
        if self.sparse_factor not in (-1, 0):
            raise ValueError(
                "[CVRPNPZVehNodeDataset] sparse_factor > 0 is disabled because the old "
                "implementation used GT labels to construct candidate edges."
            )

        self.seed = int(seed)
        self.keep_raw = bool(keep_raw)
        self.knn_k = int(knn_k)  # accepted for compatibility; GNN builds n2n edges on the fly.

        self.locs = self.data["locs"]
        self.dem = self.data["demand_linehaul"]
        self.cap = self.data["vehicle_capacity"]
        self.speed = self.data["speed"] if "speed" in self.data.files else None
        self.num_depots = self.data["num_depots"] if "num_depots" in self.data.files else None

        if "actions" in self.data.files:
            self._actions = self.data["actions"]
            self._costs = self.data["costs"] if "costs" in self.data.files else None
            self._cost_mode = "costs"
        elif "best_tour" in self.data.files:
            self._actions = self.data["best_tour"]
            self._costs = self.data["best_cost"] if "best_cost" in self.data.files else None
            self._cost_mode = "best_cost"
        else:
            raise KeyError(f"[CVRPNPZVehNodeDataset] need actions/best_tour. keys={self.data.files}")

        self.B = int(self.locs.shape[0])
        if max_instances is not None:
            self.B = min(self.B, int(max_instances))

    def __len__(self):
        return self.B

    def __getitem__(self, idx: int) -> CVRPVehNodeData:
        idx = int(idx) % self.B

        if self.num_depots is not None:
            nd = int(np.asarray(self.num_depots[idx]).reshape(-1)[0])
            if nd != 1:
                raise ValueError(
                    f"[CVRPNPZVehNodeDataset] current bipartite loader assumes single depot, got num_depots={nd}"
                )

        points = np.asarray(self.locs[idx], dtype=np.float32)
        demand = np.asarray(self.dem[idx], dtype=np.float32)
        cap = float(np.asarray(self.cap[idx]).reshape(-1)[0]) if np.asarray(self.cap).ndim > 0 else float(self.cap)
        spd = 1.0 if self.speed is None else float(np.asarray(self.speed[idx]).reshape(-1)[0])

        actions = np.asarray(self._actions[idx], dtype=np.int64).reshape(-1)
        actions = strip_trailing_zeros(actions)

        gt_cost = None
        if self._costs is not None:
            raw = float(np.asarray(self._costs[idx]).reshape(-1)[0])
            if np.isfinite(raw):
                gt_cost = abs(raw) if self._cost_mode == "costs" else raw

        return build_bipartite_edge_data(
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


__all__ = [
    "CVRPNPZVehNodeDataset",
    "build_bipartite_edge_data",
    "decode_routes_zero_sep",
    "strip_trailing_zeros",
]
