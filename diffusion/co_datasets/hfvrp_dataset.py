# diffusion/co_datasets/hfvrp_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.models.gnn import CVRPVehNodeData


# -----------------------------------------------------------------------------
# Route / array helpers
# -----------------------------------------------------------------------------

def decode_routes_zero_sep(actions: np.ndarray, depot: int = 0):
    """Decode 0-separated route encoding into a list of client routes."""
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
    """Strip padding zeros, while keeping internal depot separators."""
    actions_1d = np.asarray(actions_1d, dtype=np.int64).reshape(-1)
    if actions_1d.size == 0:
        return actions_1d
    end = actions_1d.size
    while end > 0 and actions_1d[end - 1] == 0:
        end -= 1
    return actions_1d[: max(end, 1)]


def _as_int64_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"[hfvrp_dataset] {name} must be 1D after reshape.")
    return arr


def _as_float32_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"[hfvrp_dataset] {name} must be 1D after reshape.")
    return arr


def _filter_valid_route_slots(route_vehicle_slots) -> np.ndarray:
    slots = _as_int64_1d(route_vehicle_slots, "route_vehicle_slots")
    return slots[slots >= 0]

def _normalize_mode_to_str(x) -> str:
    """Best-effort scalar/string normalization for NPZ metadata fields."""
    if x is None:
        return ""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return str(arr.item()).strip().lower()
    flat = arr.reshape(-1)
    return "" if flat.size == 0 else str(flat[0]).strip().lower()


def _take_instance_or_shared(arr, idx: int, B: int):
    """Read per-instance, singleton, shared, or scalar NPZ fields."""
    if arr is None:
        return None
    if np.isscalar(arr):
        return arr

    arr = np.asarray(arr, dtype=object if getattr(arr, "dtype", None) == object else None)
    if arr.ndim == 0:
        return arr.item()
    if arr.shape[0] == B:
        return arr[idx]
    if arr.shape[0] == 1:
        x = arr[0]
        return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x
    return arr


# -----------------------------------------------------------------------------
# Strict slot supervision: raw labels + type-wise geometric canonicalization
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HFVRPStrictSlotTargets:
    """Canonical slot labels for strict HFVRP Stage-A supervision."""

    y_slot: np.ndarray
    perm_canonical_to_raw: np.ndarray


def infer_assignment_from_routes_with_slots(
    routes: Sequence[Sequence[int]],
    V: int,
    route_vehicle_slots,
    K_all: int,
) -> np.ndarray:
    """Return y_raw_slot (N,), where y_raw_slot[i] is the raw slot of client i+1."""
    N = int(V) - 1
    if N <= 0:
        raise ValueError(f"[hfvrp_dataset] invalid V={V}")

    slots = _filter_valid_route_slots(route_vehicle_slots)
    if len(slots) < len(routes):
        raise ValueError(
            f"[hfvrp_dataset] route_vehicle_slots has only {len(slots)} valid slots, "
            f"but decoded {len(routes)} routes."
        )

    y = -np.ones((N,), dtype=np.int64)
    for rid, route in enumerate(routes):
        raw_slot = int(slots[rid])
        if raw_slot < 0 or raw_slot >= int(K_all):
            raise ValueError(f"[hfvrp_dataset] illegal raw slot {raw_slot}, K_all={K_all}")

        for cid in route:
            cid = int(cid)
            if cid <= 0 or cid >= int(V):
                raise ValueError(f"[hfvrp_dataset] illegal client id {cid}, V={V}")
            pos = cid - 1
            if y[pos] != -1:
                raise ValueError(f"[hfvrp_dataset] duplicate client {cid} across routes")
            y[pos] = raw_slot

    if (y < 0).any():
        raise ValueError(f"[hfvrp_dataset] unassigned clients: {int((y < 0).sum())}/{N}")
    return y


def _attribute_only_slot_permutation(
    tier_vec: np.ndarray,
    fixed_vec: np.ndarray,
    unit_cost_vec: np.ndarray,
    cap_vec: np.ndarray,
    speed_vec: np.ndarray,
    K_all: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a solution-independent canonical slot order from vehicle attributes only.

    For the current 3-type HFVRP data, this is effectively type-first:
    small -> medium -> large, with raw slot id as the final deterministic
    tie-breaker inside each type. The extra attribute keys make the ordering
    robust to future datasets where the tier id is present but not sufficient.
    """
    tier_vec = _as_int64_1d(tier_vec, "tier_vec")
    fixed_vec = _as_float32_1d(fixed_vec, "fixed_vec")
    unit_cost_vec = _as_float32_1d(unit_cost_vec, "unit_cost_vec")
    cap_vec = _as_float32_1d(cap_vec, "cap_vec")
    speed_vec = _as_float32_1d(speed_vec, "speed_vec")

    if not (len(tier_vec) == len(fixed_vec) == len(unit_cost_vec) == len(cap_vec) == len(speed_vec) == int(K_all)):
        raise ValueError("[hfvrp_dataset] all vehicle arrays must have length K_all")

    def _q(x: float, ndigits: int = 8) -> float:
        return float(np.round(float(x), ndigits))

    perm_canonical_to_raw = np.asarray(
        sorted(
            range(int(K_all)),
            key=lambda raw: (
                int(tier_vec[raw]),
                -_q(cap_vec[raw]),
                _q(unit_cost_vec[raw]),
                _q(fixed_vec[raw]),
                -_q(speed_vec[raw]),
                int(raw),
            ),
        ),
        dtype=np.int64,
    )
    raw_to_canonical = np.empty((int(K_all),), dtype=np.int64)
    raw_to_canonical[perm_canonical_to_raw] = np.arange(int(K_all), dtype=np.int64)
    return perm_canonical_to_raw, raw_to_canonical


def _route_geometry_canonical_key(
    route: Sequence[int],
    points: np.ndarray,
    demand_linehaul: np.ndarray,
):
    """Geometry-based route key used only for within-type label canonicalization.

    Client ids in routes are 1-based. The key mirrors CVRP route canonicalization:
      1) polar angle of the route centroid around the depot,
      2) centroid radius,
      3) route load, descending,
      4) minimum client id,
      5) route length,
      6) full client tuple as a deterministic final tie-breaker.
    """
    r = np.asarray(route, dtype=np.int64).reshape(-1)
    if r.size == 0:
        return (float("inf"), float("inf"), 0.0, 10**18, 0, ())

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(f"[hfvrp_dataset] bad points shape for route canonicalization: {points.shape}")

    V = int(points.shape[0])
    N = V - 1
    if np.any(r <= 0) or np.any(r > N):
        raise ValueError(f"[hfvrp_dataset] illegal route client id in route={route}, N={N}")

    demand_arr = np.asarray(demand_linehaul, dtype=np.float32).reshape(-1)
    if demand_arr.size == V:
        demand = demand_arr[1:]
    elif demand_arr.size == N:
        demand = demand_arr
    else:
        raise ValueError(
            f"[hfvrp_dataset] demand shape mismatch for route canonicalization: "
            f"got {demand_arr.shape}, expected {(N,)} or {(V,)}"
        )

    depot_xy = points[0].astype(np.float32)
    pts = points[r].astype(np.float32)  # route ids are 1-based, so points[r] is correct.
    centroid = pts.mean(axis=0)
    vec = centroid - depot_xy

    angle = float(np.arctan2(vec[1], vec[0]))
    if angle < 0.0:
        angle += float(2.0 * np.pi)

    radius = float(np.linalg.norm(vec))
    load = float(demand[r - 1].sum())
    min_cid = int(r.min())

    return (
        angle,
        radius,
        -load,
        min_cid,
        int(r.size),
        tuple(int(x) for x in r.tolist()),
    )


def _canonicalize_routes_within_type_by_geometry(
    routes: Sequence[Sequence[int]],
    route_slots_raw: np.ndarray,
    tier_vec: np.ndarray,
    points: np.ndarray,
    demand_linehaul: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    """Reassign used raw slots within each vehicle type by geometric route order.

    Different vehicle types are not exchangeable, so no route can move across types.
    Inside each type, slots are exchangeable for the current 3-type data. We sort
    routes by geometry and assign them to the sorted raw slots of the same type.
    This stabilizes supervision labels but never changes the full candidate slot
    space or masks any vehicle-customer edge.
    """
    routes_l = [list(map(int, r)) for r in routes]
    route_slots_raw = _as_int64_1d(route_slots_raw, "route_slots_raw")[: len(routes_l)]
    tier_vec = _as_int64_1d(tier_vec, "tier_vec")

    if len(route_slots_raw) < len(routes_l):
        raise ValueError(
            f"[hfvrp_dataset] route_slots_raw has only {len(route_slots_raw)} slots, "
            f"but decoded {len(routes_l)} routes."
        )

    buckets: dict[int, list[tuple[list[int], int]]] = {}
    for route, raw_slot in zip(routes_l, route_slots_raw.tolist()):
        raw_slot = int(raw_slot)
        if raw_slot < 0 or raw_slot >= int(tier_vec.size):
            raise ValueError(f"[hfvrp_dataset] illegal raw_slot={raw_slot}, K_all={tier_vec.size}")
        gid = int(tier_vec[raw_slot])
        buckets.setdefault(gid, []).append((route, raw_slot))

    out_routes: list[list[int]] = []
    out_slots: list[int] = []
    for gid in sorted(buckets.keys()):
        items = buckets[gid]
        slots_sorted = sorted(int(slot) for _, slot in items)
        routes_sorted = sorted(
            [route for route, _ in items],
            key=lambda r: _route_geometry_canonical_key(r, points=points, demand_linehaul=demand_linehaul),
        )
        if len(routes_sorted) != len(slots_sorted):
            raise RuntimeError("[hfvrp_dataset] internal type-wise route canonicalization mismatch")
        out_routes.extend(routes_sorted)
        out_slots.extend(slots_sorted)

    return out_routes, np.asarray(out_slots, dtype=np.int64)
def build_hfvrp_strict_slot_targets(
    routes: Sequence[Sequence[int]],
    V: int,
    route_vehicle_slots,
    tier_vec,
    fixed_vec,
    unit_cost_vec,
    cap_vec,
    speed_vec,
    K_all: int,
    points=None,
    demand_linehaul=None,
) -> HFVRPStrictSlotTargets:
    """Build type-wise geometry-canonical supervision labels.

    Vehicle slots are still ordered by their own attributes. The used reference
    routes are canonicalized only within the same vehicle type by a CVRP-style
    geometric key. This is the current HFVRP label-stabilization rule.
    """

    K_all = int(K_all)
    route_slots_raw = _filter_valid_route_slots(route_vehicle_slots)[: len(routes)]

    if points is None or demand_linehaul is None:
        raise ValueError(
            "[hfvrp_dataset] type-wise geometric label canonicalization requires "
            "points and demand_linehaul."
        )

    routes, route_slots_raw = _canonicalize_routes_within_type_by_geometry(
        routes=routes,
        route_slots_raw=route_slots_raw,
        tier_vec=np.asarray(tier_vec, dtype=np.int64),
        points=np.asarray(points, dtype=np.float32),
        demand_linehaul=np.asarray(demand_linehaul, dtype=np.float32),
    )

    y_raw_slot = infer_assignment_from_routes_with_slots(
        routes=routes,
        V=V,
        route_vehicle_slots=route_slots_raw,
        K_all=K_all,
    )

    perm_canonical_to_raw, raw_to_canonical = _attribute_only_slot_permutation(
        tier_vec=np.asarray(tier_vec, dtype=np.int64),
        fixed_vec=np.asarray(fixed_vec, dtype=np.float32),
        unit_cost_vec=np.asarray(unit_cost_vec, dtype=np.float32),
        cap_vec=np.asarray(cap_vec, dtype=np.float32),
        speed_vec=np.asarray(speed_vec, dtype=np.float32),
        K_all=K_all,
    )

    y_slot = raw_to_canonical[y_raw_slot]

    return HFVRPStrictSlotTargets(
        y_slot=y_slot.astype(np.int64),
        perm_canonical_to_raw=perm_canonical_to_raw.astype(np.int64),
    )

# -----------------------------------------------------------------------------
# Main graph builder
# -----------------------------------------------------------------------------

def build_bipartite_edge_data_hfvrp(
    points,
    demand_linehaul,
    vehicle_capacity,
    vehicle_tier,
    vehicle_unit_distance_cost,
    actions,
    route_vehicle_slots,
    gt_cost,
) -> CVRPVehNodeData:

    points = np.asarray(points, dtype=np.float32)
    V = int(points.shape[0])
    if V < 2:
        raise ValueError(f"[build_bipartite_edge_data_hfvrp] V too small: {V}")
    N = V - 1
    depot_xy = points[0].astype(np.float32)
    client_xy = points[1:].astype(np.float32)

    demand_linehaul = np.asarray(demand_linehaul, dtype=np.float32).reshape(-1)
    if demand_linehaul.size == V:
        demand = demand_linehaul[1:]
    elif demand_linehaul.size == N:
        demand = demand_linehaul
    else:
        raise ValueError(
            f"[build_bipartite_edge_data_hfvrp] demand shape mismatch: got {demand_linehaul.shape}, "
            f"expected {(N,)} or {(V,)}"
        )
    demand = demand.astype(np.float32).reshape(N)

    cap_vec_raw = _as_float32_1d(vehicle_capacity, "vehicle_capacity")
    K_all = int(cap_vec_raw.size)
    if K_all <= 0:
        raise ValueError("[build_bipartite_edge_data_hfvrp] empty vehicle_capacity")

    tier_raw = _as_int64_1d(vehicle_tier, "vehicle_tier")
    unit_raw = _as_float32_1d(vehicle_unit_distance_cost, "vehicle_unit_distance_cost")

    if tier_raw.size != K_all:
        raise ValueError(
            f"[build_bipartite_edge_data_hfvrp] vehicle_tier size mismatch: "
            f"got {tier_raw.size}, expected K_all={K_all}"
        )
    if unit_raw.size != K_all:
        raise ValueError(
            f"[build_bipartite_edge_data_hfvrp] vehicle_unit_distance_cost size mismatch: "
            f"got {unit_raw.size}, expected K_all={K_all}"
        )

    fixed_raw = np.zeros((K_all,), dtype=np.float32)
    speed_raw = np.ones((K_all,), dtype=np.float32)

    actions = strip_trailing_zeros(np.asarray(actions, dtype=np.int64).reshape(-1))
    if actions.size == 0 or np.all(actions == 0):
        raise ValueError("[build_bipartite_edge_data_hfvrp] bad actions: empty/all zeros")
    routes = decode_routes_zero_sep(actions, depot=0)
    K_ref_used = int(len(routes))
    if K_ref_used <= 0:
        raise ValueError(f"[build_bipartite_edge_data_hfvrp] no decoded routes")

    targets = build_hfvrp_strict_slot_targets(
        routes=routes,
        V=V,
        route_vehicle_slots=route_vehicle_slots,
        tier_vec=tier_raw,
        fixed_vec=fixed_raw,
        unit_cost_vec=unit_raw,
        cap_vec=cap_vec_raw,
        speed_vec=speed_raw,
        K_all=K_all,
        points=points,
        demand_linehaul=demand,
    )

    perm = targets.perm_canonical_to_raw
    y = targets.y_slot

    cap_vec = cap_vec_raw[perm]
    tier_vec = tier_raw[perm]
    fixed_vec = fixed_raw[perm]
    unit_cost_vec = unit_raw[perm]

    # ------------------------------------------------------------------
    # Node / vehicle / edge features
    # ------------------------------------------------------------------
    rel = (client_xy - depot_xy[None, :]).astype(np.float32)
    x_rel = rel[:, 0]
    y_rel = rel[:, 1]
    r = np.sqrt((rel ** 2).sum(axis=1) + 1e-12).astype(np.float32)
    theta = np.arctan2(y_rel, x_rel).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)

    cap_ref = max(float(cap_vec.max()), 1e-6)

    demand_feat = (demand / cap_ref).astype(np.float32)
    veh_cap_feat = (cap_vec / cap_ref).astype(np.float32)
    veh_fixed_feat = np.zeros_like(fixed_vec, dtype=np.float32)
    veh_unit_cost_feat = (
            unit_cost_vec / max(float(np.max(unit_cost_vec)), 1.0)
    ).astype(np.float32)

    # Do not precompute/store node-kNN graphs or dataset-side KNN statistics.
    # The GNN builds batched KNN on GPU from --n2n_knn_k. Keep node_in_dim=10
    # by padding the last four local-stat columns with zeros.
    zeros4 = np.zeros((N, 4), dtype=np.float32)
    node_feat = np.concatenate(
        [
            np.stack([x_rel, y_rel, r, sin_theta, cos_theta, demand_feat], axis=1).astype(np.float32),
            zeros4,
        ],
        axis=1,
    ).astype(np.float32)

    veh_feat = np.zeros((K_all, 7), dtype=np.float32)
    veh_feat[:, 0] = veh_cap_feat
    veh_feat[:, 1] = 1.0
    veh_feat[:, 2] = veh_fixed_feat
    veh_feat[:, 3] = veh_unit_cost_feat
    tier_max = float(np.max(np.abs(tier_vec))) if tier_vec.size > 0 else 0.0
    veh_feat[:, 4] = (tier_vec / tier_max if tier_max > 0 else tier_vec).astype(np.float32)
    ids = (np.arange(K_all, dtype=np.float32) + 0.5) / max(1, K_all)
    veh_feat[:, 5] = np.sin(2.0 * np.pi * ids)
    veh_feat[:, 6] = np.cos(2.0 * np.pi * ids)

    src = np.repeat(np.arange(K_all, dtype=np.int64), N)
    dst = np.tile(np.arange(N, dtype=np.int64), K_all)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = np.stack(
        [
            x_rel[dst],
            y_rel[dst],
            r[dst],
            sin_theta[dst],
            cos_theta[dst],
            demand_feat[dst],
            np.zeros_like(x_rel[dst], dtype=np.float32),
            np.zeros_like(x_rel[dst], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    data = CVRPVehNodeData(
        veh_features=torch.from_numpy(veh_feat),
        node_features=torch.from_numpy(node_feat),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y.astype(np.int64)),
    )

    data.vehicle_capacity = torch.from_numpy(cap_vec.astype(np.float32)).float()
    data.vehicle_tier = torch.from_numpy(tier_vec.astype(np.int64)).long()
    data.vehicle_fixed_cost = torch.from_numpy(fixed_vec.astype(np.float32)).float()
    data.vehicle_unit_distance_cost = torch.from_numpy(unit_cost_vec.astype(np.float32)).float()

    data.demand_linehaul = torch.from_numpy(demand.astype(np.float32)).float()
    data.num_vehicle_slots = torch.tensor([int(K_all)], dtype=torch.long)
    data.K_used = data.num_vehicle_slots  # backward compatibility
    data.depot_xy = torch.from_numpy(depot_xy).float().unsqueeze(0).unsqueeze(0)
    data.gt_cost = torch.tensor([float(gt_cost)], dtype=torch.float32)

    data.node_batch = torch.zeros(N, dtype=torch.long)
    data.veh_batch = torch.zeros(K_all, dtype=torch.long)

    return data


# -----------------------------------------------------------------------------
# NPZ dataset
# -----------------------------------------------------------------------------

class HFVRPNPZVehNodeDataset(Dataset):
    """Strict HFVRP NPZ dataset.

    Required NPZ schema:
        locs
        demand_linehaul
        vehicle_capacity
        vehicle_tier
        vehicle_unit_distance_cost
        route_vehicle_slots
        actions
        costs

    Notes:
        - vehicle_fixed_cost is not read from NPZ. It is fixed to zero in the
          graph builder because the current dataset does not contain fixed costs.
        - No dataset-side KNN, no sparse GT-aware edge pruning, no legacy slot
          ordering modes.
    """

    REQUIRED_KEYS = (
        "locs",
        "demand_linehaul",
        "vehicle_capacity",
        "vehicle_tier",
        "vehicle_unit_distance_cost",
        "route_vehicle_slots",
        "actions",
        "costs",
    )

    def __init__(self, npz_path: str, *, max_instances=None):
        self.path = npz_path
        self.data = np.load(npz_path, allow_pickle=False)

        missing = [k for k in self.REQUIRED_KEYS if k not in self.data.files]
        if missing:
            raise KeyError(f"[HFVRPNPZVehNodeDataset] missing required NPZ fields: {missing}")

        self.locs = self.data["locs"]
        self.dem = self.data["demand_linehaul"]
        self.vehicle_capacity = self.data["vehicle_capacity"]
        self.vehicle_tier = self.data["vehicle_tier"]
        self.vehicle_unit_distance_cost = self.data["vehicle_unit_distance_cost"]
        self.route_vehicle_slots = self.data["route_vehicle_slots"]
        self.actions = self.data["actions"]
        self.costs = self.data["costs"]

        self.B = int(self.locs.shape[0])
        if max_instances is not None:
            self.B = min(self.B, int(max_instances))

    def __len__(self):
        return self.B

    def __getitem__(self, idx: int) -> CVRPVehNodeData:
        idx = int(idx) % self.B

        points = np.asarray(self.locs[idx], dtype=np.float32)
        demand = np.asarray(self.dem[idx], dtype=np.float32)
        actions = strip_trailing_zeros(
            np.asarray(self.actions[idx], dtype=np.int64).reshape(-1)
        )

        gt_cost = float(np.asarray(self.costs[idx]).reshape(-1)[0])
        gt_cost = abs(gt_cost) if np.isfinite(gt_cost) else float("nan")

        cap_vec = np.asarray(
            _take_instance_or_shared(self.vehicle_capacity, idx, self.B),
            dtype=np.float32,
        ).reshape(-1)

        tier_vec = np.asarray(
            _take_instance_or_shared(self.vehicle_tier, idx, self.B),
            dtype=np.int64,
        ).reshape(-1)

        unit_cost_vec = np.asarray(
            _take_instance_or_shared(self.vehicle_unit_distance_cost, idx, self.B),
            dtype=np.float32,
        ).reshape(-1)

        route_slots = np.asarray(
            _take_instance_or_shared(self.route_vehicle_slots, idx, self.B),
            dtype=np.int64,
        ).reshape(-1)

        return build_bipartite_edge_data_hfvrp(
            points=points,
            demand_linehaul=demand,
            vehicle_capacity=cap_vec,
            vehicle_tier=tier_vec,
            vehicle_unit_distance_cost=unit_cost_vec,
            actions=actions,
            route_vehicle_slots=route_slots,
            gt_cost=gt_cost,
        )

__all__ = [
    "HFVRPNPZVehNodeDataset",
    "build_bipartite_edge_data_hfvrp",
    "build_hfvrp_strict_slot_targets",
    "infer_assignment_from_routes_with_slots",
    "decode_routes_zero_sep",
    "strip_trailing_zeros",
]
