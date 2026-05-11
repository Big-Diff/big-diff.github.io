# diffusion/co_datasets/hfvrp_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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


def _expand_vehicle_field(arr, K_all: int, *, default_value: float, name: str, dtype=np.float32) -> np.ndarray:
    """Expand scalar / length-K vehicle field into shape (K_all,)."""
    if arr is None:
        return np.full((K_all,), default_value, dtype=dtype)

    x = np.asarray(arr, dtype=dtype).reshape(-1)
    if x.size == 0:
        return np.full((K_all,), default_value, dtype=dtype)
    if x.size == 1:
        return np.full((K_all,), x.item(), dtype=dtype)
    if x.size == K_all:
        return x.astype(dtype, copy=False)
    raise ValueError(f"[hfvrp_dataset] {name} size mismatch: got {x.size}, expected 1 or K_all={K_all}")


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
    """Solution-safe HFVRP slot supervision bundle.

    y_raw_slot is derived from the reference solution for supervision only.
    The canonical slot order, when enabled, is computed from vehicle attributes
    only and never from GT-used slots, route geometry, route load, or route order.
    """

    y_raw_slot: np.ndarray
    y_slot: np.ndarray
    route_slots_raw: np.ndarray
    route_slots_canonical: np.ndarray
    perm_canonical_to_raw: np.ndarray
    raw_to_canonical: np.ndarray
    used_vehicle_mask: np.ndarray
    slot_group: np.ndarray
    slot_local_index: np.ndarray
    y_group: np.ndarray
    y_slot_in_group: np.ndarray


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


def _slot_local_index_within_group(slot_group: np.ndarray) -> np.ndarray:
    slot_group = _as_int64_1d(slot_group, "slot_group")
    out = np.empty_like(slot_group, dtype=np.int64)
    counts: dict[int, int] = {}
    for cslot, g in enumerate(slot_group.tolist()):
        gid = int(g)
        out[cslot] = counts.get(gid, 0)
        counts[gid] = out[cslot] + 1
    return out


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
    slot_order: str = "type_geo",
    points=None,
    demand_linehaul=None,
) -> HFVRPStrictSlotTargets:
    """Build type-wise geometry-canonical supervision labels.

    Vehicle slots are still ordered by their own attributes. The used reference
    routes are canonicalized only within the same vehicle type by a CVRP-style
    geometric key. This is the current HFVRP label-stabilization rule.
    """
    del slot_order  # kept for CLI/backward compatibility; no alternate mode is used.

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
    route_slots_canonical = raw_to_canonical[route_slots_raw]

    used_vehicle_mask = np.zeros((K_all,), dtype=np.bool_)
    used_vehicle_mask[route_slots_canonical] = True

    slot_group = _as_int64_1d(tier_vec, "tier_vec")[perm_canonical_to_raw].astype(np.int64, copy=False)
    slot_local_index = _slot_local_index_within_group(slot_group)
    y_group = slot_group[y_slot]
    y_slot_in_group = slot_local_index[y_slot]

    return HFVRPStrictSlotTargets(
        y_raw_slot=y_raw_slot.astype(np.int64),
        y_slot=y_slot.astype(np.int64),
        route_slots_raw=route_slots_raw.astype(np.int64),
        route_slots_canonical=route_slots_canonical.astype(np.int64),
        perm_canonical_to_raw=perm_canonical_to_raw.astype(np.int64),
        raw_to_canonical=raw_to_canonical.astype(np.int64),
        used_vehicle_mask=used_vehicle_mask.astype(np.bool_),
        slot_group=slot_group.astype(np.int64),
        slot_local_index=slot_local_index.astype(np.int64),
        y_group=y_group.astype(np.int64),
        y_slot_in_group=y_slot_in_group.astype(np.int64),
    )



# -----------------------------------------------------------------------------
# Local customer graph
# -----------------------------------------------------------------------------

def _build_knn_single(client_xy: np.ndarray, demand_feat: np.ndarray, k: int = 8):
    """Build per-instance node-kNN graph and local node statistics."""
    N = int(client_xy.shape[0])
    if N <= 1 or int(k) <= 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((N,), dtype=np.float32),
            np.zeros((N,), dtype=np.float32),
            np.zeros((N,), dtype=np.float32),
            np.zeros((N,), dtype=np.float32),
        )

    kk = max(1, min(int(k), N - 1))
    dmat = np.sqrt(((client_xy[:, None, :] - client_xy[None, :, :]) ** 2).sum(axis=-1) + 1e-12).astype(np.float32)
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argpartition(dmat, kth=kk - 1, axis=1)[:, :kk]

    row = np.arange(N, dtype=np.int64)
    src = np.repeat(row, kk)
    dst = nn_idx.reshape(-1).astype(np.int64)

    dxy = client_xy[src] - client_xy[dst]
    dist = np.sqrt((dxy ** 2).sum(axis=-1) + 1e-12).astype(np.float32)
    inv = (1.0 / (dist + 1e-6)).astype(np.float32)
    di = demand_feat[src].astype(np.float32)
    dj = demand_feat[dst].astype(np.float32)
    node_knn_edge_attr = np.stack([dxy[:, 0], dxy[:, 1], dist, inv, di, dj, np.abs(di - dj)], axis=1).astype(np.float32)

    nn_dist_mean = dmat[np.arange(N)[:, None], nn_idx].mean(axis=1).astype(np.float32)
    nn_dist_min = dmat[np.arange(N), nn_idx[:, 0]].astype(np.float32)
    nn_dem_sum = demand_feat[nn_idx].sum(axis=1).astype(np.float32)
    boundary_score = np.clip(nn_dist_mean / (nn_dist_min + 1e-6), 0.0, 10.0).astype(np.float32)
    return np.stack([src, dst], axis=0).astype(np.int64), node_knn_edge_attr, nn_dist_mean, nn_dist_min, nn_dem_sum, boundary_score


# -----------------------------------------------------------------------------
# Main graph builder
# -----------------------------------------------------------------------------

def build_bipartite_edge_data_hfvrp(
    points,
    demand_linehaul,
    vehicle_capacity,
    speed,
    actions,
    route_vehicle_slots,
    vehicle_tier=None,
    vehicle_fixed_cost=None,
    vehicle_unit_distance_cost=None,
    gt_cost=None,
    *,
    sparse_factor: int = -1,
    keep_raw: bool = False,
    dbg: bool = False,
    knn_k: int = 8,
    hf_slot_order: str = "type_geo",
    normalize_by=None,
    hf_feature_scale: str = "auto",
) -> CVRPVehNodeData:
    """Build a strict full-fleet HFVRP bipartite graph.

    Semantics:
      - slot space is always K_all = len(vehicle_capacity), not GT used-route count;
      - graph edges are always full N x K_all;
      - sparse_factor > 0 is forbidden because the old sparse mode kept GT edges;
      - labels use type-wise route-geometry canonicalization;
      - vehicle_slot_mask is all ones for the solver; used_vehicle_mask is diagnostic.
    """
    sparse_factor = int(-1 if sparse_factor is None else sparse_factor)
    if sparse_factor > 0:
        raise ValueError(
            "[build_bipartite_edge_data_hfvrp] sparse_factor > 0 is forbidden. "
            "The old sparse construction used the GT slot to keep positive edges. Use sparse_factor=-1."
        )

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

    speed_raw = _expand_vehicle_field(speed, K_all, default_value=1.0, name="speed", dtype=np.float32)
    tier_raw = _expand_vehicle_field(vehicle_tier, K_all, default_value=0.0, name="vehicle_tier", dtype=np.int64).astype(np.int64)
    fixed_raw = _expand_vehicle_field(vehicle_fixed_cost, K_all, default_value=0.0, name="vehicle_fixed_cost", dtype=np.float32)
    unit_raw = _expand_vehicle_field(vehicle_unit_distance_cost, K_all, default_value=1.0, name="vehicle_unit_distance_cost", dtype=np.float32)

    actions = strip_trailing_zeros(np.asarray(actions, dtype=np.int64).reshape(-1))
    if actions.size == 0 or np.all(actions == 0):
        raise ValueError("[build_bipartite_edge_data_hfvrp] bad actions: empty/all zeros")
    routes = decode_routes_zero_sep(actions, depot=0)
    K_ref_used = int(len(routes))
    if K_ref_used <= 0:
        raise ValueError(f"[build_bipartite_edge_data_hfvrp] no decoded routes")

    # Current mainline: always use type-wise geometric label canonicalization.
    # The argument is kept only for old command compatibility.
    hf_slot_order = "type_geo"

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
        slot_order=str(hf_slot_order),
        points=points,
        demand_linehaul=demand,
    )

    perm = targets.perm_canonical_to_raw
    raw_to_canonical = targets.raw_to_canonical
    y = targets.y_slot

    cap_vec = cap_vec_raw[perm]
    speed_vec = speed_raw[perm]
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

    normalize_mode = _normalize_mode_to_str(normalize_by)
    hf_feature_scale = str(hf_feature_scale).strip().lower()
    use_dataset_scaled_features = hf_feature_scale == "dataset" or (hf_feature_scale == "auto" and normalize_mode not in {"", "none"})

    cap_ref = max(float(cap_vec.max()), 1e-6)
    if use_dataset_scaled_features:
        demand_feat = demand.astype(np.float32)
        veh_cap_feat = cap_vec.astype(np.float32)
        veh_fixed_feat = fixed_vec.astype(np.float32)
        veh_unit_cost_feat = unit_cost_vec.astype(np.float32)
    else:
        demand_feat = (demand / cap_ref).astype(np.float32)
        veh_cap_feat = (cap_vec / cap_ref).astype(np.float32)
        veh_fixed_feat = (fixed_vec / max(float(np.max(fixed_vec)), 1.0)).astype(np.float32)
        veh_unit_cost_feat = (unit_cost_vec / max(float(np.max(unit_cost_vec)), 1.0)).astype(np.float32)

    # Do not precompute/store node-kNN graphs or dataset-side KNN statistics.
    # The GNN builds batched KNN on GPU from --n2n_knn_k. Keep node_in_dim=10
    # by padding the last four local-stat columns with zeros.
    del knn_k
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
    veh_feat[:, 1] = speed_vec.astype(np.float32)
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

    data.capacity = torch.tensor([float(cap_ref)], dtype=torch.float32)  # compatibility shim only
    data.vehicle_capacity = torch.from_numpy(cap_vec.astype(np.float32)).float()
    data.vehicle_speed = torch.from_numpy(speed_vec.astype(np.float32)).float()
    data.vehicle_tier = torch.from_numpy(tier_vec.astype(np.int64)).long()
    data.vehicle_fixed_cost = torch.from_numpy(fixed_vec.astype(np.float32)).float()
    data.vehicle_unit_distance_cost = torch.from_numpy(unit_cost_vec.astype(np.float32)).float()
    data.slot_group = torch.from_numpy(targets.slot_group.astype(np.int64)).long()
    data.slot_local_index = torch.from_numpy(targets.slot_local_index.astype(np.int64)).long()
    data.y_group = torch.from_numpy(targets.y_group.astype(np.int64)).long()
    data.y_slot_in_group = torch.from_numpy(targets.y_slot_in_group.astype(np.int64)).long()

    # Diagnostic-only GT usage targets. They must not be used to crop active edges.
    # The pairwise/type objective needs only graph.y and graph.vehicle_tier.
    data.used_vehicle_mask = torch.from_numpy(targets.used_vehicle_mask.astype(np.bool_))

    # Strict solver semantics: all fleet slots are available candidates.
    data.vehicle_slot_mask = torch.ones((K_all,), dtype=torch.bool)

    data.demand_linehaul = torch.from_numpy(demand.astype(np.float32)).float()
    data.K_used = torch.tensor([int(K_all)], dtype=torch.long)
    data.K_max = torch.tensor([int(K_all)], dtype=torch.long)
    data.K_ref_used = torch.tensor([int(K_ref_used)], dtype=torch.long)
    data.max_vehicles = torch.tensor([int(K_all)], dtype=torch.long)

    data.depot_xy = torch.from_numpy(depot_xy).float().unsqueeze(0).unsqueeze(0)
    data.gt_cost = torch.tensor([float("nan") if gt_cost is None else float(gt_cost)], dtype=torch.float32)
    data.src_count = int(K_all)
    data.dst_count = int(N)
    data.node_batch = torch.zeros(N, dtype=torch.long)
    data.veh_batch = torch.zeros(K_all, dtype=torch.long)
    data.problem_type = "hfvrp"
    data.hf_feature_scale = hf_feature_scale
    data.normalize_by = normalize_mode
    data.hf_slot_order = str(hf_slot_order)

    data.route_vehicle_slots = torch.from_numpy(targets.route_slots_canonical.astype(np.int64)).long()
    data.route_vehicle_slots_raw = torch.from_numpy(targets.route_slots_raw.astype(np.int64)).long()
    data.canonical_to_raw_slot = torch.from_numpy(targets.perm_canonical_to_raw.astype(np.int64)).long()
    data.raw_to_canonical_slot = torch.from_numpy(raw_to_canonical.astype(np.int64)).long()
    data.y_raw_slot = torch.from_numpy(targets.y_raw_slot.astype(np.int64)).long()

    if keep_raw:
        data.points = torch.from_numpy(points).float()
        data.actions = torch.from_numpy(actions.astype(np.int64))
        data.demand_full = torch.from_numpy(np.concatenate([[0.0], demand]).astype(np.float32))

    if dbg:
        print(
            "[DBG][HFVRP strict] "
            f"V={V} N={N} K_all={K_all} K_ref_used={K_ref_used} E={int(edge_index.shape[1])} "
            f"slot_order={hf_slot_order}"
        )
    return data


# -----------------------------------------------------------------------------
# NPZ dataset
# -----------------------------------------------------------------------------

class HFVRPNPZVehNodeDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        *,
        sparse_factor: int = -1,
        max_instances=None,
        keep_raw: bool = False,
        knn_k: Optional[int] = None,
        dataset_knn_k: Optional[int] = None,
        hf_slot_order: str = "type_geo",
    ):
        self.path = npz_path
        self.data = np.load(npz_path, allow_pickle=True)

        self.sparse_factor = int(sparse_factor) if sparse_factor is not None else -1
        if self.sparse_factor > 0:
            raise ValueError(
                "[HFVRPNPZVehNodeDataset] sparse_factor > 0 is forbidden in strict HFVRP mode. "
                "Use sparse_factor=-1 for full N x K_all bipartite edges."
            )

        self.keep_raw = bool(keep_raw)
        # Kept for CLI compatibility only. Node-kNN is not precomputed in the
        # dataset; the GNN builds batched KNN on GPU from its own n2n_knn_k.
        del knn_k, dataset_knn_k
        self.knn_k = 0

        order_arg = str(hf_slot_order or "type_geo").strip().lower()
        if order_arg in {"solution", "reference", "gt"}:
            raise ValueError(
                "[HFVRPNPZVehNodeDataset] global solution-dependent slot ordering is not supported. "
                "The current loader uses type-wise geometric label canonicalization."
            )
        # Ignore legacy values such as attribute/raw: the current HFVRP dataset
        # always canonicalizes labels by type-wise route geometry.
        self.hf_slot_order = "type_geo"

        self.locs = self.data["locs"]
        self.dem = self.data["demand_linehaul"]
        self.vehicle_capacity_norm = self.data["vehicle_capacity"]
        self.speed = self.data["speed"] if "speed" in self.data.files else None
        self.num_depots = self.data["num_depots"] if "num_depots" in self.data.files else None
        self.vehicle_tier = self.data["vehicle_tier"] if "vehicle_tier" in self.data.files else None
        self.vehicle_fixed_cost_norm = self.data["vehicle_fixed_cost"] if "vehicle_fixed_cost" in self.data.files else None
        self.vehicle_unit_distance_cost_norm = self.data["vehicle_unit_distance_cost"] if "vehicle_unit_distance_cost" in self.data.files else None
        self.normalize_by = self.data["normalize_by"] if "normalize_by" in self.data.files else None
        self.route_vehicle_slots = self.data["route_vehicle_slots"] if "route_vehicle_slots" in self.data.files else None

        if "actions" in self.data.files:
            self._actions = self.data["actions"]
            self._costs = self.data["costs"] if "costs" in self.data.files else None
            self._cost_mode = "costs"
        elif "best_tour" in self.data.files:
            self._actions = self.data["best_tour"]
            self._costs = self.data["best_cost"] if "best_cost" in self.data.files else None
            self._cost_mode = "best_cost"
        else:
            raise KeyError(f"[HFVRPNPZVehNodeDataset] need actions/best_tour. keys={self.data.files}")

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
                raise ValueError(f"[HFVRPNPZVehNodeDataset] current loader assumes single depot, got num_depots={nd}")

        points = np.asarray(self.locs[idx], dtype=np.float32)
        demand = np.asarray(self.dem[idx], dtype=np.float32)
        actions = strip_trailing_zeros(np.asarray(self._actions[idx], dtype=np.int64).reshape(-1))

        gt_cost = None
        if self._costs is not None:
            raw = float(np.asarray(self._costs[idx]).reshape(-1)[0])
            if np.isfinite(raw):
                gt_cost = abs(raw) if self._cost_mode == "costs" else raw

        cap_vec = np.asarray(_take_instance_or_shared(self.vehicle_capacity_norm, idx, self.B), dtype=np.float32).reshape(-1)
        spd = 1.0 if self.speed is None else np.asarray(_take_instance_or_shared(self.speed, idx, self.B), dtype=np.float32).reshape(-1)
        tier = None if self.vehicle_tier is None else np.asarray(_take_instance_or_shared(self.vehicle_tier, idx, self.B), dtype=np.float32).reshape(-1)
        vfix = None if self.vehicle_fixed_cost_norm is None else np.asarray(_take_instance_or_shared(self.vehicle_fixed_cost_norm, idx, self.B), dtype=np.float32).reshape(-1)
        vudc = None if self.vehicle_unit_distance_cost_norm is None else np.asarray(_take_instance_or_shared(self.vehicle_unit_distance_cost_norm, idx, self.B), dtype=np.float32).reshape(-1)
        rslots = None if self.route_vehicle_slots is None else np.asarray(_take_instance_or_shared(self.route_vehicle_slots, idx, self.B), dtype=np.int64).reshape(-1)
        if rslots is None:
            raise ValueError("[HFVRPNPZVehNodeDataset] HFVRP mode requires route_vehicle_slots")

        normalize_by = _take_instance_or_shared(self.normalize_by, idx, self.B)

        return build_bipartite_edge_data_hfvrp(
            points=points,
            demand_linehaul=demand,
            vehicle_capacity=cap_vec,
            speed=spd,
            vehicle_tier=tier,
            vehicle_fixed_cost=vfix,
            vehicle_unit_distance_cost=vudc,
            actions=actions,
            route_vehicle_slots=rslots,
            gt_cost=gt_cost,
            sparse_factor=self.sparse_factor,
            keep_raw=self.keep_raw,
            knn_k=self.knn_k,
            hf_slot_order=self.hf_slot_order,
            normalize_by=normalize_by,
            hf_feature_scale="auto",
        )


# Drop-in aliases for the HF pipeline.
VRPNPZVehNodeDataset = HFVRPNPZVehNodeDataset

__all__ = [
    "HFVRPNPZVehNodeDataset",
    "VRPNPZVehNodeDataset",
    "build_bipartite_edge_data_hfvrp",
    "build_hfvrp_strict_slot_targets",
    "infer_assignment_from_routes_with_slots",
    "decode_routes_zero_sep",
    "strip_trailing_zeros",
]
