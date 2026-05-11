"""Competitive customer-neighbourhood builders for READ-style VRP decoders.

This module contains only the PyVRP customer-neighbourhood construction logic.
It is intentionally problem-agnostic:

* CVRP passes no ``group_vec``. The builder uses geometry + route-cover, and
  disables type/tier heat by default because homogeneous vehicle slots are
  permutation-symmetric.
* HFVRP passes ``group_vec=tier_vec``. The builder then adds a small number of
  tier/type heat candidates in addition to geometry + route-cover.

All returned neighbours are 1-based customer ids, matching the decoder-side
convention used before handing the list to PyVRP normalisation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import row_normalize, uncertainty_stats
import numpy as np


@dataclass(frozen=True, slots=True)
class CompetitiveNeighbourDefaults:
    """Default parameters for fixed-budget competitive neighbourhoods."""

    max_neigh: int = 50
    geo_core_k: int = 28
    geo_pool_k: int = 50
    geom_shortlist: int = 96

    # Enabled only when group_vec has at least two distinct active groups.
    group_heat_k: int = 12
    group_heat_k_low: int = 16

    route_cover_top_slots: int = 5
    route_cover_top_slots_low: int = 6
    route_cover_per_slot: int = 2

    symmetric: bool = True
    slot_sim_weight: float = 0.0
    group_sim_weight: float = 0.25
    dist_penalty: float = 1.0

    conf_high_top1: float = 0.85
    conf_high_margin: float = 0.20
    conf_low_top1: float = 0.60
    conf_low_margin: float = 0.08

    global_conf_adapt: bool = True
    global_slot_p1_min: float = 0.70
    global_slot_margin_min: float = 0.10
    global_group_p1_min: float = 0.72
    global_group_margin_min: float = 0.10
    global_low_frac_max: float = 0.35

def masked_slot_prob(prob_slot: np.ndarray, slot_mask: Optional[np.ndarray]) -> np.ndarray:
    prob = row_normalize(prob_slot)
    if slot_mask is None:
        return prob
    mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
    if mask.size != prob.shape[1]:
        raise ValueError("slot_mask length must match prob_slot second dimension")
    out = prob.copy()
    out[:, ~mask] = 0.0
    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12
    if np.any(bad):
        active = np.where(mask)[0]
        if active.size > 0:
            out[bad] = 0.0
            out[np.ix_(bad, active)] = 1.0 / float(active.size)
        else:
            out[bad] = 1.0 / float(max(1, out.shape[1]))
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def build_group_prob_from_slot(
    prob_slot: np.ndarray,
    group_vec: Optional[np.ndarray],
    slot_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], int]:
    """Aggregates slot probabilities into groups/tier probabilities.

    Returns ``(None, 0)`` when no useful grouping is supplied or when all active
    slots are in a single group. This is the default CVRP case.
    """
    if group_vec is None:
        return None, 0

    prob = masked_slot_prob(prob_slot, slot_mask)
    group_vec = np.asarray(group_vec, dtype=np.int64).reshape(-1)
    if group_vec.size != prob.shape[1]:
        raise ValueError("group_vec length must match prob_slot second dimension")

    if slot_mask is None:
        active_slots = np.arange(group_vec.size, dtype=np.int64)
    else:
        mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
        active_slots = np.where(mask)[0].astype(np.int64)

    uniq = sorted(np.unique(group_vec[active_slots]).tolist()) if active_slots.size else []
    if len(uniq) <= 1:
        return None, int(len(uniq))

    group_to_idx = {int(g): j for j, g in enumerate(uniq)}
    out = np.zeros((prob.shape[0], len(uniq)), dtype=np.float32)
    for s in active_slots.tolist():
        out[:, group_to_idx[int(group_vec[int(s)])]] += prob[:, int(s)]
    return row_normalize(out), int(len(uniq))


def nearest_customer_distances(clients_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(clients_xy, dtype=np.float32).reshape(-1, 2)
    diff = xy[:, None, :] - xy[None, :, :]
    dcc = np.sqrt((diff * diff).sum(axis=2) + 1e-12).astype(np.float32)
    np.fill_diagonal(dcc, np.inf)
    return dcc

def _exclusive_source_from_mask(mask: int) -> str:
    # Core geometry is kept separate from extra geometry for diagnostics.
    if int(mask) & 1:
        return "geo_core"
    if int(mask) & 4:
        return "route"
    if int(mask) & 2:
        return "group"
    if int(mask) & 16:
        return "geo_extra"
    return "sym"


def build_competitive_slot_route_neighbours(
    prob_slot: np.ndarray,
    clients_xy: np.ndarray,
    routes0_full: List[List[int]],
    *,
    group_vec: Optional[np.ndarray] = None,
    slot_mask: Optional[np.ndarray] = None,
    defaults: CompetitiveNeighbourDefaults = CompetitiveNeighbourDefaults(),
    enable_group_heat: Optional[bool] = None,
) -> Tuple[List[List[int]], Dict[str, object]]:
    """Builds a fixed-budget competitive PyVRP customer-neighbourhood.

    Each row gets up to ``max_neigh`` candidates. The first ``geo_core_k`` are
    protected geometric nearest-neighbour edges. The remaining budget is filled
    by competition between geometry-extra, group/tier heat, and route-cover
    candidates. Confidence controls source scores, not hard constraints.
    """
    prob_slot = np.asarray(prob_slot, dtype=np.float32)
    clients_xy = np.asarray(clients_xy, dtype=np.float32).reshape(-1, 2)
    n, k = prob_slot.shape
    if clients_xy.shape[0] != n:
        raise ValueError("clients_xy and prob_slot must have the same number of clients")

    if slot_mask is None:
        slot_mask_np = np.ones((k,), dtype=np.bool_)
    else:
        slot_mask_np = np.asarray(slot_mask, dtype=np.bool_).reshape(k)

    max_neigh = defaults.max_neigh
    geo_core_k = defaults.geo_core_k
    geo_pool_k = defaults.geo_pool_k
    geom_shortlist = defaults.geom_shortlist

    group_heat_k_mid = defaults.group_heat_k
    group_heat_k_low = defaults.group_heat_k_low

    route_slots_mid = defaults.route_cover_top_slots
    route_slots_low = defaults.route_cover_top_slots_low
    route_per_slot = defaults.route_cover_per_slot

    symmetric = defaults.symmetric
    slot_sim_weight = defaults.slot_sim_weight
    group_sim_weight = defaults.group_sim_weight
    dist_penalty = defaults.dist_penalty

    high_top1 = defaults.conf_high_top1
    high_margin = defaults.conf_high_margin
    low_top1 = defaults.conf_low_top1
    low_margin = defaults.conf_low_margin

    global_conf_adapt = defaults.global_conf_adapt
    global_slot_p1_min = defaults.global_slot_p1_min
    global_slot_margin_min = defaults.global_slot_margin_min
    global_group_p1_min = defaults.global_group_p1_min
    global_group_margin_min = defaults.global_group_margin_min
    global_low_frac_max = defaults.global_low_frac_max

    max_neigh = max(1, min(int(max_neigh), max(1, n - 1)))
    geo_core_k = max(0, min(int(geo_core_k), max_neigh, max(0, n - 1)))
    geo_pool_k = max(geo_core_k, min(int(geo_pool_k), max(0, n - 1)))
    geom_shortlist = max(geo_pool_k, min(int(geom_shortlist), max(0, n - 1)))

    prob = masked_slot_prob(prob_slot, slot_mask_np)
    p1_slot, _, margin_slot, ent_slot = uncertainty_stats(prob)
    low_slot = (p1_slot <= low_top1) | (margin_slot <= low_margin)

    prob_group, num_groups = build_group_prob_from_slot(prob, group_vec, slot_mask_np)
    if enable_group_heat is None:
        enable_group_heat = prob_group is not None and num_groups >= 2
    if not bool(enable_group_heat):
        prob_group = None

    if prob_group is not None:
        p1_group, _, margin_group, ent_group = uncertainty_stats(prob_group)
        sim_group = (prob_group @ prob_group.T).astype(np.float32)
    else:
        p1_group = p1_slot.copy()
        margin_group = margin_slot.copy()
        ent_group = ent_slot.copy()
        sim_group = None
        group_heat_k_mid = 0
        group_heat_k_low = 0

    mean_p1_slot = float(np.mean(p1_slot)) if p1_slot.size else 0.0
    mean_margin_slot = float(np.mean(margin_slot)) if margin_slot.size else 0.0
    mean_p1_group = float(np.mean(p1_group)) if p1_group.size else 0.0
    mean_margin_group = float(np.mean(margin_group)) if margin_group.size else 0.0
    low_frac = float(np.mean(low_slot)) if low_slot.size else 1.0

    # Local uncertainty: entropy + low margin. This remains continuous and does
    # not freeze any client into a hard confidence bucket.
    u_local = 0.50 * ent_slot + 0.50 * np.clip(1.0 - margin_slot, 0.0, 1.0)
    u_local = np.clip(u_local.astype(np.float32), 0.0, 1.0)

    global_bad = False
    if bool(global_conf_adapt):
        if prob_group is not None:
            global_bad = (
                mean_p1_group < global_group_p1_min
                or mean_margin_group < global_group_margin_min
                or low_frac > global_low_frac_max
            )
        else:
            global_bad = (
                mean_p1_slot < global_slot_p1_min
                or mean_margin_slot < global_slot_margin_min
                or low_frac > global_low_frac_max
            )
    if global_bad:
        # Smooth severity rather than a binary all-or-nothing switch.
        if prob_group is not None:
            p_gap = max(0.0, global_group_p1_min - mean_p1_group) / max(global_group_p1_min, 1e-6)
            m_gap = max(0.0, global_group_margin_min - mean_margin_group) / max(global_group_margin_min, 1e-6)
        else:
            p_gap = max(0.0, global_slot_p1_min - mean_p1_slot) / max(global_slot_p1_min, 1e-6)
            m_gap = max(0.0, global_slot_margin_min - mean_margin_slot) / max(global_slot_margin_min, 1e-6)
        l_gap = max(0.0, low_frac - global_low_frac_max) / max(1.0 - global_low_frac_max, 1e-6)
        u_global = float(np.clip(max(p_gap, m_gap, l_gap, 0.35), 0.0, 1.0))
    else:
        u_global = 0.0

    u_eff = np.maximum(u_local, u_global).astype(np.float32)

    dcc = nearest_customer_distances(clients_xy)
    geo_order = np.argsort(dcc, axis=1).astype(np.int64)
    active_slots = np.where(slot_mask_np)[0].astype(np.int64)

    # bit 1: geo-core, bit 2: group heat, bit 4: route-cover,
    # bit 8: symmetric fill, bit 16: geo-extra.
    cands: List[Dict[int, Tuple[int, float]]] = [dict() for _ in range(n)]
    sim_slot = (prob @ prob.T).astype(np.float32) if slot_sim_weight > 0.0 else None

    def add(i: int, j: int, source: int, score: float) -> None:
        if i == j or j < 0 or j >= n:
            return
        old_mask, old_score = cands[i].get(j, (0, -1e18))
        cands[i][j] = (int(old_mask) | int(source), max(float(old_score), float(score)))

    for i in range(n):
        u = float(u_eff[i])
        geo_extra_weight = 0.30 - 0.10 * u
        group_weight = 0.30 + 0.40 * u
        route_weight = 0.35 + 0.55 * u

        # 1) Protected geometric core.
        core = [int(j) for j in geo_order[i, :geo_core_k].tolist() if int(j) != i and np.isfinite(dcc[i, int(j)])]
        for rank, j in enumerate(core):
            score = 100.0 - float(rank) / max(1, geo_core_k)
            add(i, j, 1, score)

        # 2) Extra geometric candidates compete for remaining slots.
        extra_geo = [int(j) for j in geo_order[i, geo_core_k:geo_pool_k].tolist() if int(j) != i and np.isfinite(dcc[i, int(j)])]
        for offset, j in enumerate(extra_geo):
            rank_norm = float(geo_core_k + offset) / max(1, geo_pool_k)
            score = geo_extra_weight * (1.0 - rank_norm)
            add(i, j, 16, score)

        # 3) Route-cover candidates. Slot probabilities propose seed routes;
        # route geometry picks the nearest clients in those routes.
        route_slots_i = route_slots_low if bool(low_slot[i]) or global_bad else route_slots_mid
        if route_slots_i > 0 and route_per_slot > 0 and active_slots.size > 0:
            slot_order = active_slots[np.argsort(-prob[i, active_slots])]
            used_slots = 0
            for slot in slot_order.tolist():
                if used_slots >= route_slots_i:
                    break
                if slot < 0 or slot >= len(routes0_full):
                    continue
                route = [int(cid) - 1 for cid in routes0_full[int(slot)] if 1 <= int(cid) <= n and int(cid) - 1 != i]
                if not route:
                    continue
                route = sorted(route, key=lambda j: (float(dcc[i, int(j)]), int(j)))
                finite = np.asarray([dcc[i, int(j)] for j in route if np.isfinite(dcc[i, int(j)])], dtype=np.float32)
                scale = float(np.mean(finite)) if finite.size else 1.0
                scale = max(scale, 1e-6)
                added_from_route = 0
                for j in route:
                    dist_score = 1.0 - min(2.0, float(dcc[i, int(j)]) / scale) / 2.0
                    score = route_weight * float(prob[i, int(slot)]) + 0.30 * dist_score
                    add(i, int(j), 4, score)
                    added_from_route += 1
                    if added_from_route >= route_per_slot:
                        break
                if added_from_route > 0:
                    used_slots += 1

        # 4) Group/tier heat candidates. Disabled by default for homogeneous
        # CVRP, enabled for HFVRP when group_vec=tier_vec has >=2 active groups.
        group_k_i = group_heat_k_low if bool(low_slot[i]) or global_bad else group_heat_k_mid
        if prob_group is not None and sim_group is not None and group_k_i > 0:
            short = [
                int(j)
                for j in geo_order[i, :geom_shortlist].tolist()
                if int(j) != i and np.isfinite(dcc[i, int(j)])
            ]
            if short:
                local_dist = dcc[i, short]
                finite = local_dist[np.isfinite(local_dist)]
                scale = float(np.mean(finite)) if finite.size else 1.0
                scale = max(scale, 1e-6)
                scores = group_weight * group_sim_weight * sim_group[i, short] - dist_penalty * (local_dist / scale)
                order = np.argsort(-scores)
                for idx in order[:max(0, int(group_k_i))]:
                    j = int(short[int(idx)])
                    add(i, j, 2, float(scores[int(idx)]))

        # Optional direct slot-sim tie-breaker. Kept disabled by default.
        if sim_slot is not None:
            short = [int(j) for j in geo_order[i, :geom_shortlist].tolist() if int(j) != i and np.isfinite(dcc[i, int(j)])]
            if short:
                vals = slot_sim_weight * sim_slot[i, short]
                for pos in np.argsort(-vals)[:2].tolist():
                    add(i, int(short[int(pos)]), 32, float(vals[int(pos)]))

    if symmetric:
        directed = [(i, j, mask_score[0], mask_score[1]) for i in range(n) for j, mask_score in cands[i].items()]
        for i, j, mask, score in directed:
            if i == j:
                continue
            if i in cands[j]:
                old_mask, old_score = cands[j][i]
                cands[j][i] = (int(old_mask) | 8, max(float(old_score), 0.50 * float(score)))
            else:
                cands[j][i] = (8, 0.50 * float(score))

    neigh: List[List[int]] = []
    final_masks: List[Dict[int, int]] = []
    for i in range(n):
        row_items = [(int(j), int(ms[0]), float(ms[1])) for j, ms in cands[i].items()]

        def key_fn(item):
            j, mask, score = item
            src = _exclusive_source_from_mask(mask)
            # Protected geo-core always stays. The remaining candidates compete
            # by score, with route-cover slightly preferred on ties.
            pri = 0 if src == "geo_core" else 1 if src == "route" else 2 if src == "group" else 3 if src == "geo_extra" else 4
            if src == "geo_core":
                return (0, float(dcc[i, j]), -score, j)
            return (pri, -score, float(dcc[i, j]), j)

        row_items.sort(key=key_fn)
        row_items = row_items[:max_neigh]
        neigh.append([int(j) + 1 for j, _, _ in row_items])
        final_masks.append({int(j): int(mask) for j, mask, _ in row_items})

    sizes = np.asarray([len(row) for row in neigh], dtype=np.float32)
    edge_total = int(sizes.sum())
    src_counts = {"geo_core": 0, "geo_extra": 0, "group": 0, "route": 0, "sym": 0}
    sym_added = 0
    for row in final_masks:
        for mask in row.values():
            src_counts[_exclusive_source_from_mask(mask)] += 1
            if int(mask) & 8:
                sym_added += 1

    denom = max(1, edge_total)
    geo_total = src_counts["geo_core"] + src_counts["geo_extra"]
    profile = {
        "neigh_min": int(sizes.min()) if sizes.size else 0,
        "neigh_mean": float(sizes.mean()) if sizes.size else 0.0,
        "neigh_max": int(sizes.max()) if sizes.size else 0,
        "edge_total": edge_total,
        "geo_edge_ratio": float(geo_total / denom),
        "route_cover_edge_ratio": float(src_counts["route"] / denom),
        "group_heat_edge_ratio": float(src_counts["group"] / denom),
        "sym_added_edge_ratio": float(sym_added / denom),
        "heat_global_unreliable": bool(global_bad),
        "heat_num_groups": int(num_groups),
        "heat_group_enabled": bool(prob_group is not None),
    }
    return neigh, profile

__all__ = [
    "CompetitiveNeighbourDefaults",
    "row_normalize",
    "uncertainty_stats",
    "masked_slot_prob",
    "build_group_prob_from_slot",
    "nearest_customer_distances",
    "build_competitive_slot_route_neighbours",
]
