"""Competitive customer-neighbourhood builders for READ-style VRP decoders.

This module contains the PyVRP customer-neighbourhood construction logic.
All returned neighbours are 1-based customer ids, matching PyVRP conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from .common import row_normalize
import numpy as np


@dataclass(frozen=True, slots=True)
class HeatmapNeighbourDefaults:
    num_neighbours: int = 50

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


def nearest_customer_distances(clients_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(clients_xy, dtype=np.float32).reshape(-1, 2)
    diff = xy[:, None, :] - xy[None, :, :]
    dcc = np.sqrt((diff * diff).sum(axis=2) + 1e-12).astype(np.float32)
    np.fill_diagonal(dcc, np.inf)
    return dcc


def build_heatmap_neighbours(
    prob_slot: np.ndarray,
    clients_xy: np.ndarray,
    *,
    dem: Optional[np.ndarray] = None,
    cap_vec: Optional[np.ndarray] = None,
    slot_mask: Optional[np.ndarray] = None,
    defaults: HeatmapNeighbourDefaults = HeatmapNeighbourDefaults(),
    tol: float = 1e-6,
) -> List[List[int]]:
    """Build a heatmap-geometry PyVRP neighbourhood.

    Candidate pool:
        1. posterior co-assignment neighbours from the slot-customer heatmap;
        2. geometric nearest neighbours.

    The heatmap controls candidate inclusion, while geometry controls the final
    neighbour order. The only protocol parameter is the total neighbour budget B.
    """
    prob = masked_slot_prob(prob_slot, slot_mask)
    clients_xy = np.asarray(clients_xy, dtype=np.float32).reshape(-1, 2)

    n, k = prob.shape
    if clients_xy.shape[0] != n:
        raise ValueError("clients_xy and prob_slot must have the same number of clients")

    budget = max(1, min(int(defaults.num_neighbours), max(1, n - 1)))
    heat_budget = int(np.ceil(float(budget) / 2.0))
    geo_budget = budget - heat_budget

    if slot_mask is None:
        active = np.arange(k, dtype=np.int64)
    else:
        mask = np.asarray(slot_mask, dtype=np.bool_).reshape(k)
        active = np.where(mask)[0].astype(np.int64)

    if active.size == 0:
        raise ValueError("slot_mask must contain at least one active slot")

    dcc = nearest_customer_distances(clients_xy)
    ids = np.arange(n, dtype=np.int64)

    # Posterior co-assignment score:
    #   A_ij = sum_s P_is P_js
    # Optionally capacity-aware for HFVRP:
    #   A_ij = sum_s P_is P_js 1[q_i + q_j <= Q_s].
    if dem is not None and cap_vec is not None:
        dem_np = np.asarray(dem, dtype=np.float32).reshape(n)
        cap_np = np.asarray(cap_vec, dtype=np.float32).reshape(k)

        pair_load = dem_np[:, None] + dem_np[None, :]
        heat_score = np.zeros((n, n), dtype=np.float32)

        for s in active.tolist():
            feasible_pair = pair_load <= float(cap_np[int(s)]) + float(tol)
            heat_score += (
                np.outer(prob[:, int(s)], prob[:, int(s)]).astype(np.float32)
                * feasible_pair.astype(np.float32)
            )
    else:
        heat_score = (prob[:, active] @ prob[:, active].T).astype(np.float32)

    np.fill_diagonal(heat_score, -np.inf)

    geo_order_all = np.argsort(dcc, axis=1)

    neigh: List[List[int]] = []

    for i in range(n):
        heat_order = np.lexsort((ids, dcc[i], -heat_score[i]))

        heat_set: List[int] = []
        for j in heat_order.tolist():
            j = int(j)
            score_ij = float(heat_score[i, j])
            if j == i or not np.isfinite(score_ij) or score_ij <= 0.0:
                continue
            heat_set.append(j)
            if len(heat_set) >= heat_budget:
                break

        geo_order = geo_order_all[i]

        geo_set: List[int] = []
        for j in geo_order.tolist():
            j = int(j)
            if j == i or not np.isfinite(dcc[i, j]):
                continue
            geo_set.append(j)
            if len(geo_set) >= geo_budget:
                break

        heat_seen = set(heat_set)
        geo_seen = set(geo_set)
        cand = list(heat_seen | geo_seen)

        # If heat/geo overlap or weak heat evidence leaves fewer than B
        # candidates, fill the rest by pure geometry.
        if len(cand) < budget:
            seen = set(cand)
            for j in geo_order.tolist():
                j = int(j)
                if j == i or j in seen or not np.isfinite(dcc[i, j]):
                    continue
                seen.add(j)
                cand.append(j)
                if len(cand) >= budget:
                    break

        final = sorted(cand, key=lambda j: (float(dcc[i, int(j)]), int(j)))
        final = final[:budget]

        neigh.append([int(j) + 1 for j in final])

    return neigh

__all__ = [
    "HeatmapNeighbourDefaults",
    "row_normalize",
    "masked_slot_prob",
    "nearest_customer_distances",
    "build_heatmap_neighbours",
]
