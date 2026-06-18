"""Shared Stage-A seed construction for READ slot decoders."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .read_config import READDecodeCfg
from .common import row_normalize


def active_slot_mask(k: int, slot_mask: Optional[np.ndarray]) -> np.ndarray:
    """Return the active slot mask, defaulting to all slots active."""
    if slot_mask is None:
        return np.ones((int(k),), dtype=np.bool_)
    mask = np.asarray(slot_mask, dtype=np.bool_).reshape(int(k))
    if not np.any(mask):
        raise ValueError("slot_mask must contain at least one active slot.")
    return mask


def assert_instance_feasible(
    demand: np.ndarray,
    capacity_vec: np.ndarray,
    slot_mask: np.ndarray,
    *,
    tol: float = 1e-6,
) -> None:
    """Reject instances whose demand cannot fit in the active fleet."""
    demand = np.asarray(demand, dtype=np.float32).reshape(-1)
    capacity_vec = np.asarray(capacity_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    active_cap = capacity_vec[slot_mask]
    if active_cap.size == 0:
        raise ValueError("READ instance has no active vehicles.")
    if demand.size == 0:
        return
    if float(demand.sum()) > float(active_cap.sum()) + float(tol):
        raise ValueError("READ instance is infeasible: total demand exceeds active fleet capacity.")
    if float(demand.max()) > float(active_cap.max()) + float(tol):
        raise ValueError("READ instance is infeasible: one customer exceeds every active vehicle capacity.")


def assert_labels_feasible(
    labels: np.ndarray,
    demand: np.ndarray,
    capacity_vec: np.ndarray,
    slot_mask: np.ndarray,
    *,
    context: str,
    tol: float = 1e-6,
) -> None:
    """Check that Stage-A labels use active slots and satisfy capacity."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    demand = np.asarray(demand, dtype=np.float32).reshape(-1)
    capacity_vec = np.asarray(capacity_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n = int(demand.size)
    k = int(capacity_vec.size)
    if labels.shape != (n,):
        raise RuntimeError(f"{context} produced {labels.size} labels for {n} customers.")
    if np.any(labels < 0) or np.any(labels >= k):
        bad = int(np.where((labels < 0) | (labels >= k))[0][0])
        raise RuntimeError(f"{context} assigned customer {bad} to out-of-range slot {int(labels[bad])}.")
    if np.any(~slot_mask[labels]):
        bad = int(np.where(~slot_mask[labels])[0][0])
        raise RuntimeError(f"{context} assigned customer {bad} to inactive slot {int(labels[bad])}.")

    loads = np.zeros((k,), dtype=np.float32)
    np.add.at(loads, labels, demand)
    overflow = np.maximum(loads - capacity_vec, 0.0)
    overflow[~slot_mask] = 0.0
    if float(overflow.sum()) > float(tol):
        slot = int(np.argmax(overflow))
        raise RuntimeError(
            f"{context} exceeded capacity on slot {slot}: "
            f"load={float(loads[slot]):.6f}, capacity={float(capacity_vec[slot]):.6f}."
        )


def _candidate_slots(prob: np.ndarray, slot_mask: np.ndarray, top_slot_k: int) -> List[np.ndarray]:
    active = np.where(slot_mask)[0].astype(np.int64)
    top_slot_k = max(1, min(int(top_slot_k), int(active.size)))

    candidates: List[np.ndarray] = []
    for i in range(prob.shape[0]):
        order = np.argsort(-prob[i])
        order = order[slot_mask[order]]
        candidates.append(order[:top_slot_k].astype(np.int64))
    return candidates


def _insertion_delta(
    route: List[int],
    client_id: int,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
) -> Tuple[float, int]:
    xy = clients_xy[int(client_id) - 1]

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    if not route:
        return float(2.0 * dist(depot_xy, xy)), 0

    best_delta = float("inf")
    best_pos = 0

    first_xy = clients_xy[int(route[0]) - 1]
    delta = dist(depot_xy, xy) + dist(xy, first_xy) - dist(depot_xy, first_xy)
    if delta < best_delta:
        best_delta = delta
        best_pos = 0

    for pos in range(1, len(route)):
        prev_xy = clients_xy[int(route[pos - 1]) - 1]
        next_xy = clients_xy[int(route[pos]) - 1]
        delta = dist(prev_xy, xy) + dist(xy, next_xy) - dist(prev_xy, next_xy)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    last_xy = clients_xy[int(route[-1]) - 1]
    delta = dist(last_xy, xy) + dist(xy, depot_xy) - dist(last_xy, depot_xy)
    if delta < best_delta:
        best_delta = delta
        best_pos = len(route)

    return float(best_delta), int(best_pos)


def _capacity_reserve_violation(
    remaining_demand: np.ndarray,
    residual_capacity: np.ndarray,
    *,
    tol: float = 1e-6,
) -> float:
    remaining = np.asarray(remaining_demand, dtype=np.float32).reshape(-1)
    residual = np.asarray(residual_capacity, dtype=np.float32).reshape(-1)

    remaining = remaining[remaining > float(tol)]
    residual = residual[residual > float(tol)]

    if remaining.size == 0:
        return 0.0
    if residual.size == 0:
        return float(remaining.sum())

    violation = 0.0
    violation += max(0.0, float(remaining.sum() - residual.sum()))
    violation += max(0.0, float(remaining.max() - residual.max()))

    for demand_threshold in np.unique(remaining):
        threshold = float(demand_threshold) - float(tol)
        need = float(remaining[remaining >= threshold].sum())
        available = float(residual[residual >= threshold].sum())
        violation += max(0.0, need - available)

    return float(violation)


def construct_slot_seed_labels(
    prob: np.ndarray,
    demand: np.ndarray,
    capacity_vec: np.ndarray,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    cfg: READDecodeCfg,
    *,
    fixed_cost_vec: Optional[np.ndarray] = None,
    unit_cost_vec: Optional[np.ndarray] = None,
    slot_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct capacity-feasible slot labels from a client-to-slot posterior."""
    prob = row_normalize(np.asarray(prob, dtype=np.float32))
    demand = np.asarray(demand, dtype=np.float32).reshape(-1)
    capacity_vec = np.asarray(capacity_vec, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)

    n, k = prob.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if demand.shape != (n,):
        raise ValueError(f"demand.shape={demand.shape}, expected {(n,)}")
    if capacity_vec.shape != (k,):
        raise ValueError(f"capacity_vec.shape={capacity_vec.shape}, expected {(k,)}")
    if clients_xy.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_xy.shape}, expected {(n, 2)}")

    fixed_cost = (
        np.zeros((k,), dtype=np.float32)
        if fixed_cost_vec is None
        else np.asarray(fixed_cost_vec, dtype=np.float32).reshape(k)
    )
    unit_cost = (
        np.ones((k,), dtype=np.float32)
        if unit_cost_vec is None
        else np.asarray(unit_cost_vec, dtype=np.float32).reshape(k)
    )
    mask = active_slot_mask(k, slot_mask)
    assert_instance_feasible(demand, capacity_vec, mask)

    prob = prob.copy()
    prob[:, ~mask] = 0.0
    prob = row_normalize(prob)

    active = np.where(mask)[0].astype(np.int64)
    candidates = _candidate_slots(prob, mask, cfg.top_slot_k)

    prob_active = prob[:, active]
    sorted_prob = np.sort(prob_active, axis=1)[:, ::-1]
    margin = (
        sorted_prob[:, 0] - sorted_prob[:, 1]
        if sorted_prob.shape[1] >= 2
        else np.ones((n,), dtype=np.float32)
    )
    max_prob = sorted_prob[:, 0]

    primary_order = sorted(
        range(n),
        key=lambda i: (-float(demand[i]), -float(margin[i]), float(max_prob[i]), int(i)),
    )

    feasible_count = (
        capacity_vec[active][None, :] + 1e-6 >= demand[:, None]
    ).sum(axis=1)
    constrained_order = sorted(
        range(n),
        key=lambda i: (int(feasible_count[i]), -float(demand[i]), -float(margin[i]), int(i)),
    )
    uncertain_order = sorted(
        range(n),
        key=lambda i: (float(max_prob[i]), -float(demand[i]), int(feasible_count[i]), int(i)),
    )

    def build_attempt(
        order: List[int],
        *,
        reserve_aware: bool,
        stop_after_preferred: bool,
        best_fit_only: bool,
    ) -> Optional[np.ndarray]:
        labels = np.full((n,), -1, dtype=np.int64)
        loads = np.zeros((k,), dtype=np.float32)
        routes: List[List[int]] = [[] for _ in range(k)]

        for pos, i in enumerate(order):
            i = int(i)
            demand_i = float(demand[i])

            if best_fit_only:
                pools = [(0, [int(s) for s in active.tolist()])]
            else:
                preferred = list(map(int, candidates[i].tolist()))
                fallback = [int(s) for s in active.tolist() if int(s) not in preferred]
                pools = [(0, preferred), (1, fallback)]

            remaining_idx = np.asarray(order[pos + 1 :], dtype=np.int64)
            best = None
            for pool_id, pool in pools:
                for slot in pool:
                    if loads[slot] + demand_i > capacity_vec[slot] + 1e-6:
                        continue

                    residual = capacity_vec.copy() - loads
                    residual[int(slot)] -= demand_i
                    residual[~mask] = 0.0
                    reserve = (
                        _capacity_reserve_violation(demand[remaining_idx], residual[active])
                        if reserve_aware
                        else 0.0
                    )

                    if best_fit_only:
                        residual_after = float(capacity_vec[slot] - loads[slot] - demand_i)
                        key = (float(reserve), residual_after, -float(prob[i, slot]), int(slot), 0)
                    else:
                        ins_delta, ins_pos = _insertion_delta(routes[slot], i + 1, depot_xy, clients_xy)
                        nll = -np.log(max(float(prob[i, slot]), 1e-12))
                        open_cost = float(fixed_cost[slot]) if not routes[slot] else 0.0
                        fill = (loads[slot] + demand_i) / max(float(capacity_vec[slot]), 1e-6)
                        score = (
                            float(cfg.lambda_prob) * float(nll)
                            + float(cfg.lambda_insert) * float(unit_cost[slot]) * float(ins_delta)
                            + float(cfg.lambda_open) * float(open_cost)
                            + float(cfg.lambda_fill) * float(fill * fill)
                            + 10.0 * float(pool_id)
                        )
                        key = (
                            float(reserve),
                            float(score),
                            int(pool_id),
                            -float(prob[i, slot]),
                            int(slot),
                            int(ins_pos),
                        )

                    if best is None or key < best:
                        best = key

                if stop_after_preferred and best is not None:
                    break

            if best is None:
                return None

            if best_fit_only:
                _, _, _, best_slot, _ = best
                best_pos = len(routes[int(best_slot)])
            else:
                _, _, _, _, best_slot, best_pos = best

            labels[i] = int(best_slot)
            loads[int(best_slot)] += demand_i
            routes[int(best_slot)].insert(int(best_pos), int(i + 1))

        return labels.astype(np.int64)

    attempts = [
        (primary_order, False, True, False),
        (constrained_order, True, False, False),
        (uncertain_order, True, False, False),
        (constrained_order, True, False, True),
    ]
    for order, reserve_aware, stop_after_preferred, best_fit_only in attempts:
        labels = build_attempt(
            order,
            reserve_aware=reserve_aware,
            stop_after_preferred=stop_after_preferred,
            best_fit_only=best_fit_only,
        )
        if labels is None:
            continue
        assert_labels_feasible(labels, demand, capacity_vec, mask, context="READ Stage-A seed construction")
        return labels.astype(np.int64)

    raise RuntimeError(
        "Stage-A seed construction failed after all deterministic packing attempts. "
        "The instance passed only necessary capacity checks; the slot-level packing may still be infeasible."
    )


__all__ = [
    "active_slot_mask",
    "assert_instance_feasible",
    "assert_labels_feasible",
    "construct_slot_seed_labels",
]
