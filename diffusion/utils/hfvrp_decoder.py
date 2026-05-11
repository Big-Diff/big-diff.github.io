"""READ decoder for the Heterogeneous Fleet Vehicle Routing Problem (HFVRP)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch

from .read_config import READDecodeCfg, READDecodeResult, HFVRP_READ_PRESET
from .read_competitive_neighbours import build_competitive_slot_route_neighbours
from .common import (
    StageTimer,
    build_seed_routes,
    nll_from_prob_and_labels,
    row_normalize,
)
from .pyvrp_hfvrp_common import (
    HFPyVRPBuildConfig,
    HFPyVRPILSConfig,
    HFPyVRPNeighbourConfig,
    exact_hf_cost,
    infer_tier_vec,
    labels_from_slot_routes,
    refine_hf_with_component_ils,
)
# ---------------------------------------------------------------------------
# Cost, tiers, and feasibility
# ---------------------------------------------------------------------------

def _check_hf_labels_feasible(
    labels: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    slot_mask: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, Dict[str, object]]:
    """Validate that a label assignment uses active slots and respects capacity."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n = int(dem.size)
    k = int(cap_vec.size)
    if labels.shape != (n,):
        return False, {"reason": "label_size_mismatch", "label_size": int(labels.size), "num_clients": n}
    if slot_mask.shape != (k,):
        return False, {"reason": "slot_mask_size_mismatch", "slot_mask_size": int(slot_mask.size), "num_slots": k}
    if np.any(labels < 0) or np.any(labels >= k):
        bad = int(np.where((labels < 0) | (labels >= k))[0][0])
        return False, {"reason": "label_out_of_range", "client": bad, "label": int(labels[bad]), "num_slots": k}
    inactive = np.where(~slot_mask[labels])[0]
    if inactive.size:
        i = int(inactive[0])
        return False, {"reason": "inactive_slot_used", "client": i, "slot": int(labels[i])}

    loads = np.zeros((k,), dtype=np.float32)
    np.add.at(loads, labels, dem)
    overflow = np.maximum(loads - cap_vec, 0.0)
    overflow[~slot_mask] = 0.0
    if float(overflow.sum()) > float(tol):
        slot = int(np.argmax(overflow))
        return False, {
            "reason": "capacity_overflow",
            "slot": slot,
            "load": float(loads[slot]),
            "capacity": float(cap_vec[slot]),
            "total_overflow": float(overflow.sum()),
        }

    return True, {
        "reason": "ok",
        "used_routes": int(np.count_nonzero(loads > 0.0)),
        "max_load": float(loads.max()) if loads.size else 0.0,
    }


def _check_hf_routes_feasible(
    routes_by_slot: List[List[int]],
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    num_clients: int,
    slot_mask: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, Dict[str, object]]:
    """Validate route coverage, active-slot use, and slot-specific capacities."""
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n = int(num_clients)
    k = int(cap_vec.size)
    if dem.size != n:
        return False, {"reason": "demand_size_mismatch", "num_clients": n, "dem_size": int(dem.size)}
    if len(routes_by_slot) != k:
        return False, {"reason": "route_slot_count_mismatch", "routes": len(routes_by_slot), "num_slots": k}
    if slot_mask.shape != (k,):
        return False, {"reason": "slot_mask_size_mismatch", "slot_mask_size": int(slot_mask.size), "num_slots": k}

    seen = np.zeros((n,), dtype=np.int32)
    used_routes = 0
    max_load = 0.0

    for slot, route in enumerate(routes_by_slot):
        if not route:
            continue
        if not bool(slot_mask[slot]):
            return False, {"reason": "inactive_route_nonempty", "slot": int(slot), "route_size": len(route)}
        used_routes += 1
        load = 0.0
        for cid in route:
            cid = int(cid)
            if cid < 1 or cid > n:
                return False, {"reason": "client_id_out_of_range", "slot": int(slot), "client_id": cid}
            seen[cid - 1] += 1
            if seen[cid - 1] > 1:
                return False, {"reason": "duplicate_client", "slot": int(slot), "client_id": cid}
            load += float(dem[cid - 1])
        max_load = max(max_load, load)
        if load > float(cap_vec[slot]) + float(tol):
            return False, {
                "reason": "capacity_overflow",
                "slot": int(slot),
                "load": float(load),
                "capacity": float(cap_vec[slot]),
            }

    missing = np.where(seen == 0)[0]
    if missing.size:
        return False, {
            "reason": "missing_clients",
            "missing_count": int(missing.size),
            "first_missing_client": int(missing[0] + 1),
        }

    return True, {"reason": "ok", "used_routes": int(used_routes), "max_route_load": float(max_load)}


def _require_hf_labels_feasible(
    labels: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    slot_mask: np.ndarray,
    context: str,
) -> Dict[str, object]:
    ok, info = _check_hf_labels_feasible(labels, dem, cap_vec, slot_mask=slot_mask)
    if not ok:
        raise ValueError(f"{context} infeasible: {info}")
    return info


def _require_hf_routes_feasible(
    routes_by_slot: List[List[int]],
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    num_clients: int,
    slot_mask: np.ndarray,
    context: str,
) -> Dict[str, object]:
    ok, info = _check_hf_routes_feasible(
        routes_by_slot,
        dem,
        cap_vec,
        num_clients=num_clients,
        slot_mask=slot_mask,
    )
    if not ok:
        raise ValueError(f"{context} infeasible: {info}")
    return info

# ---------------------------------------------------------------------------
# Stage A: heterogeneous capacity/economics projection
# ---------------------------------------------------------------------------

def _hf_build_full_cost(
    prob: np.ndarray,
    fixed_vec: np.ndarray,
    unit_cost_vec: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    *,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Approximate assignment cost aligned with the HFVRP objective."""
    prob = row_normalize(prob)
    n, k = prob.shape
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32).reshape(n, 2)
    fixed_vec = np.asarray(fixed_vec, dtype=np.float32).reshape(k)
    unit_cost_vec = np.asarray(unit_cost_vec, dtype=np.float32).reshape(k)
    dem = np.asarray(dem, dtype=np.float32).reshape(n)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(k)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(k)

    hp = HFVRP_READ_PRESET.hf_projector
    if hp is None:
        raise RuntimeError("HFVRP_READ_PRESET.hf_projector must be defined.")

    radial = np.linalg.norm(clients_xy - depot_xy[None, :], axis=1).astype(np.float32)
    radial_norm = radial / max(1e-6, float(radial.mean())) if n else radial
    unit_norm = unit_cost_vec / max(1e-6, float(np.mean(unit_cost_vec))) if k else unit_cost_vec
    fixed_norm = fixed_vec / max(1e-6, float(np.mean(np.abs(fixed_vec))) if k else 1.0)
    demand_ratio = dem[:, None] / np.maximum(cap_vec[None, :], 1e-6)
    nll = -np.log(np.clip(prob, 1e-12, None)).astype(np.float32)

    cost = (
        float(hp.lam_prob) * nll
        + float(hp.lam_econ) * (radial_norm[:, None] * unit_norm[None, :])
        + float(hp.lam_fixed) * (demand_ratio * fixed_norm[None, :])
    ).astype(np.float32)
    cost[:, ~slot_mask] = 1e18
    return cost


def _hf_candidate_sets_topk(
    prob: np.ndarray,
    *,
    topk: int,
    cum_prob: float,
    slot_mask: np.ndarray,
) -> List[np.ndarray]:
    """Build active-slot top-k/cumulative-probability candidate sets."""
    prob = row_normalize(prob)
    n, k = prob.shape
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(k)
    active = np.where(slot_mask)[0].astype(np.int64)
    if active.size == 0:
        raise ValueError("slot_mask must contain at least one active slot.")

    sets: List[np.ndarray] = []
    for i in range(n):
        row = prob[i].copy()
        row[~slot_mask] = 0.0
        mass = float(row.sum())
        if mass <= 1e-12:
            row[active] = 1.0 / float(active.size)
        else:
            row /= mass

        order = np.argsort(-row)
        p_sorted = row[order]
        csum = np.cumsum(p_sorted)
        keep = int(np.searchsorted(csum, float(cum_prob), side="left") + 1)
        keep = max(1, min(int(topk), keep, k))
        cand = order[:keep].astype(np.int64)
        cand = cand[slot_mask[cand]]
        if cand.size == 0:
            cand = active.copy()
        sets.append(cand.astype(np.int64))
    return sets


def _hf_assign_with_duals(
    cand_cost: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    mu: np.ndarray,
    prev_load: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each client to the minimum reduced-cost candidate slot."""
    n, k = cand_cost.shape
    dem = np.asarray(dem, dtype=np.float32).reshape(n)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(k)
    mu = np.asarray(mu, dtype=np.float32).reshape(k)
    prev_load = np.asarray(prev_load, dtype=np.float32).reshape(k)

    hp = HFVRP_READ_PRESET.hf_projector
    if hp is None:
        raise RuntimeError("HFVRP_READ_PRESET.hf_projector must be defined.")

    fill_ratio = prev_load / np.maximum(cap_vec, 1e-6)
    fill_bias = float(hp.lam_load) * np.clip(fill_ratio, 0.0, 10.0)
    reduced = cand_cost + dem[:, None] * mu[None, :] + fill_bias[None, :]
    labels = np.argmin(reduced, axis=1).astype(np.int64)

    load = np.zeros((k,), dtype=np.float32)
    np.add.at(load, labels, dem)
    return labels, load


def _hf_repair_assignment(
    labels: np.ndarray,
    full_cost: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    slot_mask: np.ndarray,
    max_rounds: int,
) -> np.ndarray:
    """Repair overloaded HF slots by capacity-safe moves first, then
    overflow-reducing penalty moves.

    This is still a local repair: it improves many overloaded assignments but
    does not mathematically guarantee feasibility. The caller should still keep
    a final global fallback.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    full_cost = np.asarray(full_cost, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n, k = full_cost.shape
    if labels.shape != (n,):
        raise ValueError(f"labels.shape={labels.shape}, expected {(n,)}")
    if cap_vec.shape != (k,) or slot_mask.shape != (k,):
        raise ValueError("cap_vec and slot_mask must have shape (K,).")

    loads = np.zeros((k,), dtype=np.float32)
    np.add.at(loads, labels, dem)
    rank = np.argsort(full_cost, axis=1).astype(np.int64)

    for _ in range(int(max_rounds)):
        overflow = np.maximum(loads - cap_vec, 0.0)
        overflow[~slot_mask] = 0.0
        total_over = float(overflow.sum())

        if total_over <= 1e-6:
            break

        overloaded = np.where(overflow > 1e-6)[0]
        overloaded = overloaded[np.argsort(-overflow[overloaded])]

        # ------------------------------------------------------------------
        # Phase 1: try strictly feasible moves.
        # Move one client from an overloaded slot to a slot that remains feasible.
        # ------------------------------------------------------------------
        best = None

        for old_slot in overloaded.tolist():
            custs = np.where(labels == int(old_slot))[0]

            # Larger demand first helps remove overflow faster.
            custs = custs[np.argsort(-dem[custs])]

            for i in custs.tolist():
                old_cost = float(full_cost[i, int(old_slot)])

                for new_slot in rank[i].tolist():
                    new_slot = int(new_slot)
                    if new_slot == int(old_slot) or not bool(slot_mask[new_slot]):
                        continue

                    if loads[new_slot] + dem[i] > cap_vec[new_slot] + 1e-6:
                        continue

                    delta = float(full_cost[i, new_slot] - old_cost)
                    # Prefer low cost increase, then larger demand.
                    cand = (delta, -float(dem[i]), int(i), int(old_slot), int(new_slot))

                    if best is None or cand < best:
                        best = cand

                    # Since rank[i] is sorted by cost, first feasible target
                    # is enough for this client.
                    break

        if best is not None:
            _, _, i, old_slot, new_slot = best
            loads[old_slot] -= dem[i]
            loads[new_slot] += dem[i]
            labels[i] = int(new_slot)
            continue

        # ------------------------------------------------------------------
        # Phase 2: allow a penalty move only if it reduces total overflow.
        # This is stronger than blindly moving to the cheapest penalised slot.
        # ------------------------------------------------------------------
        best = None

        for old_slot in overloaded.tolist():
            custs = np.where(labels == int(old_slot))[0]
            custs = custs[np.argsort(-dem[custs])]

            for i in custs.tolist():
                old_slot = int(old_slot)
                old_cost = float(full_cost[i, old_slot])

                for new_slot in rank[i].tolist():
                    new_slot = int(new_slot)
                    if new_slot == old_slot or not bool(slot_mask[new_slot]):
                        continue

                    before_pair_over = (
                        max(0.0, float(loads[old_slot] - cap_vec[old_slot]))
                        + max(0.0, float(loads[new_slot] - cap_vec[new_slot]))
                    )
                    after_pair_over = (
                        max(0.0, float(loads[old_slot] - dem[i] - cap_vec[old_slot]))
                        + max(0.0, float(loads[new_slot] + dem[i] - cap_vec[new_slot]))
                    )

                    improve = before_pair_over - after_pair_over
                    if improve <= 1e-9:
                        continue

                    delta = float(full_cost[i, new_slot] - old_cost)
                    score = delta + 1000.0 * after_pair_over

                    # Prefer larger overflow reduction, then lower penalised cost.
                    cand = (-float(improve), float(score), -float(dem[i]), int(i), old_slot, new_slot)

                    if best is None or cand < best:
                        best = cand

                # 注意：这里不要 break。penalty 阶段要检查所有 active target。

        if best is None:
            break

        _, _, _, i, old_slot, new_slot = best
        loads[old_slot] -= dem[i]
        loads[new_slot] += dem[i]
        labels[i] = int(new_slot)

    return labels.astype(np.int64)

def _hf_greedy_feasible_assignment(
    full_cost: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    *,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Build a capacity-feasible assignment by global greedy insertion.

    This fallback is used only when dual assignment + local repair still leaves
    overload. It prioritises feasibility while still using full_cost as a
    deterministic tie-breaker.
    """
    full_cost = np.asarray(full_cost, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n, k = full_cost.shape
    active = np.where(slot_mask)[0].astype(np.int64)

    if active.size == 0:
        raise ValueError("slot_mask must contain at least one active slot.")

    if float(dem.sum()) > float(cap_vec[active].sum()) + 1e-6:
        raise ValueError(
            "HFVRP instance is capacity-infeasible: total demand exceeds active fleet capacity."
        )

    max_cap = float(cap_vec[active].max())
    if np.any(dem > max_cap + 1e-6):
        i = int(np.argmax(dem))
        raise ValueError(
            f"HFVRP instance is capacity-infeasible: client {i} demand={float(dem[i])} "
            f"exceeds max active capacity={max_cap}."
        )

    active_cost = full_cost[:, active]
    order_active = np.argsort(active_cost, axis=1)

    best_cost = active_cost[np.arange(n), order_active[:, 0]]
    if active.size >= 2:
        second_cost = active_cost[np.arange(n), order_active[:, 1]]
    else:
        second_cost = best_cost
    regret = second_cost - best_cost

    orders = [
        # Large customers first, then high-regret customers.
        sorted(range(n), key=lambda i: (-float(dem[i]), -float(regret[i]), float(best_cost[i]), int(i))),
        # High-regret customers first; useful when a customer strongly prefers one tier.
        sorted(range(n), key=lambda i: (-float(regret[i]), -float(dem[i]), float(best_cost[i]), int(i))),
    ]

    def try_build(order, mode: str) -> Optional[np.ndarray]:
        labels = np.full((n,), -1, dtype=np.int64)
        loads = np.zeros((k,), dtype=np.float32)

        for i in order:
            feasible = [
                int(s)
                for s in active.tolist()
                if loads[int(s)] + dem[i] <= cap_vec[int(s)] + 1e-6
            ]

            if not feasible:
                return None

            if mode == "best_fit":
                # Prefer tighter remaining capacity to avoid fragmentation.
                def score(s: int):
                    rem_after = float(cap_vec[s] - loads[s] - dem[i])
                    return (rem_after, float(full_cost[i, s]), int(s))
            else:
                # Prefer model/objective proxy cost.
                def score(s: int):
                    fill_after = float((loads[s] + dem[i]) / max(float(cap_vec[s]), 1e-6))
                    return (float(full_cost[i, s]), fill_after, int(s))

            s_best = min(feasible, key=score)
            labels[i] = int(s_best)
            loads[s_best] += dem[i]

        return labels.astype(np.int64)

    for order in orders:
        for mode in ("cost", "best_fit"):
            labels = try_build(order, mode)
            if labels is not None:
                return labels

    raise ValueError("HF greedy fallback failed to construct a feasible assignment.")


def stage_a_capacity_project_hf(
    prob: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    fixed_vec: np.ndarray,
    unit_cost_vec: np.ndarray,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    *,
    slot_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project slot posteriors to a capacity-feasible HFVRP assignment."""
    prob = row_normalize(prob)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    fixed_vec = np.asarray(fixed_vec, dtype=np.float32).reshape(-1)
    unit_cost_vec = np.asarray(unit_cost_vec, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)

    n, k = prob.shape
    if n <= 0 or k <= 0:
        raise ValueError(f"prob must have positive shape, got {prob.shape}")
    if dem.shape != (n,):
        raise ValueError(f"dem.shape={dem.shape}, expected {(n,)}")
    if cap_vec.shape != (k,) or fixed_vec.shape != (k,) or unit_cost_vec.shape != (k,):
        raise ValueError("cap_vec, fixed_vec, and unit_cost_vec must all have shape (K,).")
    if clients_xy.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_xy.shape}, expected {(n, 2)}")

    if slot_mask is None:
        slot_mask_np = np.ones((k,), dtype=np.bool_)
    else:
        slot_mask_np = np.asarray(slot_mask, dtype=np.bool_).reshape(k)
    active_slots = np.where(slot_mask_np)[0].astype(np.int64)
    if active_slots.size == 0:
        raise ValueError("slot_mask must contain at least one active slot.")
    if float(dem.sum()) > float(cap_vec[slot_mask_np].sum()) + 1e-6:
        raise ValueError("HFVRP instance is capacity-infeasible: total demand exceeds active fleet capacity.")

    full_cost = _hf_build_full_cost(
        prob=prob,
        fixed_vec=fixed_vec,
        unit_cost_vec=unit_cost_vec,
        dem=dem,
        cap_vec=cap_vec,
        depot_xy=depot_xy,
        clients_xy=clients_xy,
        slot_mask=slot_mask_np,
    )

    projection = HFVRP_READ_PRESET.projection
    cand_sets = _hf_candidate_sets_topk(
        prob=prob,
        topk=int(projection.topk),
        cum_prob=float(projection.cum_prob),
        slot_mask=slot_mask_np,
    )

    cand_cost = np.full_like(full_cost, 1e18, dtype=np.float32)
    for i, cand in enumerate(cand_sets):
        cand_cost[i, cand] = full_cost[i, cand]

    hp = HFVRP_READ_PRESET.hf_projector
    if hp is None:
        raise RuntimeError("HFVRP_READ_PRESET.hf_projector must be defined.")

    mu = np.zeros((k,), dtype=np.float32)
    prev_load = np.zeros((k,), dtype=np.float32)
    best_labels: Optional[np.ndarray] = None
    best_violation = float("inf")
    best_obj = float("inf")
    cap_scale = max(1e-6, float(np.mean(cap_vec[active_slots])))

    for t in range(int(hp.dual_iters)):
        labels_t, load_t = _hf_assign_with_duals(cand_cost, dem, cap_vec, mu, prev_load)
        overload = np.maximum(load_t - cap_vec, 0.0)
        overload[~slot_mask_np] = 0.0
        violation = float(overload.sum())
        obj = float(full_cost[np.arange(n), labels_t].sum())

        if (
            best_labels is None
            or violation < best_violation - 1e-9
            or (abs(violation - best_violation) <= 1e-9 and obj < best_obj)
        ):
            best_labels = labels_t.copy()
            best_violation = violation
            best_obj = obj

        if violation <= 1e-6:
            break

        subgrad = (load_t - cap_vec) / np.maximum(cap_vec, 1e-6)
        subgrad[~slot_mask_np] = 0.0
        step = float(hp.dual_step0) * cap_scale / np.sqrt(float(t + 1))
        mu = np.maximum(0.0, mu + step * subgrad.astype(np.float32))
        prev_load = load_t

    if best_labels is None:
        raise RuntimeError("HF Stage-A projection failed to produce an assignment.")

    labels = _hf_repair_assignment(
        labels=best_labels,
        full_cost=full_cost,
        dem=dem,
        cap_vec=cap_vec,
        slot_mask=slot_mask_np,
        max_rounds=max(1000, 20 * n),
    )

    ok, info = _check_hf_labels_feasible(
        labels,
        dem,
        cap_vec,
        slot_mask=slot_mask_np,
    )

    if not ok:
        labels = _hf_greedy_feasible_assignment(
            full_cost=full_cost,
            dem=dem,
            cap_vec=cap_vec,
            slot_mask=slot_mask_np,
        )

    _require_hf_labels_feasible(
        labels,
        dem,
        cap_vec,
        slot_mask=slot_mask_np,
        context="Stage-A assignment",
    )
    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Stage C: PyVRP component refinement
# ---------------------------------------------------------------------------

def _build_hf_neighbours(
    prob_slot: np.ndarray,
    clients_xy: np.ndarray,
    routes0_full: List[List[int]],
    tier_vec: np.ndarray,
    *,
    slot_mask: np.ndarray,
) -> Tuple[List[List[int]], Dict[str, object]]:
    """Build the fixed-budget geometry + tier + route-cover neighbourhood."""
    return build_competitive_slot_route_neighbours(
        prob_slot=prob_slot,
        clients_xy=clients_xy,
        routes0_full=routes0_full,
        group_vec=tier_vec,
        slot_mask=slot_mask,
        defaults=HFVRP_READ_PRESET.neighbours,
        enable_group_heat=True,
    )


def pyvrp_refine_hf(
    labels: np.ndarray,
    prob_slot: np.ndarray,
    dem: np.ndarray,
    cap_vec: np.ndarray,
    fixed_vec: np.ndarray,
    unit_cost_vec: np.ndarray,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    routes0_full: List[List[int]],
    *,
    budget_ms: float,
    seed: int = 0,
    slot_mask: Optional[np.ndarray] = None,
    tier_vec: Optional[np.ndarray] = None,
    return_profile: bool = False,
):
    """Refine the Stage-B HFVRP seed solution with the common PyVRP runner."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    prob_slot = row_normalize(prob_slot)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    cap_vec = np.asarray(cap_vec, dtype=np.float32).reshape(-1)
    fixed_vec = np.asarray(fixed_vec, dtype=np.float32).reshape(-1)
    unit_cost_vec = np.asarray(unit_cost_vec, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)

    n, k = prob_slot.shape
    if labels.shape != (n,):
        raise ValueError(f"labels.shape={labels.shape}, expected {(n,)}")
    if dem.shape != (n,):
        raise ValueError(f"dem.shape={dem.shape}, expected {(n,)}")
    if cap_vec.shape != (k,) or fixed_vec.shape != (k,) or unit_cost_vec.shape != (k,):
        raise ValueError("cap_vec, fixed_vec, and unit_cost_vec must all have shape (K,).")
    if clients_xy.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_xy.shape}, expected {(n, 2)}")
    if len(routes0_full) != k:
        raise ValueError(f"routes0_full must contain one route per slot: got {len(routes0_full)}, expected {k}")

    if slot_mask is None:
        slot_mask_np = np.ones((k,), dtype=np.bool_)
    else:
        slot_mask_np = np.asarray(slot_mask, dtype=np.bool_).reshape(k)
    if not np.any(slot_mask_np):
        raise ValueError("slot_mask must contain at least one active slot.")

    if tier_vec is None:
        tier_vec_np = infer_tier_vec(cap_vec, fixed_vec, unit_cost_vec)
    else:
        tier_vec_np = np.asarray(tier_vec, dtype=np.int64).reshape(k)

    routes0 = [list(map(int, route)) for route in routes0_full]
    base_info = _require_hf_routes_feasible(
        routes0,
        dem,
        cap_vec,
        num_clients=n,
        slot_mask=slot_mask_np,
        context="Stage-B seed solution",
    )
    base_cost = exact_hf_cost(routes0, depot_xy, clients_xy, fixed_vec, unit_cost_vec)

    neighbours, neigh_profile = _build_hf_neighbours(
        prob_slot=prob_slot,
        clients_xy=clients_xy,
        routes0_full=routes0,
        tier_vec=tier_vec_np,
        slot_mask=slot_mask_np,
    )

    build_cfg = HFPyVRPBuildConfig(
        dist_scale=int(HFVRP_READ_PRESET.pyvrp.dist_scale),
        demand_scale=int(HFVRP_READ_PRESET.pyvrp.demand_scale),
        cost_scale=float(HFVRP_READ_PRESET.pyvrp.cost_scale),
    )
    neigh_cfg = HFPyVRPNeighbourConfig(
        num_neighbours=int(HFVRP_READ_PRESET.neighbours.max_neigh),
        weight_wait_time=0.2,
        symmetric_proximity=bool(HFVRP_READ_PRESET.neighbours.symmetric),
    )
    ils_cfg = HFPyVRPILSConfig(
        budget_ms=float(budget_ms),
        seed=int(seed),
        collect_stats=False,
        display=False,
    )

    run = refine_hf_with_component_ils(
        routes0_by_slot=routes0,
        dem=dem,
        cap_vec=cap_vec,
        fixed_vec=fixed_vec,
        unit_cost_vec=unit_cost_vec,
        depot_xy=depot_xy,
        clients_xy=clients_xy,
        tier_vec=tier_vec_np,
        slot_mask=slot_mask_np,
        neighbours=neighbours,
        build_config=build_cfg,
        neighbour_config=neigh_cfg,
        ils_config=ils_cfg,
        canonicalize=True,
    )

    status = str(getattr(run, "status", "unknown"))
    if status not in {"ok_feasible", "ok_unknown_feasibility"}:
        detail = getattr(run, "exception_text", None)
        raise RuntimeError(f"HF PyVRP refinement failed with status={status}: {detail}")
    if getattr(run, "routes_by_slot", None) is None or getattr(run, "exact_cost", None) is None:
        raise RuntimeError("HF PyVRP refinement returned no complete route solution.")

    refined_routes = [list(map(int, route)) for route in run.routes_by_slot]
    refined_info = _require_hf_routes_feasible(
        refined_routes,
        dem,
        cap_vec,
        num_clients=n,
        slot_mask=slot_mask_np,
        context="refined solution",
    )
    refined_cost = float(run.exact_cost)
    if not np.isfinite(refined_cost):
        raise RuntimeError("HF PyVRP refinement returned a non-finite cost.")

    accepted = bool(refined_cost <= float(base_cost) + 1e-9)
    if accepted:
        final_routes = refined_routes
        final_lab = labels_from_slot_routes(final_routes, n)
        final_cost = refined_cost
    else:
        final_routes = routes0
        final_lab = labels.copy()
        final_cost = float(base_cost)

    if return_profile:
        profile: Dict[str, object] = {
            "hf_pyvrp_used": True,
            "hf_pyvrp_accepted": accepted,
            "hf_pyvrp_improved": bool(refined_cost + 1e-9 < float(base_cost)),
            "pyvrp_status": status,
            "pyvrp_ms": float(getattr(run, "elapsed_ms", 0.0)),
            "pyvrp_result_feasible": getattr(run, "is_feasible", None),
            "pyvrp_result_cost": getattr(run, "pyvrp_cost", None),
            "seed_used_routes": int(base_info["used_routes"]),
            "final_used_routes": int(refined_info["used_routes"] if accepted else base_info["used_routes"]),
        }
        if getattr(run, "operators", None):
            profile["pyvrp_num_ops"] = int(len(run.operators))
            profile["pyvrp_ops"] = ",".join(map(str, run.operators))
        if getattr(run, "neighbourhood_stats", None):
            stats = dict(run.neighbourhood_stats)
            profile["common_neigh_min"] = stats.get("min")
            profile["common_neigh_mean"] = stats.get("mean")
            profile["common_neigh_max"] = stats.get("max")
            profile["common_neigh_zero"] = stats.get("zero")
        profile.update(neigh_profile)
        return final_lab.astype(np.int64), final_cost, final_routes, profile

    return final_lab.astype(np.int64), final_cost, final_routes


# ---------------------------------------------------------------------------
# Top-level decode entry points
# ---------------------------------------------------------------------------

def _as_numpy(x, *, dtype=None) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _make_result(
    prob_c,
    seed_lab: np.ndarray,
    final_lab: np.ndarray,
    stagea_cost: float,
    refined_cost: float,
    nll_seed: float,
    nll_final: float,
    *,
    seed_routes: List[List[int]],
    final_routes: List[List[int]],
    profile: Optional[Dict[str, object]],
) -> READDecodeResult:
    target_device = prob_c.device if torch.is_tensor(prob_c) else torch.device("cpu")
    return READDecodeResult(
        seed_lab_t=torch.from_numpy(seed_lab.astype(np.int64)).long().to(target_device),
        final_lab_t=torch.from_numpy(final_lab.astype(np.int64)).long().to(target_device),
        stagea_cost=float(stagea_cost),
        refined_cost=float(refined_cost),
        nll_seed=float(nll_seed),
        nll_final=float(nll_final),
        seed_routes=seed_routes,
        final_routes=final_routes,
        profile=profile,
    )


def decode_read_struct(
    prob_c,
    dem,
    cap_vec,
    fixed_vec,
    unit_cost_vec,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
    *,
    seed: int = 0,
    return_profile: bool = False,
    slot_mask: Optional[np.ndarray] = None,
    tier_vec: Optional[np.ndarray] = None,
) -> READDecodeResult:
    """Decode a single HFVRP instance from a client-to-slot posterior."""
    timer = StageTimer(enabled=bool(return_profile))
    profile: Dict[str, object] = {}

    prob_np = row_normalize(_as_numpy(prob_c, dtype=np.float32))
    dem_np = _as_numpy(dem, dtype=np.float32).reshape(-1)
    cap_np = _as_numpy(cap_vec, dtype=np.float32).reshape(-1)
    fixed_np = _as_numpy(fixed_vec, dtype=np.float32).reshape(-1)
    unit_np = _as_numpy(unit_cost_vec, dtype=np.float32).reshape(-1)
    depot_np = _as_numpy(depot_xy, dtype=np.float32).reshape(2)
    clients_np = _as_numpy(clients_xy, dtype=np.float32)

    n, k = prob_np.shape
    if dem_np.shape != (n,):
        raise ValueError(f"dem.shape={dem_np.shape}, expected {(n,)}")
    if cap_np.shape != (k,) or fixed_np.shape != (k,) or unit_np.shape != (k,):
        raise ValueError("cap_vec, fixed_vec, and unit_cost_vec must all have shape (K,).")
    if clients_np.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_np.shape}, expected {(n, 2)}")

    if slot_mask is None:
        slot_mask_np = np.ones((k,), dtype=np.bool_)
    else:
        slot_mask_np = np.asarray(slot_mask, dtype=np.bool_).reshape(k)
    if not np.any(slot_mask_np):
        raise ValueError("slot_mask must contain at least one active slot.")

    if tier_vec is None:
        tier_vec_np = infer_tier_vec(cap_np, fixed_np, unit_np)
    else:
        tier_vec_np = np.asarray(tier_vec, dtype=np.int64).reshape(k)

    timer.tic("projector")
    seed_lab = stage_a_capacity_project_hf(
        prob=prob_np,
        dem=dem_np,
        cap_vec=cap_np,
        fixed_vec=fixed_np,
        unit_cost_vec=unit_np,
        depot_xy=depot_np,
        clients_xy=clients_np,
        slot_mask=slot_mask_np,
    )
    timer.toc("projector")

    nll_seed = nll_from_prob_and_labels(prob_np, seed_lab)

    route_cfg = HFVRP_READ_PRESET.route
    timer.tic("seed_route")
    seed_routes = build_seed_routes(
        seed_lab,
        int(k),
        depot_np,
        clients_np,
        two_opt=bool(route_cfg.two_opt),
        two_opt_iter=int(route_cfg.two_opt_iter),
    )
    seed_info = _require_hf_routes_feasible(
        seed_routes,
        dem_np,
        cap_np,
        num_clients=int(n),
        slot_mask=slot_mask_np,
        context="Stage-B seed solution",
    )
    stagea_cost = exact_hf_cost(
        seed_routes,
        depot_np,
        clients_np,
        fixed_np,
        unit_np,
    )
    timer.toc("seed_route")

    final_lab = seed_lab.copy()
    refined_cost = float(stagea_cost)
    final_routes = [list(route) for route in seed_routes]

    if float(cfg.pyvrp_budget_ms) > 0.0:
        timer.tic("pyvrp")
        if return_profile:
            final_lab, refined_cost, final_routes, pyvrp_profile = pyvrp_refine_hf(
                labels=seed_lab,
                prob_slot=prob_np,
                dem=dem_np,
                cap_vec=cap_np,
                fixed_vec=fixed_np,
                unit_cost_vec=unit_np,
                depot_xy=depot_np,
                clients_xy=clients_np,
                routes0_full=seed_routes,
                budget_ms=float(cfg.pyvrp_budget_ms),
                seed=int(seed),
                slot_mask=slot_mask_np,
                tier_vec=tier_vec_np,
                return_profile=True,
            )
            profile.update(pyvrp_profile)
        else:
            final_lab, refined_cost, final_routes = pyvrp_refine_hf(
                labels=seed_lab,
                prob_slot=prob_np,
                dem=dem_np,
                cap_vec=cap_np,
                fixed_vec=fixed_np,
                unit_cost_vec=unit_np,
                depot_xy=depot_np,
                clients_xy=clients_np,
                routes0_full=seed_routes,
                budget_ms=float(cfg.pyvrp_budget_ms),
                seed=int(seed),
                slot_mask=slot_mask_np,
                tier_vec=tier_vec_np,
                return_profile=False,
            )
        timer.toc("pyvrp")
    elif return_profile:
        profile["hf_pyvrp_used"] = False
        profile["seed_used_routes"] = int(seed_info["used_routes"])
        profile["final_used_routes"] = int(seed_info["used_routes"])

    if return_profile:
        profile.update(timer.as_dict())

    nll_final = nll_from_prob_and_labels(prob_np, np.asarray(final_lab, dtype=np.int64))
    return _make_result(
        prob_c=prob_c,
        seed_lab=seed_lab,
        final_lab=np.asarray(final_lab, dtype=np.int64),
        stagea_cost=stagea_cost,
        refined_cost=refined_cost,
        nll_seed=nll_seed,
        nll_final=nll_final,
        seed_routes=seed_routes,
        final_routes=final_routes,
        profile=profile if return_profile else None,
    )


def decode_read_batch_struct(
    jobs: List[dict],
    *,
    max_workers: Optional[int] = None,
    return_profile: bool = False,
) -> List[READDecodeResult]:
    """Decode a list of HFVRP instances. Exceptions propagate to the caller."""
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 8)
    max_workers = max(1, int(max_workers))

    results: List[Optional[READDecodeResult]] = [None] * len(jobs)

    def worker(i: int, job: dict) -> Tuple[int, READDecodeResult]:
        return i, decode_read_struct(
            prob_c=job["prob_c"],
            dem=job["dem"],
            cap_vec=job["cap_vec"],
            fixed_vec=job["fixed_vec"],
            unit_cost_vec=job["unit_cost_vec"],
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
            seed=int(job.get("seed", 0)),
            return_profile=bool(return_profile),
            slot_mask=job.get("slot_mask", None),
            tier_vec=job.get("tier_vec", None),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i, jobs[i]) for i in range(len(jobs))]
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"HFVRP decoder lost {len(missing)} results; first missing index={missing[0]}")
    return results  # type: ignore[return-value]


__all__ = [
    "READDecodeCfg",
    "READDecodeResult",
    "decode_read_struct",
    "decode_read_batch_struct",
    "pyvrp_refine_hf",
    "stage_a_capacity_project_hf",
]
