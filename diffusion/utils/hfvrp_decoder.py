"""READ decoder for the Heterogeneous Fleet Vehicle Routing Problem (HFVRP).

The inference pipeline is deliberately split into two stages because the model
uses best-of-S sampling: construct one feasible seed per sampled posterior,
select the lowest-cost seed, and refine only the selected seed with PyVRP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .common import (
    as_numpy,
    build_read_neighbours,
    build_seed_routes,
    decode_jobs_in_threads,
    make_decode_result,
    nll_from_prob_and_labels,
    row_normalize,
)
from .pyvrp_hfvrp_common import (
    HFPyVRPBuildConfig,
    HFPyVRPILSConfig,
    exact_hf_cost,
    infer_tier_vec,
    labels_from_slot_routes,
    refine_hf_with_component_ils,
)
from .read_config import READDecodeCfg, HFVRP_READ_PRESET, READDecodeResult
from .read_seed_constructor import (
    active_slot_mask,
    assert_instance_feasible,
    construct_slot_seed_labels,
)


# ---------------------------------------------------------------------------
# Instance conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HFVRPInstance:
    prob: np.ndarray
    demand: np.ndarray
    capacity: np.ndarray
    fixed_cost: np.ndarray
    unit_cost: np.ndarray
    depot_xy: np.ndarray
    clients_xy: np.ndarray
    slot_mask: np.ndarray
    tier: np.ndarray


def _as_instance(
    prob_c,
    demand,
    capacity,
    fixed_cost,
    unit_cost,
    depot_xy,
    clients_xy,
    *,
    slot_mask: Optional[np.ndarray] = None,
    tier_vec: Optional[np.ndarray] = None,
) -> _HFVRPInstance:
    prob = row_normalize(as_numpy(prob_c, dtype=np.float32))
    dem = as_numpy(demand, dtype=np.float32).reshape(-1)
    cap = as_numpy(capacity, dtype=np.float32).reshape(-1)
    depot = as_numpy(depot_xy, dtype=np.float32).reshape(2)
    clients = as_numpy(clients_xy, dtype=np.float32)

    n, k = prob.shape
    fixed = (
        np.zeros((k,), dtype=np.float32)
        if fixed_cost is None
        else as_numpy(fixed_cost, dtype=np.float32).reshape(-1)
    )
    unit = (
        np.ones((k,), dtype=np.float32)
        if unit_cost is None
        else as_numpy(unit_cost, dtype=np.float32).reshape(-1)
    )
    if dem.shape != (n,):
        raise ValueError(f"demand.shape={dem.shape}, expected {(n,)}")
    if cap.shape != (k,) or fixed.shape != (k,) or unit.shape != (k,):
        raise ValueError("capacity, fixed_cost, and unit_cost must all have shape (K,).")
    if clients.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients.shape}, expected {(n, 2)}")

    mask = active_slot_mask(k, slot_mask)
    tier = (
        infer_tier_vec(cap, fixed, unit)
        if tier_vec is None
        else np.asarray(tier_vec, dtype=np.int64).reshape(k)
    )

    assert_instance_feasible(dem, cap, mask)
    return _HFVRPInstance(
        prob=prob,
        demand=dem,
        capacity=cap,
        fixed_cost=fixed,
        unit_cost=unit,
        depot_xy=depot,
        clients_xy=clients,
        slot_mask=mask,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Feasibility checks
# ---------------------------------------------------------------------------


def _assert_routes_feasible(
    routes_by_slot: List[List[int]],
    demand: np.ndarray,
    capacity: np.ndarray,
    slot_mask: np.ndarray,
    *,
    num_clients: int,
    context: str,
    tol: float = 1e-6,
) -> int:
    demand = np.asarray(demand, dtype=np.float32).reshape(-1)
    capacity = np.asarray(capacity, dtype=np.float32).reshape(-1)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)

    n = int(num_clients)
    k = int(capacity.size)
    if len(routes_by_slot) != k:
        raise RuntimeError(f"{context} produced {len(routes_by_slot)} routes for {k} slots.")

    seen = np.zeros((n,), dtype=np.int32)
    used_routes = 0
    for slot, route in enumerate(routes_by_slot):
        if not route:
            continue
        if not bool(slot_mask[slot]):
            raise RuntimeError(f"{context} produced a non-empty inactive route at slot {slot}.")
        used_routes += 1

        load = 0.0
        for client_id in route:
            cid = int(client_id)
            if cid < 1 or cid > n:
                raise RuntimeError(f"{context} used client id {cid}, outside 1..{n}.")
            seen[cid - 1] += 1
            if seen[cid - 1] > 1:
                raise RuntimeError(f"{context} visits client {cid} more than once.")
            load += float(demand[cid - 1])

        if load > float(capacity[slot]) + float(tol):
            raise RuntimeError(
                f"{context} exceeded capacity on slot {slot}: "
                f"load={load:.6f}, capacity={float(capacity[slot]):.6f}."
            )

    missing = np.where(seen == 0)[0]
    if missing.size:
        raise RuntimeError(f"{context} missed client {int(missing[0] + 1)}.")
    return int(used_routes)


# ---------------------------------------------------------------------------
# Seed construction
# ---------------------------------------------------------------------------


def construct_seed_struct(
    prob_c,
    demand,
    capacity,
    fixed_cost,
    unit_cost,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
    *,
    slot_mask: Optional[np.ndarray] = None,
) -> READDecodeResult:
    inst = _as_instance(
        prob_c,
        demand,
        capacity,
        fixed_cost,
        unit_cost,
        depot_xy,
        clients_xy,
        slot_mask=slot_mask,
    )

    seed_labels = construct_slot_seed_labels(
        inst.prob,
        inst.demand,
        inst.capacity,
        inst.depot_xy,
        inst.clients_xy,
        cfg,
        fixed_cost_vec=inst.fixed_cost,
        unit_cost_vec=inst.unit_cost,
        slot_mask=inst.slot_mask,
    )

    nll_seed = nll_from_prob_and_labels(inst.prob, seed_labels)

    seed_routes = build_seed_routes(seed_labels, int(inst.capacity.size), inst.depot_xy, inst.clients_xy)
    _assert_routes_feasible(
        seed_routes,
        inst.demand,
        inst.capacity,
        inst.slot_mask,
        num_clients=int(inst.demand.size),
        context="HFVRP seed routes",
    )
    stagea_cost = exact_hf_cost(seed_routes, inst.depot_xy, inst.clients_xy, inst.fixed_cost, inst.unit_cost)

    return make_decode_result(
        prob_c=prob_c,
        seed_lab=seed_labels,
        final_lab=seed_labels.copy(),
        stagea_cost=stagea_cost,
        refined_cost=stagea_cost,
        nll_seed=nll_seed,
        nll_final=nll_seed,
        seed_routes=seed_routes,
        final_routes=[list(route) for route in seed_routes],
    )


def construct_seed_batch(
    jobs: List[dict],
    *,
    max_workers: Optional[int] = None,
) -> List[READDecodeResult]:
    def worker(i: int, job: dict) -> READDecodeResult:
        del i
        return construct_seed_struct(
            prob_c=job["prob_c"],
            demand=job["dem"],
            capacity=job["cap_vec"],
            fixed_cost=job.get("fixed_vec", None),
            unit_cost=job.get("unit_cost_vec", None),
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
            slot_mask=job.get("slot_mask", None),
        )

    return decode_jobs_in_threads(
        jobs,
        worker,
        max_workers=max_workers,
        missing_message="HFVRP seed construction produced incomplete results.",
    )


# ---------------------------------------------------------------------------
# PyVRP refinement
# ---------------------------------------------------------------------------


def _build_hf_neighbours(
    prob_slot: np.ndarray,
    clients_xy: np.ndarray,
    *,
    demand: np.ndarray,
    capacity: np.ndarray,
    slot_mask: np.ndarray,
) -> List[List[int]]:
    return build_read_neighbours(
        prob_slot=prob_slot,
        clients_xy=clients_xy,
        dem=demand,
        cap_vec=capacity,
        slot_mask=slot_mask,
        defaults=HFVRP_READ_PRESET.neighbours,
        prepend_depot=False,
    )


def refine_with_pyvrp(
    labels: np.ndarray,
    prob_slot: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    fixed_cost: np.ndarray,
    unit_cost: np.ndarray,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    routes_by_slot: List[List[int]],
    cfg: READDecodeCfg,
    *,
    seed: int = 0,
    slot_mask: np.ndarray,
    tier_vec: np.ndarray,
    base_cost: float,
):
    prob_slot = row_normalize(prob_slot)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    demand = np.asarray(demand, dtype=np.float32).reshape(-1)
    capacity = np.asarray(capacity, dtype=np.float32).reshape(-1)
    fixed_cost = np.asarray(fixed_cost, dtype=np.float32).reshape(-1)
    unit_cost = np.asarray(unit_cost, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)
    slot_mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
    tier_vec = np.asarray(tier_vec, dtype=np.int64).reshape(-1)
    routes0 = [list(map(int, route)) for route in routes_by_slot]

    n, k = prob_slot.shape
    if labels.shape != (n,):
        raise ValueError(f"labels.shape={labels.shape}, expected {(n,)}")
    if demand.shape != (n,):
        raise ValueError(f"demand.shape={demand.shape}, expected {(n,)}")
    if capacity.shape != (k,) or fixed_cost.shape != (k,) or unit_cost.shape != (k,):
        raise ValueError("capacity, fixed_cost, and unit_cost must all have shape (K,).")
    if clients_xy.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_xy.shape}, expected {(n, 2)}")
    if len(routes0) != k:
        raise ValueError(f"routes_by_slot must contain {k} slot routes, got {len(routes0)}.")

    _assert_routes_feasible(
        routes0,
        demand,
        capacity,
        slot_mask,
        num_clients=int(n),
        context="HFVRP refinement input",
    )

    neighbours = _build_hf_neighbours(
        prob_slot=prob_slot,
        clients_xy=clients_xy,
        demand=demand,
        capacity=capacity,
        slot_mask=slot_mask,
    )
    build_cfg = HFPyVRPBuildConfig(
        dist_scale=int(HFVRP_READ_PRESET.pyvrp.dist_scale),
        demand_scale=int(HFVRP_READ_PRESET.pyvrp.demand_scale),
        cost_scale=float(HFVRP_READ_PRESET.pyvrp.cost_scale),
    )
    ils_cfg = HFPyVRPILSConfig(
        budget_ms=float(cfg.pyvrp_budget_ms),
        seed=int(seed),
        collect_stats=False,
        display=False,
    )

    run = refine_hf_with_component_ils(
        routes0_by_slot=routes0,
        dem=demand,
        cap_vec=capacity,
        fixed_vec=fixed_cost,
        unit_cost_vec=unit_cost,
        depot_xy=depot_xy,
        clients_xy=clients_xy,
        tier_vec=tier_vec,
        slot_mask=slot_mask,
        neighbours=neighbours,
        build_config=build_cfg,
        ils_config=ils_cfg,
        canonicalize=True,
    )

    status = str(getattr(run, "status", "unknown"))
    refined_routes: List[List[int]] = []
    refined_cost = float("inf")
    refined_feasible = False

    if status in {"ok_feasible", "ok_unknown_feasibility"}:
        if getattr(run, "routes_by_slot", None) is not None and getattr(run, "exact_cost", None) is not None:
            refined_routes = [list(map(int, route)) for route in run.routes_by_slot]
            try:
                _assert_routes_feasible(
                    refined_routes,
                    demand,
                    capacity,
                    slot_mask,
                    num_clients=int(n),
                    context="HFVRP PyVRP refinement result",
                )
                refined_feasible = True
                refined_cost = float(run.exact_cost)
            except RuntimeError:
                refined_feasible = False

    accepted = bool(
        refined_feasible
        and np.isfinite(refined_cost)
        and refined_cost <= float(base_cost) + 1e-9
    )
    if accepted:
        final_routes = refined_routes
        final_labels = labels_from_slot_routes(final_routes, n)
        final_cost = float(refined_cost)
    else:
        final_routes = routes0
        final_labels = labels.copy()
        final_cost = float(base_cost)

    return final_labels.astype(np.int64), final_cost, final_routes


def refine_seed_struct(
    prob_c,
    demand,
    capacity,
    fixed_cost,
    unit_cost,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
    *,
    seed_lab,
    seed_routes,
    seed: int = 0,
    slot_mask: Optional[np.ndarray] = None,
    tier_vec: Optional[np.ndarray] = None,
) -> READDecodeResult:
    inst = _as_instance(
        prob_c,
        demand,
        capacity,
        fixed_cost,
        unit_cost,
        depot_xy,
        clients_xy,
        slot_mask=slot_mask,
        tier_vec=tier_vec,
    )
    seed_labels = as_numpy(seed_lab, dtype=np.int64).reshape(int(inst.demand.size))
    seed_routes = [list(map(int, route)) for route in seed_routes]

    _assert_routes_feasible(
        seed_routes,
        inst.demand,
        inst.capacity,
        inst.slot_mask,
        num_clients=int(inst.demand.size),
        context="Selected HFVRP seed routes",
    )
    stagea_cost = exact_hf_cost(seed_routes, inst.depot_xy, inst.clients_xy, inst.fixed_cost, inst.unit_cost)

    nll_seed = nll_from_prob_and_labels(inst.prob, seed_labels)
    final_labels = seed_labels.copy()
    final_routes = [list(route) for route in seed_routes]
    refined_cost = float(stagea_cost)

    if float(cfg.pyvrp_budget_ms) > 0.0:
        final_labels, refined_cost, final_routes = refine_with_pyvrp(
            labels=seed_labels,
            prob_slot=inst.prob,
            demand=inst.demand,
            capacity=inst.capacity,
            fixed_cost=inst.fixed_cost,
            unit_cost=inst.unit_cost,
            depot_xy=inst.depot_xy,
            clients_xy=inst.clients_xy,
            routes_by_slot=seed_routes,
            cfg=cfg,
            seed=int(seed),
            slot_mask=inst.slot_mask,
            tier_vec=inst.tier,
            base_cost=float(stagea_cost),
        )

    nll_final = nll_from_prob_and_labels(inst.prob, np.asarray(final_labels, dtype=np.int64))
    return make_decode_result(
        prob_c=prob_c,
        seed_lab=seed_labels,
        final_lab=np.asarray(final_labels, dtype=np.int64),
        stagea_cost=stagea_cost,
        refined_cost=refined_cost,
        nll_seed=nll_seed,
        nll_final=nll_final,
        seed_routes=seed_routes,
        final_routes=final_routes,
    )


def refine_seed_batch(
    jobs: List[dict],
    *,
    max_workers: Optional[int] = None,
) -> List[READDecodeResult]:
    def worker(i: int, job: dict) -> READDecodeResult:
        del i
        return refine_seed_struct(
            prob_c=job["prob_c"],
            demand=job["dem"],
            capacity=job["cap_vec"],
            fixed_cost=job.get("fixed_vec", None),
            unit_cost=job.get("unit_cost_vec", None),
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
            seed_lab=job["seed_lab"],
            seed_routes=job["seed_routes"],
            seed=int(job.get("seed", 0)),
            slot_mask=job.get("slot_mask", None),
            tier_vec=job.get("tier_vec", None),
        )

    return decode_jobs_in_threads(
        jobs,
        worker,
        max_workers=max_workers,
        missing_message="HFVRP refinement produced incomplete results.",
    )


__all__ = [
    "READDecodeCfg",
    "READDecodeResult",
    "construct_seed_struct",
    "construct_seed_batch",
    "refine_with_pyvrp",
    "refine_seed_struct",
    "refine_seed_batch",
]
