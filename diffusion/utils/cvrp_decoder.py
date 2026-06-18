"""READ decoder for the Capacitated Vehicle Routing Problem (CVRP)."""

from __future__ import annotations



from typing import Dict, List, Optional, Tuple


import numpy as np


from .read_config import READDecodeCfg, READDecodeResult, CVRP_READ_PRESET

from .common import (
    build_seed_routes,
    nll_from_prob_and_labels,
    pyvrp_route_clients,
    row_normalize,
    run_pyvrp_ils,
    as_numpy,
    build_read_neighbours,
    decode_jobs_in_threads,
    make_decode_result,
)
from .read_seed_constructor import construct_slot_seed_labels



# ---------------------------------------------------------------------------
# Cost and feasibility
# ---------------------------------------------------------------------------

def _cost_from_routes(routes: List[List[int]], depot_xy: np.ndarray,
                      clients_xy: np.ndarray) -> float:
    """Sum of Euclidean route lengths, depot included on both ends."""
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)
    total = 0.0
    for route in routes:
        if not route:
            continue
        idx = np.asarray(route, dtype=np.int64) - 1
        pts = clients_xy[idx]
        total += float(np.linalg.norm(depot_xy - pts[0]))
        if pts.shape[0] >= 2:
            total += float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())
        total += float(np.linalg.norm(pts[-1] - depot_xy))
    return float(total)


def _check_routes_feasible(
    routes: List[List[int]],
    dem: np.ndarray,
    cap: float,
    *,
    num_clients: int,
    k_max: Optional[int] = None,
    tol: float = 1e-6,
) -> Tuple[bool, Dict[str, object]]:
    """Validate customer coverage, capacity, and route-count feasibility."""
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    n = int(num_clients)
    if dem.size != n:
        return False, {"reason": "demand_size_mismatch", "num_clients": n, "dem_size": int(dem.size)}

    seen = np.zeros((n,), dtype=np.int32)
    used_routes = 0
    max_load = 0.0

    for route_idx, route in enumerate(routes):
        if not route:
            continue
        used_routes += 1
        load = 0.0
        for cid in route:
            cid = int(cid)
            if cid < 1 or cid > n:
                return False, {
                    "reason": "client_id_out_of_range",
                    "route_idx": int(route_idx),
                    "client_id": int(cid),
                    "num_clients": n,
                }
            seen[cid - 1] += 1
            if seen[cid - 1] > 1:
                return False, {"reason": "duplicate_client", "route_idx": int(route_idx), "client_id": int(cid)}
            load += float(dem[cid - 1])
        max_load = max(max_load, load)
        if load > float(cap) + float(tol):
            return False, {
                "reason": "capacity_overflow",
                "route_idx": int(route_idx),
                "route_load": float(load),
                "capacity": float(cap),
            }

    missing = np.where(seen == 0)[0]
    if missing.size:
        return False, {
            "reason": "missing_clients",
            "missing_count": int(missing.size),
            "first_missing_client": int(missing[0] + 1),
        }

    if k_max is not None and used_routes > int(k_max):
        return False, {"reason": "too_many_routes", "used_routes": int(used_routes), "k_max": int(k_max)}

    return True, {"reason": "ok", "used_routes": int(used_routes), "max_route_load": float(max_load)}


def _require_feasible_routes(
    routes: List[List[int]],
    dem: np.ndarray,
    cap: float,
    *,
    num_clients: int,
    k_max: int,
    context: str,
) -> Dict[str, object]:
    ok, info = _check_routes_feasible(routes, dem, cap, num_clients=num_clients, k_max=k_max)
    if not ok:
        raise ValueError(f"{context} infeasible: {info}")
    return info


# ---------------------------------------------------------------------------
# Stage C: PyVRP problem construction and refinement
# ---------------------------------------------------------------------------

def _build_pyvrp_neighbours(
    prob: np.ndarray,
    clients_xy: np.ndarray,
    *,
    dem: np.ndarray,
    cap: float,
) -> List[List[int]]:
    cap_vec = np.full((prob.shape[1],), float(cap), dtype=np.float32)

    return build_read_neighbours(
        prob_slot=prob,
        clients_xy=clients_xy,
        dem=dem,
        cap_vec=cap_vec,
        defaults=CVRP_READ_PRESET.neighbours,
        prepend_depot=True,
    )


def _build_pyvrp_problem(dem: np.ndarray, cap: float, k: int,
                         depot_xy: np.ndarray, clients_xy: np.ndarray):
    """Build homogeneous CVRP ProblemData using the public CVRP preset."""
    from pyvrp import Client, Depot, ProblemData, VehicleType

    scale_dist = int(CVRP_READ_PRESET.pyvrp.dist_scale)
    scale_dem = int(CVRP_READ_PRESET.pyvrp.demand_scale)

    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    n = int(dem.size)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32).reshape(n, 2)

    eps = 1e-9
    cap_i = int(np.ceil(float(cap) * float(scale_dem) - eps))
    dem_i = np.maximum(
        0,
        np.floor(dem * float(scale_dem) + eps).astype(np.int64),
    )

    dep_xy_i = np.round(depot_xy * float(scale_dist)).astype(np.int64)
    cli_xy_i = np.round(clients_xy * float(scale_dist)).astype(np.int64)

    depots = [Depot(x=int(dep_xy_i[0]), y=int(dep_xy_i[1]))]
    clients = [
        Client(x=int(cli_xy_i[i, 0]), y=int(cli_xy_i[i, 1]), delivery=[int(dem_i[i])])
        for i in range(n)
    ]
    vehicle_types = [VehicleType(num_available=int(k), capacity=[cap_i], unit_distance_cost=1)]

    locs_i = np.vstack([dep_xy_i.reshape(1, 2), cli_xy_i]).astype(np.float64)
    diff = locs_i[:, None, :] - locs_i[None, :, :]
    dist_i = np.round(np.sqrt((diff * diff).sum(axis=2) + 1e-12)).astype(np.int64)

    return ProblemData(
        clients=clients,
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrices=[dist_i],
        duration_matrices=[dist_i],
    )


def _routes_from_solution(solution, num_clients: int) -> List[List[int]]:
    """Read all non-empty routes from a PyVRP solution."""
    return [
        route
        for route in (pyvrp_route_clients(route, num_clients) for route in solution.routes())
        if route
    ]


def _align_refined_routes_to_seed_slots(
    routes: List[List[int]],
    seed_lab: np.ndarray,
    prob: np.ndarray,
) -> np.ndarray:
    """Map each refined route to a seed slot via Hungarian assignment."""
    from scipy.optimize import linear_sum_assignment

    n, k = prob.shape
    if len(routes) > k:
        raise ValueError(f"refined solution uses {len(routes)} routes, but only {k} slots are available")

    score = np.zeros((len(routes), k), dtype=np.float32)
    for r_idx, route in enumerate(routes):
        idx = np.asarray(route, dtype=np.int64) - 1
        overlap = np.bincount(seed_lab[idx], minlength=k).astype(np.float32)
        posterior_mass = prob[idx].sum(axis=0).astype(np.float32)
        score[r_idx] = 2.0 * overlap + 0.5 * posterior_mass

    row_ind, col_ind = linear_sum_assignment(-score)
    route_to_slot = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    final_lab = np.zeros((n,), dtype=np.int64)
    for r_idx, route in enumerate(routes):
        slot = route_to_slot[int(r_idx)]
        for cid in route:
            final_lab[int(cid) - 1] = int(slot)
    return final_lab


def pyvrp_refine(
    labels: np.ndarray,
    prob: np.ndarray,
    dem: np.ndarray,
    cap: float,
    k: int,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    routes0_full: List[List[int]],
    *,
    budget_ms: float,
    seed: int = 0,
):
    """Improve the Stage-B seed solution with PyVRP local search."""
    from pyvrp import Solution

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    prob = row_normalize(prob)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)

    n = int(dem.size)
    k = int(k)
    if labels.shape != (n,):
        raise ValueError(f"labels.shape={labels.shape}, expected {(n,)}")
    if prob.shape != (n, k):
        raise ValueError(f"prob.shape={prob.shape}, expected {(n, k)}")
    if len(routes0_full) != k:
        raise ValueError(f"routes0_full must contain one route per slot: got {len(routes0_full)}, expected {k}")

    routes0_by_slot = [list(map(int, route)) for route in routes0_full]
    base_routes = [route for route in routes0_by_slot if route]
    _require_feasible_routes(base_routes, dem, float(cap), num_clients=n, k_max=k, context="seed solution")
    base_cost = _cost_from_routes(base_routes, depot_xy, clients_xy)

    data = _build_pyvrp_problem(dem=dem, cap=float(cap), k=k, depot_xy=depot_xy, clients_xy=clients_xy)
    init_sol = Solution(data, base_routes)
    if not init_sol.is_complete():
        raise RuntimeError("PyVRP rejected the Stage-B seed solution as incomplete.")

    neigh = _build_pyvrp_neighbours(
        prob=prob,
        clients_xy=clients_xy,
        dem=dem,
        cap=float(cap),
    )
    result = run_pyvrp_ils(data=data, init_sol=init_sol, neigh=neigh, seed=int(seed), budget_ms=float(budget_ms))

    best = result.best
    if not best.is_feasible():
        raise RuntimeError("PyVRP returned an infeasible best solution.")

    refined_routes = _routes_from_solution(best, n)
    _require_feasible_routes(
        refined_routes,
        dem,
        float(cap),
        num_clients=n,
        k_max=k,
        context="refined solution",
    )
    refined_cost = _cost_from_routes(refined_routes, depot_xy, clients_xy)

    if not np.isfinite(refined_cost):
        raise RuntimeError("PyVRP returned a non-finite refined cost.")

    accepted = bool(refined_cost <= float(base_cost) + 1e-9)
    if accepted:
        final_lab = _align_refined_routes_to_seed_slots(refined_routes, labels, prob)
        final_cost = float(refined_cost)
        final_routes = refined_routes
    else:
        final_lab = labels.copy()
        final_cost = float(base_cost)
        final_routes = base_routes

    return final_lab.astype(np.int64), final_cost, final_routes


# ---------------------------------------------------------------------------
# Top-level seed and refinement entry points
# ---------------------------------------------------------------------------

def construct_seed_struct(
    prob_c,
    dem,
    cap,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
) -> READDecodeResult:
    """Construct a capacity-feasible CVRP seed from a client-to-slot posterior."""
    prob_np = row_normalize(as_numpy(prob_c, dtype=np.float32))
    dem_np = as_numpy(dem, dtype=np.float32).reshape(-1)
    depot_np = as_numpy(depot_xy, dtype=np.float32).reshape(2)
    clients_np = as_numpy(clients_xy, dtype=np.float32)

    n, k = prob_np.shape
    if dem_np.shape != (n,):
        raise ValueError(f"dem.shape={dem_np.shape}, expected {(n,)}")
    if clients_np.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_np.shape}, expected {(n, 2)}")

    cap_vec = np.full((k,), float(cap), dtype=np.float32)

    seed_lab = construct_slot_seed_labels(
        prob=prob_np,
        demand=dem_np,
        capacity_vec=cap_vec,
        depot_xy=depot_np,
        clients_xy=clients_np,
        cfg=cfg,
    )

    nll_seed = nll_from_prob_and_labels(prob_np, seed_lab)

    routes_seed = build_seed_routes(seed_lab, int(k), depot_np, clients_np)
    _require_feasible_routes(
        routes_seed,
        dem_np,
        float(cap),
        num_clients=int(n),
        k_max=int(k),
        context="Stage-B seed solution",
    )
    stagea_cost = _cost_from_routes(routes_seed, depot_np, clients_np)

    return make_decode_result(
        prob_c=prob_c,
        seed_lab=seed_lab,
        final_lab=seed_lab.copy(),
        stagea_cost=stagea_cost,
        refined_cost=stagea_cost,
        nll_seed=nll_seed,
        nll_final=nll_seed,
        seed_routes=routes_seed,
        final_routes=[list(route) for route in routes_seed],
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
            dem=job["dem"],
            cap=job["cap"],
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
        )

    return decode_jobs_in_threads(
        jobs,
        worker,
        max_workers=max_workers,
        missing_message="CVRP seed construction produced incomplete results.",
    )


def refine_seed_struct(
    prob_c,
    dem,
    cap,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
    *,
    seed_lab,
    seed_routes,
    seed: int = 0,
) -> READDecodeResult:
    """Refine a selected CVRP seed with PyVRP."""
    prob_np = row_normalize(as_numpy(prob_c, dtype=np.float32))
    dem_np = as_numpy(dem, dtype=np.float32).reshape(-1)
    depot_np = as_numpy(depot_xy, dtype=np.float32).reshape(2)
    clients_np = as_numpy(clients_xy, dtype=np.float32)

    n, k = prob_np.shape
    seed_lab_np = as_numpy(seed_lab, dtype=np.int64).reshape(n)
    routes_seed = [list(map(int, route)) for route in seed_routes]

    if dem_np.shape != (n,):
        raise ValueError(f"dem.shape={dem_np.shape}, expected {(n,)}")
    if clients_np.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_np.shape}, expected {(n, 2)}")
    if len(routes_seed) != k:
        raise ValueError(f"seed_routes must contain {k} slot routes, got {len(routes_seed)}.")

    _require_feasible_routes(
        routes_seed,
        dem_np,
        float(cap),
        num_clients=int(n),
        k_max=int(k),
        context="Selected CVRP seed solution",
    )
    stagea_cost = _cost_from_routes(routes_seed, depot_np, clients_np)

    nll_seed = nll_from_prob_and_labels(prob_np, seed_lab_np)
    final_lab = seed_lab_np.copy()
    refined_cost = float(stagea_cost)
    routes_final = [list(route) for route in routes_seed]

    if float(cfg.pyvrp_budget_ms) > 0.0:
        final_lab, refined_cost, routes_final = pyvrp_refine(
            labels=seed_lab_np,
            prob=prob_np,
            dem=dem_np,
            cap=float(cap),
            k=int(k),
            depot_xy=depot_np,
            clients_xy=clients_np,
            routes0_full=routes_seed,
            budget_ms=float(cfg.pyvrp_budget_ms),
            seed=int(seed),
        )

    nll_final = nll_from_prob_and_labels(prob_np, np.asarray(final_lab, dtype=np.int64))
    return make_decode_result(
        prob_c=prob_c,
        seed_lab=seed_lab_np,
        final_lab=np.asarray(final_lab, dtype=np.int64),
        stagea_cost=stagea_cost,
        refined_cost=refined_cost,
        nll_seed=nll_seed,
        nll_final=nll_final,
        seed_routes=routes_seed,
        final_routes=routes_final,
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
            dem=job["dem"],
            cap=job["cap"],
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
            seed_lab=job["seed_lab"],
            seed_routes=job["seed_routes"],
            seed=int(job.get("seed", 0)),
        )

    return decode_jobs_in_threads(
        jobs,
        worker,
        max_workers=max_workers,
        missing_message="CVRP refinement produced incomplete results.",
    )

__all__ = [
    "READDecodeCfg",
    "READDecodeResult",
    "construct_seed_struct",
    "construct_seed_batch",
    "refine_seed_struct",
    "refine_seed_batch",
    "pyvrp_refine",
]
