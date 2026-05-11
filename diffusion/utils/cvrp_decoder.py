"""READ decoder for the Capacitated Vehicle Routing Problem (CVRP)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch

from .read_config import READDecodeCfg, READDecodeResult, CVRP_READ_PRESET
from .read_competitive_neighbours import build_competitive_slot_route_neighbours
from .common import (
    StageTimer,
    build_seed_routes,
    nll_from_prob_and_labels,
    pyvrp_route_clients,
    row_normalize,
    run_pyvrp_ils,
    uncertainty_stats,
)



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
# Stage A: capacity projection
# ---------------------------------------------------------------------------

def _build_probability_vehicle_sets(prob: np.ndarray, cum_prob: float) -> List[np.ndarray]:
    """For each client, keep the smallest top-mass slot set reaching cum_prob."""
    prob = row_normalize(prob)
    n, k = prob.shape
    order = np.argsort(-prob, axis=1)
    p_sorted = np.take_along_axis(prob, order, axis=1)
    csum = np.cumsum(p_sorted, axis=1)
    sets: List[np.ndarray] = []
    for i in range(n):
        keep = int(np.searchsorted(csum[i], float(cum_prob), side="left") + 1)
        sets.append(order[i, : max(1, min(keep, k))].astype(np.int64))
    return sets


def stage_a_capacity_project(
    prob: np.ndarray,
    dem: np.ndarray,
    cap: float,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    *,
    proj_topk: int,
    cum_prob: float,
    lam_balance: float,
    lam_compact: float,
) -> np.ndarray:
    """Project client-slot probabilities to a capacity-feasible assignment."""
    prob = row_normalize(prob)
    dem = np.asarray(dem, dtype=np.float32).reshape(-1)
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)

    n, k = prob.shape
    if n <= 0 or k <= 0:
        raise ValueError(f"prob must have positive shape, got {prob.shape}")
    if dem.shape != (n,):
        raise ValueError(f"dem.shape={dem.shape}, expected {(n,)}")
    if clients_xy.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_xy.shape}, expected {(n, 2)}")
    if float(dem.sum()) > float(k) * float(cap) + 1e-6:
        raise ValueError(
            "CVRP instance is capacity-infeasible: total demand exceeds fleet capacity."
        )

    proj_topk = max(1, min(int(proj_topk), k))
    _, _, margin, ent = uncertainty_stats(prob)
    veh_sets = _build_probability_vehicle_sets(prob, cum_prob=float(cum_prob))

    dem_norm = dem / max(1e-6, float(cap))
    priority = dem_norm + 0.50 * margin - 0.25 * ent
    order = np.argsort(-priority)

    labels = np.full((n,), -1, dtype=np.int64)
    load = np.zeros((k,), dtype=np.float32)
    sum_xy = np.zeros((k, 2), dtype=np.float32)
    cnt = np.zeros((k,), dtype=np.int32)

    for i in order.tolist():
        demand_i = float(dem[i])
        candidates = [int(v) for v in veh_sets[i][:proj_topk].tolist()]
        feasible_candidates = [v for v in candidates if load[v] + demand_i <= float(cap) + 1e-6]
        if not feasible_candidates:
            feasible_candidates = [v for v in range(k) if load[v] + demand_i <= float(cap) + 1e-6]
        if not feasible_candidates:
            raise ValueError("Stage-A projection failed: no capacity-feasible slot remains.")

        def score(v: int) -> float:
            p_cost = float(-np.log(max(1e-12, float(prob[i, v]))))
            center = (sum_xy[v] / float(cnt[v])) if cnt[v] > 0 else depot_xy
            compact = float(np.linalg.norm(clients_xy[i] - center))
            balance = float(((load[v] + demand_i) / max(1e-6, float(cap))) ** 2)
            return p_cost + float(lam_compact) * compact + float(lam_balance) * balance

        best_v = min(feasible_candidates, key=score)
        labels[i] = int(best_v)
        load[best_v] += demand_i
        sum_xy[best_v] += clients_xy[i]
        cnt[best_v] += 1

    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Stage C: PyVRP problem construction and refinement
# ---------------------------------------------------------------------------

def _build_pyvrp_neighbours(
    prob: np.ndarray,
    clients_xy: np.ndarray,
    *,
    routes0_full: List[List[int]],
) -> Tuple[List[List[int]], Dict[str, object]]:
    rows, profile = build_competitive_slot_route_neighbours(
        prob_slot=prob,
        clients_xy=clients_xy,
        routes0_full=routes0_full,
        group_vec=None,
        slot_mask=None,
        defaults=CVRP_READ_PRESET.neighbours,
        enable_group_heat=False,
    )
    return [[]] + rows, profile


def _build_pyvrp_problem(
    dem: np.ndarray,
    cap: float,
    k: int,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
):
    """Build homogeneous CVRP ProblemData using the fixed CVRP READ preset."""
    from pyvrp import Model

    scale_dist = int(CVRP_READ_PRESET.pyvrp.dist_scale)
    scale_dem = int(CVRP_READ_PRESET.pyvrp.demand_scale)

    dem = np.asarray(dem, dtype=np.float64).reshape(-1)
    n = int(dem.size)
    depot_xy = np.asarray(depot_xy, dtype=np.float64).reshape(2)
    clients_xy = np.asarray(clients_xy, dtype=np.float64).reshape(n, 2)

    eps = 1e-9
    cap_i = max(1, int(np.ceil(float(cap) * float(scale_dem) - eps)))
    dem_i = np.maximum(
        0,
        np.floor(dem * float(scale_dem) + eps).astype(np.int64),
    )

    depot_i = np.round(depot_xy.reshape(1, 2) * float(scale_dist)).astype(np.int64)
    clients_i = np.round(clients_xy * float(scale_dist)).astype(np.int64)
    locs_i = np.vstack([depot_i, clients_i])

    diff = locs_i[:, None, :].astype(np.float64) - locs_i[None, :, :].astype(np.float64)
    dist_i = np.round(np.sqrt((diff * diff).sum(axis=2) + 1e-12)).astype(np.int64)

    model = Model()
    depot = model.add_depot(x=int(locs_i[0, 0]), y=int(locs_i[0, 1]))

    clients = [
        model.add_client(
            x=int(locs_i[i + 1, 0]),
            y=int(locs_i[i + 1, 1]),
            delivery=[int(dem_i[i])],
        )
        for i in range(n)
    ]

    model.add_vehicle_type(
        num_available=int(k),
        capacity=[int(cap_i)],
        unit_distance_cost=1,
    )

    locations = [depot] + clients
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            model.add_edge(
                frm,
                to,
                distance=int(dist_i[i, j]),
                duration=int(dist_i[i, j]),
            )

    return model.data()


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
    return_profile: bool = False,
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
    base_info = _require_feasible_routes(base_routes, dem, float(cap), num_clients=n, k_max=k, context="seed solution")
    base_cost = _cost_from_routes(base_routes, depot_xy, clients_xy)

    data = _build_pyvrp_problem(dem=dem, cap=float(cap), k=k, depot_xy=depot_xy, clients_xy=clients_xy)
    init_sol = Solution(data, base_routes)
    if not init_sol.is_complete():
        raise RuntimeError("PyVRP rejected the Stage-B seed solution as incomplete.")

    neigh, neigh_profile = _build_pyvrp_neighbours(prob=prob, clients_xy=clients_xy, routes0_full=routes0_by_slot)
    result = run_pyvrp_ils(data=data, init_sol=init_sol, neigh=neigh, seed=int(seed), budget_ms=float(budget_ms))

    best = result.best
    if not best.is_feasible():
        raise RuntimeError("PyVRP returned an infeasible best solution.")

    refined_routes = _routes_from_solution(best, n)
    refined_info = _require_feasible_routes(
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

    if return_profile:
        profile: Dict[str, object] = {
            "pyvrp_used": True,
            "pyvrp_accepted": accepted,
            "pyvrp_improved": bool(refined_cost + 1e-9 < float(base_cost)),
            "seed_used_routes": int(base_info["used_routes"]),
            "final_used_routes": int(refined_info["used_routes"] if accepted else base_info["used_routes"]),
        }
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
    cap,
    depot_xy,
    clients_xy,
    cfg: READDecodeCfg,
    *,
    seed: int = 0,
    return_profile: bool = False,
) -> READDecodeResult:
    """Decode a single CVRP instance from a client-to-slot posterior."""
    timer = StageTimer(enabled=bool(return_profile))
    profile: Dict[str, object] = {}

    prob_np = row_normalize(_as_numpy(prob_c, dtype=np.float32))
    dem_np = _as_numpy(dem, dtype=np.float32).reshape(-1)
    depot_np = _as_numpy(depot_xy, dtype=np.float32).reshape(2)
    clients_np = _as_numpy(clients_xy, dtype=np.float32)

    n, k = prob_np.shape
    if dem_np.shape != (n,):
        raise ValueError(f"dem.shape={dem_np.shape}, expected {(n,)}")
    if clients_np.shape != (n, 2):
        raise ValueError(f"clients_xy.shape={clients_np.shape}, expected {(n, 2)}")

    projection = CVRP_READ_PRESET.projection
    timer.tic("projector")
    seed_lab = stage_a_capacity_project(
        prob=prob_np,
        dem=dem_np,
        cap=float(cap),
        depot_xy=depot_np,
        clients_xy=clients_np,
        proj_topk=int(projection.topk),
        cum_prob=float(projection.cum_prob),
        lam_balance=float(projection.lam_balance),
        lam_compact=float(projection.lam_compact),
    )
    timer.toc("projector")

    nll_seed = nll_from_prob_and_labels(prob_np, seed_lab)

    route_cfg = CVRP_READ_PRESET.route
    timer.tic("seed_route")
    routes_seed = build_seed_routes(
        seed_lab,
        int(k),
        depot_np,
        clients_np,
        two_opt=bool(route_cfg.two_opt),
        two_opt_iter=int(route_cfg.two_opt_iter),
    )
    seed_info = _require_feasible_routes(
        routes_seed,
        dem_np,
        float(cap),
        num_clients=int(n),
        k_max=int(k),
        context="Stage-B seed solution",
    )
    stagea_cost = _cost_from_routes(routes_seed, depot_np, clients_np)
    timer.toc("seed_route")

    final_lab = seed_lab.copy()
    refined_cost = float(stagea_cost)

    if float(cfg.pyvrp_budget_ms) > 0.0:
        timer.tic("pyvrp")
        if return_profile:
            final_lab, refined_cost, routes_final, pyvrp_profile = pyvrp_refine(
                labels=seed_lab,
                prob=prob_np,
                dem=dem_np,
                cap=float(cap),
                k=int(k),
                depot_xy=depot_np,
                clients_xy=clients_np,
                routes0_full=routes_seed,
                budget_ms=float(cfg.pyvrp_budget_ms),
                seed=int(seed),
                return_profile=True,
            )
            profile.update(pyvrp_profile)
        else:
            final_lab, refined_cost, routes_final = pyvrp_refine(
                labels=seed_lab,
                prob=prob_np,
                dem=dem_np,
                cap=float(cap),
                k=int(k),
                depot_xy=depot_np,
                clients_xy=clients_np,
                routes0_full=routes_seed,
                budget_ms=float(cfg.pyvrp_budget_ms),
                seed=int(seed),
                return_profile=False,
            )
        timer.toc("pyvrp")
    else:
        routes_final = routes_seed
        if return_profile:
            profile["pyvrp_used"] = False
            profile["seed_used_routes"] = int(seed_info["used_routes"])
            profile["final_used_routes"] = int(seed_info["used_routes"])

    if return_profile:
        profile.update(timer.as_dict())

    nll_final = nll_from_prob_and_labels(prob_np, final_lab)
    return _make_result(
        prob_c=prob_c,
        seed_lab=seed_lab,
        final_lab=np.asarray(final_lab, dtype=np.int64),
        stagea_cost=stagea_cost,
        refined_cost=refined_cost,
        nll_seed=nll_seed,
        nll_final=nll_final,
        seed_routes=routes_seed,
        final_routes=routes_final,
        profile=profile if return_profile else None,
    )


def decode_read_batch_struct(
    jobs: List[dict],
    *,
    max_workers: Optional[int] = None,
    return_profile: bool = False,
) -> List[READDecodeResult]:
    """Decode a list of CVRP instances. Exceptions propagate to the caller."""
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 8)
    max_workers = max(1, int(max_workers))

    results: List[Optional[READDecodeResult]] = [None] * len(jobs)

    def worker(i: int, job: dict) -> Tuple[int, READDecodeResult]:
        return i, decode_read_struct(
            prob_c=job["prob_c"],
            dem=job["dem"],
            cap=job["cap"],
            depot_xy=job["depot_xy"],
            clients_xy=job["clients_xy"],
            cfg=job["cfg"],
            seed=int(job.get("seed", 0)),
            return_profile=bool(return_profile),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i, jobs[i]) for i in range(len(jobs))]
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

    if any(result is None for result in results):
        raise RuntimeError("decoder batch produced incomplete results.")
    return [result for result in results]  # after the check


__all__ = [
    "READDecodeCfg",
    "READDecodeResult",
    "decode_read_struct",
    "decode_read_batch_struct",
    "pyvrp_refine",
    "stage_a_capacity_project",
]
