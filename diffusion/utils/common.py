"""Shared primitives for the READ decoders.

This module groups numerical helpers, route construction utilities, and the
PyVRP search-operator wiring that are reused by both the homogeneous (CVRP)
and heterogeneous (HFVRP) READ decoders. Keeping these in one place makes
the per-problem decoders short and ensures CVRP and HFVRP behave identically
on the parts of the pipeline they share (Stage B route construction,
2-opt local moves, and the PyVRP local-search wrapper).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import os

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Probability post-processing
# ---------------------------------------------------------------------------

def row_normalize(prob: np.ndarray) -> np.ndarray:
    """Row-stochastic normalisation with a uniform fallback for degenerate rows.

    Rows whose mass is below ``1e-12`` are replaced with the uniform distribution
    so that downstream code can always treat ``prob`` as a valid posterior.
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob must be 2D, got shape={prob.shape}")
    out = prob.copy()
    s = out.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= 1e-12
    if np.any(bad):
        out[bad] = 1.0 / float(max(1, out.shape[1]))
        s = out.sum(axis=1, keepdims=True)
    return out / np.clip(s, 1e-12, None)


def nll_from_prob_and_labels(prob: np.ndarray, labels: np.ndarray) -> float:
    """Sum of negative log-likelihoods of the chosen slot per client."""
    if prob.size == 0:
        return 0.0
    p_pick = np.clip(prob[np.arange(prob.shape[0]), labels], 1e-12, None)
    return float(-np.log(p_pick).sum())


# ---------------------------------------------------------------------------
# Distance matrices and 2-opt
# ---------------------------------------------------------------------------

def prep_dist_mats(depot_xy: np.ndarray, clients_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute depot-client and client-client Euclidean distance matrices.

    A small ``1e-12`` is added inside the square root for numerical stability;
    it is far below the rounding precision used by PyVRP so it cannot affect
    feasibility or routing decisions.
    """
    depot_xy = np.asarray(depot_xy, dtype=np.float32).reshape(1, 2)
    clients_xy = np.asarray(clients_xy, dtype=np.float32)
    diff0 = clients_xy - depot_xy
    d0c = np.sqrt((diff0 * diff0).sum(axis=1) + 1e-12).astype(np.float32)
    diff = clients_xy[:, None, :] - clients_xy[None, :, :]
    dcc = np.sqrt((diff * diff).sum(axis=2) + 1e-12).astype(np.float32)
    return d0c, dcc


def two_opt_improve(route: np.ndarray, dist: np.ndarray) -> np.ndarray:
    r = np.asarray(route, dtype=np.int64).copy()
    m = int(r.size)
    if m <= 3:
        return r

    while True:
        improved = False

        for i in range(m - 2):
            a, b = r[i], r[i + 1]
            old_ab = dist[a, b]

            for j in range(i + 2, m - 1):
                c, d = r[j], r[j + 1]
                old = old_ab + dist[c, d]
                new = dist[a, c] + dist[b, d]

                if new + 1e-9 < old:
                    r[i + 1 : j + 1] = r[i + 1 : j + 1][::-1]
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return r

def as_numpy(x, *, dtype=None) -> np.ndarray:
    import torch

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def make_decode_result(
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
) -> Any:
    import torch
    from .read_config import READDecodeResult

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
    )


def build_read_neighbours(
    prob_slot: np.ndarray,
    clients_xy: np.ndarray,
    *,
    defaults: Any,
    dem: Optional[np.ndarray] = None,
    cap_vec: Optional[np.ndarray] = None,
    slot_mask: Optional[np.ndarray] = None,
    prepend_depot: bool = False,
):
    from .read_competitive_neighbours import build_heatmap_neighbours

    rows = build_heatmap_neighbours(
        prob_slot=prob_slot,
        clients_xy=clients_xy,
        dem=dem,
        cap_vec=cap_vec,
        slot_mask=slot_mask,
        defaults=defaults,
    )
    return [[]] + rows if prepend_depot else rows


def decode_jobs_in_threads(
    jobs: List[dict],
    worker: Callable[[int, dict], T],
    *,
    max_workers: Optional[int] = None,
    missing_message: str = "decoder batch produced incomplete results.",
) -> List[T]:
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 8)
    max_workers = max(1, int(max_workers))

    results: List[Optional[T]] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"{missing_message} first missing index={missing[0]}")
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Regret-based seed route construction (Stage B)
# ---------------------------------------------------------------------------

def _route_insert_delta(client: int, route: List[int], pos: int,
                        d0c: np.ndarray, dcc: np.ndarray) -> float:
    """Cost increase of inserting ``client`` at index ``pos`` in ``route``."""
    if len(route) == 0:
        return float(2.0 * d0c[client])
    if pos == 0:
        nxt = route[0]
        return float(d0c[client] + dcc[client, nxt] - d0c[nxt])
    if pos == len(route):
        prv = route[-1]
        return float(dcc[prv, client] + d0c[client] - d0c[prv])
    prv = route[pos - 1]
    nxt = route[pos]
    return float(dcc[prv, client] + dcc[client, nxt] - dcc[prv, nxt])


def _best_two_insertions(client: int, route: List[int],
                         d0c: np.ndarray, dcc: np.ndarray) -> Tuple[float, int, float]:
    """Best and second-best insertion deltas for ``client`` into ``route``."""
    best_delta = float("inf")
    best_pos = 0
    second_delta = float("inf")
    for pos in range(len(route) + 1):
        delta = _route_insert_delta(client, route, pos, d0c, dcc)
        if delta < best_delta:
            second_delta = best_delta
            best_delta = delta
            best_pos = pos
        elif delta < second_delta:
            second_delta = delta
    if not np.isfinite(second_delta):
        second_delta = best_delta
    return float(best_delta), int(best_pos), float(second_delta)


def _choose_regret_seed_pair(sub: np.ndarray, d0c: np.ndarray, dcc: np.ndarray) -> List[int]:
    """Pick the route's first two clients: the farthest from the depot, then
    the client farthest from that one."""
    sub = np.asarray(sub, dtype=np.int64).reshape(-1)
    if sub.size == 0:
        return []
    if sub.size == 1:
        return [int(sub[0])]
    a = int(sub[np.argmax(d0c[sub])])
    rem = sub[sub != a]
    b = int(rem[np.argmax(dcc[a, rem])])
    return [a, b]


def regret_insertion_order(sub: np.ndarray, d0c: np.ndarray, dcc: np.ndarray) -> np.ndarray:
    """Build a single route over ``sub`` by greedy regret-2 insertion."""
    sub = np.asarray(sub, dtype=np.int64).reshape(-1)
    if sub.size <= 1:
        return sub.copy()

    route = _choose_regret_seed_pair(sub, d0c, dcc)
    used = set(int(x) for x in route)
    remaining = [int(x) for x in sub.tolist() if int(x) not in used]

    while remaining:
        best_choice = None
        for client in remaining:
            best_delta, best_pos, second_delta = _best_two_insertions(client, route, d0c, dcc)
            regret = float(second_delta - best_delta)
            cand = (-regret, float(best_delta), -float(d0c[client]), int(client), int(best_pos))
            if best_choice is None or cand < best_choice:
                best_choice = cand
        _, _, _, client, best_pos = best_choice
        route.insert(int(best_pos), int(client))
        remaining.remove(int(client))

    return np.asarray(route, dtype=np.int64)

def build_seed_routes(
    labels: np.ndarray,
    k: int,
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
) -> List[List[int]]:
    """Build one deterministic route per slot using regret insertion and 2-opt."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    k = int(k)
    if k <= 0:
        return []

    d0c, dcc = prep_dist_mats(depot_xy, clients_xy)
    routes: List[List[int]] = [[] for _ in range(k)]

    for v in range(k):
        sub = np.nonzero(labels == v)[0].astype(np.int64)
        if sub.size == 0:
            continue

        order = regret_insertion_order(sub, d0c, dcc)
        if order.size >= 4:
            order = two_opt_improve(order, dcc)

        routes[v] = (order + 1).astype(np.int64).tolist()

    return routes

# ---------------------------------------------------------------------------
# PyVRP solution decoding and search-operator wiring
# ---------------------------------------------------------------------------

def pyvrp_route_clients(route, num_clients: int) -> List[int]:
    """Extract 1-based client ids from a PyVRP 0.13.3 Route."""
    visits = [int(cid) for cid in route.visits()]
    out: List[int] = []

    for cid in visits:
        if 1 <= cid <= int(num_clients):
            out.append(cid)
        else:
            raise ValueError(f"route contains client id {cid} outside 1..{num_clients}")

    return out


# Operator names tried during PyVRP local-search wiring. Names that are not
# present in the installed PyVRP version are silently skipped, so the same
# list is safe across PyVRP minor versions.






def add_pyvrp_operators(ls, data) -> None:
    from pyvrp.search import NODE_OPERATORS

    added = 0
    for op_cls in NODE_OPERATORS:
        ls.add_node_operator(op_cls(data))
        added += 1

    if added == 0:
        raise RuntimeError("No PyVRP local-search operators were added.")


def run_pyvrp_ils(data, init_sol, neigh, *, seed: int = 0, budget_ms: float = 10.0):
    """Run PyVRP's iterated local search with the supplied neighbour lists
    and a runtime budget given in milliseconds."""
    from pyvrp import IteratedLocalSearch, PenaltyManager, RandomNumberGenerator
    from pyvrp.search import LocalSearch
    from pyvrp.stop import MaxRuntime

    rng = RandomNumberGenerator(seed=int(seed))
    ls = LocalSearch(data, rng, neigh)
    add_pyvrp_operators(ls, data)
    pen_manager = PenaltyManager.init_from(data)
    algo = IteratedLocalSearch(data, pen_manager, rng, ls, init_sol)
    stop = MaxRuntime(max(1e-3, float(budget_ms) / 1000.0))
    return algo.run(stop)
