"""Common PyVRP component runner for HFVRP refinement.

This module isolates the PyVRP part of the HFVRP decoder.  The intended
contract is simple:

1. Build a PyVRP ProblemData from HFVRP arrays.
2. Build a warm-start Solution from slot-level routes.
3. Build either the official PyVRP neighbourhood or a custom neighbourhood.
4. Run PyVRP's component-level ILS with a fixed operator/penalty/search setup.
5. Convert PyVRP's type-level routes back to slot-level routes/labels.

Only the initial solution and the neighbourhood graph should be swapped by the
neural decoder.  The rest of the runner stays close to PyVRP's official
component workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import time
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HFPyVRPBuildConfig:
    """Scaling and construction settings for a PyVRP HFVRP instance."""

    dist_scale: int = 10_000
    demand_scale: int = 1_000
    cost_scale: float = 1_000.0
    min_capacity: int = 1
    min_unit_distance_cost: int = 1


@dataclass(frozen=True)
class HFPyVRPNeighbourConfig:
    """Official neighbourhood parameters used by ``compute_neighbours``."""

    num_neighbours: int = 50
    weight_wait_time: float = 0.2
    symmetric_proximity: bool = True


@dataclass(frozen=True)
class HFPyVRPILSConfig:
    """Component-level ILS settings for PyVRP 0.13.3."""

    budget_ms: float = 1_000.0
    seed: int = 0
    collect_stats: bool = True
    display: bool = False
    display_interval: float = 5.0
    exhaustive_on_best: Optional[bool] = None
    history_length: Optional[int] = None
    num_iters_no_improvement: Optional[int] = None


@dataclass
class HFPyVRPContext:
    """All metadata needed to map between HF slots and PyVRP vehicle types."""

    data: Any
    tier_vec: np.ndarray
    slot_mask: np.ndarray
    active_slots: np.ndarray
    uniq_tiers: List[int]
    tier_to_compact: Dict[int, int]
    compact_to_tier: Dict[int, int]
    slots_by_tier: Dict[int, np.ndarray]
    depot_xy: np.ndarray
    clients_xy: np.ndarray
    dem: np.ndarray
    cap_vec: np.ndarray
    fixed_vec: np.ndarray
    unit_cost_vec: np.ndarray
    build_config: HFPyVRPBuildConfig
    int_distance_matrix: np.ndarray


@dataclass
class HFPyVRPResult:
    """Normalised result returned by the common runner."""

    status: str
    solution: Any = None
    pyvrp_result: Any = None
    routes_by_slot: Optional[List[List[int]]] = None
    labels: Optional[np.ndarray] = None
    exact_cost: Optional[float] = None
    pyvrp_cost: Optional[float] = None
    is_feasible: Optional[bool] = None
    elapsed_ms: float = 0.0
    operators: List[str] = field(default_factory=list)
    neighbourhood_stats: Dict[str, float] = field(default_factory=dict)
    exception_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Basic HFVRP helpers
# ---------------------------------------------------------------------------


def infer_tier_vec(
    cap_vec: Sequence[float],
    fixed_vec: Sequence[float],
    unit_cost_vec: Sequence[float],
) -> np.ndarray:
    """Infer vehicle tiers from equal capacity/fixed/unit-cost signatures."""

    cap = np.asarray(cap_vec, dtype=np.float64).reshape(-1)
    fixed = np.asarray(fixed_vec, dtype=np.float64).reshape(-1)
    unit = np.asarray(unit_cost_vec, dtype=np.float64).reshape(-1)
    if not (cap.size == fixed.size == unit.size):
        raise ValueError("cap_vec, fixed_vec, and unit_cost_vec must have equal length")

    sig = np.stack([np.round(cap, 10), np.round(fixed, 10), np.round(unit, 10)], axis=1)
    _, inv = np.unique(sig, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def slot_groups_from_tier(
    tier_vec: Sequence[int],
    slot_mask: Optional[Sequence[bool]] = None,
) -> Dict[int, np.ndarray]:
    """Return active slot indices for each vehicle tier."""

    tiers = np.asarray(tier_vec, dtype=np.int64).reshape(-1)
    if slot_mask is None:
        mask = np.ones(tiers.size, dtype=np.bool_)
    else:
        mask = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
        if mask.size != tiers.size:
            raise ValueError("slot_mask and tier_vec must have equal length")

    groups: Dict[int, np.ndarray] = {}
    for tier in sorted(np.unique(tiers[mask]).tolist()):
        idx = np.where((tiers == int(tier)) & mask)[0].astype(np.int64)
        if idx.size > 0:
            groups[int(tier)] = idx
    return groups


def exact_hf_cost(
    routes_by_slot: Sequence[Sequence[int]],
    depot_xy: Sequence[float],
    clients_xy: Sequence[Sequence[float]],
    fixed_vec: Sequence[float],
    unit_cost_vec: Sequence[float],
) -> float:
    """Compute the external HFVRP objective on slot-level routes.

    Routes use PyVRP-style 1-based client ids.
    """

    depot = np.asarray(depot_xy, dtype=np.float64).reshape(2)
    clients = np.asarray(clients_xy, dtype=np.float64).reshape(-1, 2)
    fixed = np.asarray(fixed_vec, dtype=np.float64).reshape(-1)
    unit = np.asarray(unit_cost_vec, dtype=np.float64).reshape(-1)

    total = 0.0
    k = min(len(routes_by_slot), fixed.size, unit.size)
    for slot in range(k):
        route = [int(x) for x in routes_by_slot[slot] if int(x) > 0]
        if not route:
            continue
        idx = np.asarray(route, dtype=np.int64) - 1
        if np.any(idx < 0) or np.any(idx >= clients.shape[0]):
            raise ValueError(f"route for slot {slot} contains client id outside 1..n")
        pts = clients[idx]
        dist = float(np.linalg.norm(depot - pts[0]))
        if len(route) >= 2:
            dist += float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())
        dist += float(np.linalg.norm(pts[-1] - depot))
        total += float(fixed[slot]) + float(unit[slot]) * dist

    return float(total)


def labels_from_slot_routes(routes_by_slot: Sequence[Sequence[int]], num_clients: int) -> np.ndarray:
    """Convert slot-level 1-based routes to a client -> slot label vector."""

    labels = np.full((int(num_clients),), -1, dtype=np.int64)
    for slot, route in enumerate(routes_by_slot):
        for client in route:
            cid = int(client)
            if 1 <= cid <= num_clients:
                if labels[cid - 1] != -1:
                    raise ValueError(f"client {cid} appears more than once")
                labels[cid - 1] = int(slot)

    if np.any(labels < 0):
        missing = np.where(labels < 0)[0][:10] + 1
        raise ValueError(f"incomplete assignment; first missing clients: {missing.tolist()}")
    return labels


def canonicalize_routes_by_tier(
    routes_by_slot: Sequence[Sequence[int]],
    tier_vec: Sequence[int],
    slot_mask: Optional[Sequence[bool]] = None,
) -> List[List[int]]:
    """Canonicalise routes among exchangeable slots of the same tier.

    PyVRP solves at the vehicle-type level.  This helper gives deterministic
    slot-level output by sorting non-empty routes within each tier.
    """

    tiers = np.asarray(tier_vec, dtype=np.int64).reshape(-1)
    groups = slot_groups_from_tier(tiers, slot_mask=slot_mask)
    out: List[List[int]] = [[] for _ in range(len(routes_by_slot))]

    for tier in sorted(groups):
        slots = [int(s) for s in groups[tier].tolist()]
        bucket = [list(map(int, routes_by_slot[s])) for s in slots if len(routes_by_slot[s]) > 0]
        bucket.sort(key=lambda r: (len(r), r[0] if r else 10**12, int(np.sum(r)) if r else 0))
        for pos, route in enumerate(bucket[: len(slots)]):
            out[slots[pos]] = route
    return out


# ---------------------------------------------------------------------------
# PyVRP ProblemData / Solution construction
# ---------------------------------------------------------------------------


def _to_int_distance_matrix(
    depot_xy: np.ndarray,
    clients_xy: np.ndarray,
    dist_scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    depot_i = np.round(depot_xy.reshape(1, 2) * float(dist_scale)).astype(np.int64)
    clients_i = np.round(clients_xy * float(dist_scale)).astype(np.int64)
    locs_i = np.vstack([depot_i, clients_i])
    diff = locs_i[:, None, :] - locs_i[None, :, :]
    dist = np.sqrt((diff.astype(np.float64) ** 2).sum(axis=2) + 1e-12)
    return locs_i, np.round(dist).astype(np.int64)


def build_hf_problem(
    *,
    dem: Sequence[float],
    cap_vec: Sequence[float],
    fixed_vec: Sequence[float],
    unit_cost_vec: Sequence[float],
    depot_xy: Sequence[float],
    clients_xy: Sequence[Sequence[float]],
    tier_vec: Optional[Sequence[int]] = None,
    slot_mask: Optional[Sequence[bool]] = None,
    config: HFPyVRPBuildConfig = HFPyVRPBuildConfig(),
) -> HFPyVRPContext:
    """Build a PyVRP 0.13.3 HFVRP ProblemData from slot-level arrays."""

    from pyvrp import Model

    dem_np = np.asarray(dem, dtype=np.float64).reshape(-1)
    cap_np = np.asarray(cap_vec, dtype=np.float64).reshape(-1)
    fixed_np = np.asarray(fixed_vec, dtype=np.float64).reshape(-1)
    unit_np = np.asarray(unit_cost_vec, dtype=np.float64).reshape(-1)
    depot_np = np.asarray(depot_xy, dtype=np.float64).reshape(2)
    clients_np = np.asarray(clients_xy, dtype=np.float64).reshape(-1, 2)

    if clients_np.shape[0] != dem_np.size:
        raise ValueError("clients_xy and dem must describe the same number of clients")
    if not (cap_np.size == fixed_np.size == unit_np.size):
        raise ValueError("cap_vec, fixed_vec, and unit_cost_vec must have equal length")

    k = int(cap_np.size)
    if tier_vec is None:
        tier_np = infer_tier_vec(cap_np, fixed_np, unit_np)
    else:
        tier_np = np.asarray(tier_vec, dtype=np.int64).reshape(-1)
        if tier_np.size != k:
            raise ValueError("tier_vec length must equal number of vehicle slots")

    if slot_mask is None:
        mask_np = np.ones((k,), dtype=np.bool_)
    else:
        mask_np = np.asarray(slot_mask, dtype=np.bool_).reshape(-1)
        if mask_np.size != k:
            raise ValueError("slot_mask length must equal number of vehicle slots")

    active_slots = np.where(mask_np)[0].astype(np.int64)
    if active_slots.size == 0:
        raise ValueError("at least one active vehicle slot is required")

    slots_by_tier = slot_groups_from_tier(tier_np, mask_np)
    uniq_tiers = sorted(slots_by_tier.keys())
    tier_to_compact = {int(t): idx for idx, t in enumerate(uniq_tiers)}
    compact_to_tier = {idx: int(t) for t, idx in tier_to_compact.items()}

    eps = 1e-9
    dem_i = np.maximum(
        0,
        np.floor(dem_np * float(config.demand_scale) + eps).astype(np.int64),
    )

    locs_i, dist_i = _to_int_distance_matrix(
        depot_np,
        clients_np,
        int(config.dist_scale),
    )

    model = Model()
    depot = model.add_depot(x=int(locs_i[0, 0]), y=int(locs_i[0, 1]))

    clients = [
        model.add_client(
            x=int(locs_i[i + 1, 0]),
            y=int(locs_i[i + 1, 1]),
            delivery=[int(dem_i[i])],
        )
        for i in range(int(dem_i.size))
    ]

    for tier in uniq_tiers:
        slots = slots_by_tier[int(tier)]
        rep = int(slots[0])
        cap_i = max(
            int(config.min_capacity),
            int(math.ceil(float(cap_np[rep]) * float(config.demand_scale) - eps)),
        )
        fixed_i = max(
            0,
            int(round(float(fixed_np[rep]) * float(config.dist_scale) * float(config.cost_scale))),
        )
        unit_i = max(
            int(config.min_unit_distance_cost),
            int(round(float(unit_np[rep]) * float(config.cost_scale))),
        )
        model.add_vehicle_type(
            num_available=int(slots.size),
            capacity=[cap_i],
            fixed_cost=fixed_i,
            unit_distance_cost=unit_i,
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

    data = model.data()

    return HFPyVRPContext(
        data=data,
        tier_vec=tier_np,
        slot_mask=mask_np,
        active_slots=active_slots,
        uniq_tiers=list(map(int, uniq_tiers)),
        tier_to_compact=tier_to_compact,
        compact_to_tier=compact_to_tier,
        slots_by_tier=slots_by_tier,
        depot_xy=depot_np,
        clients_xy=clients_np,
        dem=dem_np,
        cap_vec=cap_np,
        fixed_vec=fixed_np,
        unit_cost_vec=unit_np,
        build_config=config,
        int_distance_matrix=dist_i,
    )

def build_solution_from_slot_routes(
    ctx: HFPyVRPContext,
    routes_by_slot: Sequence[Sequence[int]],
    *,
    require_complete: bool = True,
):
    """Build a PyVRP Solution from slot-level 1-based routes."""

    from pyvrp import Route, Solution

    if len(routes_by_slot) != int(ctx.tier_vec.size):
        raise ValueError("routes_by_slot length must equal number of vehicle slots")

    routes = []
    for slot in ctx.active_slots.tolist():
        route = [int(x) for x in routes_by_slot[int(slot)] if int(x) > 0]
        if not route:
            continue
        tier = int(ctx.tier_vec[int(slot)])
        vehicle_type = int(ctx.tier_to_compact[tier])
        routes.append(Route(ctx.data, route, vehicle_type=vehicle_type))

    sol = Solution(ctx.data, routes)
    if require_complete and not bool(sol.is_complete()):
        raise ValueError("seed PyVRP solution is incomplete")
    return sol


# ---------------------------------------------------------------------------
# Neighbourhood and LocalSearch setup
# ---------------------------------------------------------------------------


def build_official_neighbours(
    data: Any,
    config: HFPyVRPNeighbourConfig = HFPyVRPNeighbourConfig(),
) -> List[List[int]]:
    """Build PyVRP's official granular neighbourhood via compute_neighbours."""

    from pyvrp.search import NeighbourhoodParams, compute_neighbours

    params = NeighbourhoodParams(
        weight_wait_time=float(config.weight_wait_time),
        num_neighbours=int(config.num_neighbours),
        symmetric_proximity=bool(config.symmetric_proximity),
    )
    return compute_neighbours(data, params)


def normalise_custom_neighbours(
    neighbours: Sequence[Sequence[int]],
    *,
    num_clients: int,
) -> List[List[int]]:
    """Convert decoder-side n-row 1-based neighbours to PyVRP n+1 rows."""

    n = int(num_clients)
    rows = list(neighbours)
    if len(rows) != n:
        raise ValueError(f"expected n neighbour rows, got {len(rows)} for n={n}")

    out: List[List[int]] = [[] for _ in range(n + 1)]
    for i0, row in enumerate(rows):
        self_id = i0 + 1
        seen = set()
        clean: List[int] = []
        for val in row:
            cid = int(val)
            if cid == self_id or cid < 1 or cid > n or cid in seen:
                continue
            seen.add(cid)
            clean.append(cid)
        out[self_id] = clean
    return out


def merge_neighbour_lists(
    primary: Sequence[Sequence[int]],
    secondary: Sequence[Sequence[int]],
    *,
    num_clients: int,
    primary_quota: Optional[int] = None,
    secondary_quota: Optional[int] = None,
    total_limit: Optional[int] = None,
) -> List[List[int]]:
    """Merge two 1-based neighbour graphs while preserving order."""

    p = normalise_custom_neighbours(primary, num_clients=num_clients)
    s = normalise_custom_neighbours(secondary, num_clients=num_clients)
    n = int(num_clients)
    out: List[List[int]] = [[] for _ in range(n + 1)]

    for cid in range(1, n + 1):
        merged: List[int] = []
        seen = set()

        def add_many(vals: Sequence[int], quota: Optional[int]) -> None:
            count = 0
            for x in vals:
                if quota is not None and count >= int(quota):
                    break
                xx = int(x)
                if xx == cid or xx < 1 or xx > n or xx in seen:
                    continue
                if total_limit is not None and len(merged) >= int(total_limit):
                    break
                seen.add(xx)
                merged.append(xx)
                count += 1

        add_many(p[cid], primary_quota)
        add_many(s[cid], secondary_quota)
        out[cid] = merged

    return out


def neighbour_stats(neighbours: Sequence[Sequence[int]], *, num_clients: Optional[int] = None) -> Dict[str, float]:
    """Return compact neighbourhood diagnostics."""

    rows = list(neighbours)
    if num_clients is not None and len(rows) == int(num_clients) + 1:
        rows = rows[1:]
    sizes = np.asarray([len(row) for row in rows], dtype=np.float64)
    if sizes.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "zero": 0.0, "nodes": 0.0}
    return {
        "min": float(sizes.min()),
        "mean": float(sizes.mean()),
        "max": float(sizes.max()),
        "zero": float(np.count_nonzero(sizes == 0)),
        "nodes": float(sizes.size),
    }


def official_operator_classes() -> List[Any]:
    """Return PyVRP 0.13.3 official local-search operators."""
    from pyvrp.search import NODE_OPERATORS

    return list(NODE_OPERATORS)

def add_operators_to_local_search(
    ls: Any,
    data: Any,
    operator_classes: Optional[Iterable[Any]] = None,
) -> List[str]:
    """Add PyVRP 0.13.3 node operators to LocalSearch."""
    ops = list(operator_classes) if operator_classes is not None else official_operator_classes()
    added: List[str] = []

    for op_cls in ops:
        name = getattr(op_cls, "__name__", str(op_cls))
        ls.add_node_operator(op_cls(data))
        added.append(str(name))

    if not added:
        raise RuntimeError("No PyVRP local-search operators were added.")
    return added


def make_local_search(
    data: Any,
    rng: Any,
    neighbours: Sequence[Sequence[int]],
    *,
    operator_classes: Optional[Iterable[Any]] = None,
) -> Tuple[Any, List[str]]:
    """Build LocalSearch with the supplied neighbourhood and official operators."""

    from pyvrp.search import LocalSearch

    ls = LocalSearch(data, rng, [list(map(int, row)) for row in neighbours])
    operators = add_operators_to_local_search(ls, data, operator_classes=operator_classes)
    return ls, operators


# ---------------------------------------------------------------------------
# ILS runner
# ---------------------------------------------------------------------------


def _make_ils_params(config: HFPyVRPILSConfig) -> Any:
    from pyvrp import IteratedLocalSearchParams

    kwargs: Dict[str, Any] = {}
    if config.num_iters_no_improvement is not None:
        kwargs["num_iters_no_improvement"] = int(config.num_iters_no_improvement)
    if config.history_length is not None:
        kwargs["history_length"] = int(config.history_length)
    if config.exhaustive_on_best is not None:
        kwargs["exhaustive_on_best"] = bool(config.exhaustive_on_best)

    return IteratedLocalSearchParams(**kwargs)

def run_component_ils(
    *,
    data: Any,
    init_sol: Any,
    neighbours: Sequence[Sequence[int]],
    config: HFPyVRPILSConfig = HFPyVRPILSConfig(),
    operator_classes: Optional[Iterable[Any]] = None,
) -> HFPyVRPResult:
    """Run PyVRP 0.13.3 component-level ILS."""

    from pyvrp import IteratedLocalSearch, PenaltyManager, RandomNumberGenerator
    from pyvrp.stop import MaxRuntime

    t0 = time.perf_counter()
    try:
        rng = RandomNumberGenerator(seed=int(config.seed))
        ls, added_ops = make_local_search(
            data,
            rng,
            neighbours,
            operator_classes=operator_classes,
        )
        penalty_manager = PenaltyManager.init_from(data)
        params = _make_ils_params(config)
        algo = IteratedLocalSearch(data, penalty_manager, rng, ls, init_sol, params)
        stop = MaxRuntime(max(1e-3, float(config.budget_ms) / 1000.0))

        result = algo.run(
            stop=stop,
            collect_stats=bool(config.collect_stats),
            display=bool(config.display),
            display_interval=float(config.display_interval),
        )
        best = result.best

        return HFPyVRPResult(
            status="ok",
            solution=best,
            pyvrp_result=result,
            pyvrp_cost=float(result.cost()),
            is_feasible=bool(result.is_feasible()),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            operators=added_ops,
            neighbourhood_stats=neighbour_stats(neighbours),
        )
    except Exception:
        return HFPyVRPResult(
            status="exception",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            neighbourhood_stats=neighbour_stats(neighbours),
            exception_text=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Solution extraction and high-level refinement wrapper
# ---------------------------------------------------------------------------


def route_clients_1based(route: Any, num_clients: int) -> List[int]:
    """Extract 1-based client ids from a PyVRP 0.13.3 Route."""
    visits = [int(cid) for cid in route.visits()]
    out: List[int] = []

    for cid in visits:
        if 1 <= cid <= int(num_clients):
            out.append(cid)
        else:
            raise ValueError(f"route contains client id {cid} outside 1..{num_clients}")

    return out


def extract_slot_routes_from_solution(
    ctx: HFPyVRPContext,
    solution: Any,
    *,
    canonicalize: bool = True,
) -> List[List[int]]:
    """Convert a PyVRP 0.13.3 type-level solution to slot-level routes."""

    k = int(ctx.tier_vec.size)
    n = int(ctx.clients_xy.shape[0])
    routes_by_compact: Dict[int, List[List[int]]] = {
        compact: [] for compact in range(len(ctx.uniq_tiers))
    }

    for route in solution.routes():
        vt = int(route.vehicle_type())
        visits = route_clients_1based(route, n)
        if visits:
            routes_by_compact[vt].append(visits)

    routes_by_slot: List[List[int]] = [[] for _ in range(k)]
    for compact, tier_routes in routes_by_compact.items():
        tier = int(ctx.compact_to_tier[int(compact)])
        slots = [int(s) for s in ctx.slots_by_tier[tier].tolist()]
        for pos, route in enumerate(tier_routes[: len(slots)]):
            routes_by_slot[slots[pos]] = list(route)

    if canonicalize:
        routes_by_slot = canonicalize_routes_by_tier(
            routes_by_slot,
            ctx.tier_vec,
            ctx.slot_mask,
        )
    return routes_by_slot


def refine_hf_with_component_ils(
    *,
    routes0_by_slot: Sequence[Sequence[int]],
    dem: Sequence[float],
    cap_vec: Sequence[float],
    fixed_vec: Sequence[float],
    unit_cost_vec: Sequence[float],
    depot_xy: Sequence[float],
    clients_xy: Sequence[Sequence[float]],
    tier_vec: Optional[Sequence[int]] = None,
    slot_mask: Optional[Sequence[bool]] = None,
    neighbours: Optional[Sequence[Sequence[int]]] = None,
    build_config: HFPyVRPBuildConfig = HFPyVRPBuildConfig(),
    neighbour_config: HFPyVRPNeighbourConfig = HFPyVRPNeighbourConfig(),
    ils_config: HFPyVRPILSConfig = HFPyVRPILSConfig(),
    canonicalize: bool = True,
) -> HFPyVRPResult:
    """End-to-end HFVRP PyVRP refinement with a pluggable neighbourhood.

    If ``neighbours`` is ``None``, PyVRP's official ``compute_neighbours`` output
    is used.  Otherwise the provided graph is normalised and injected into
    ``LocalSearch``.
    """

    t0 = time.perf_counter()
    try:
        ctx = build_hf_problem(
            dem=dem,
            cap_vec=cap_vec,
            fixed_vec=fixed_vec,
            unit_cost_vec=unit_cost_vec,
            depot_xy=depot_xy,
            clients_xy=clients_xy,
            tier_vec=tier_vec,
            slot_mask=slot_mask,
            config=build_config,
        )
        init_sol = build_solution_from_slot_routes(ctx, routes0_by_slot, require_complete=True)

        if neighbours is None:
            neigh = build_official_neighbours(ctx.data, neighbour_config)
        else:
            neigh = normalise_custom_neighbours(neighbours, num_clients=int(ctx.clients_xy.shape[0]))

        run = run_component_ils(
            data=ctx.data,
            init_sol=init_sol,
            neighbours=neigh,
            config=ils_config,
        )
        run.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        run.neighbourhood_stats = neighbour_stats(neigh, num_clients=int(ctx.clients_xy.shape[0]))
        if run.status != "ok" or run.solution is None:
            return run

        if run.is_feasible is False:
            run.status = "solver_infeasible"
            return run

        routes = extract_slot_routes_from_solution(ctx, run.solution, canonicalize=canonicalize)
        labels = labels_from_slot_routes(routes, num_clients=int(ctx.clients_xy.shape[0]))
        exact = exact_hf_cost(routes, ctx.depot_xy, ctx.clients_xy, ctx.fixed_vec, ctx.unit_cost_vec)

        run.routes_by_slot = routes
        run.labels = labels
        run.exact_cost = float(exact)
        run.status = "ok_feasible" if run.is_feasible else "ok_unknown_feasibility"
        return run
    except Exception:
        return HFPyVRPResult(
            status="exception",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            exception_text=traceback.format_exc(),
        )


__all__ = [
    "HFPyVRPBuildConfig",
    "HFPyVRPNeighbourConfig",
    "HFPyVRPILSConfig",
    "HFPyVRPContext",
    "HFPyVRPResult",
    "infer_tier_vec",
    "slot_groups_from_tier",
    "exact_hf_cost",
    "labels_from_slot_routes",
    "canonicalize_routes_by_tier",
    "build_hf_problem",
    "build_solution_from_slot_routes",
    "normalise_custom_neighbours",
    "neighbour_stats",
    "extract_slot_routes_from_solution",
    "refine_hf_with_component_ils",
]