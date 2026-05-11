from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from .read_competitive_neighbours import CompetitiveNeighbourDefaults


@dataclass(frozen=True, slots=True)
class READDecodeCfg:
    """Public READ decoder configuration.

    The paper release exposes only the PyVRP refinement budget.
    All other constants are fixed by the READ decoding protocol.
    """

    pyvrp_budget_ms: float = 1000.0


@dataclass(frozen=True, slots=True)
class ProjectionDefaults:
    topk: int = 4
    cum_prob: float = 0.95
    lam_balance: float = 0.10
    lam_compact: float = 0.15


@dataclass(frozen=True, slots=True)
class RouteDefaults:
    two_opt: bool = True
    two_opt_iter: int = 128


@dataclass(frozen=True, slots=True)
class PyVRPDefaults:
    dist_scale: int = 10000
    demand_scale: int = 1000
    cost_scale: float = 1000.0


@dataclass(frozen=True, slots=True)
class HFProjectorDefaults:
    lam_prob: float = 1.0
    lam_fixed: float = 0.05
    lam_econ: float = 0.10
    lam_load: float = 0.10
    dual_iters: int = 8
    dual_step0: float = 1.0


@dataclass(frozen=True, slots=True)
class READPreset:
    projection: ProjectionDefaults
    route: RouteDefaults
    pyvrp: PyVRPDefaults
    neighbours: CompetitiveNeighbourDefaults
    hf_projector: Optional[HFProjectorDefaults] = None

@dataclass(slots=True)
class READDecodeResult:
    seed_lab_t: torch.Tensor
    final_lab_t: torch.Tensor
    stagea_cost: float
    refined_cost: float
    nll_seed: float
    nll_final: float
    seed_routes: Optional[List[List[int]]] = None
    final_routes: Optional[List[List[int]]] = None
    profile: Optional[Dict[str, object]] = None


CVRP_READ_PRESET = READPreset(
    projection=ProjectionDefaults(
        topk=4,
        cum_prob=0.95,
        lam_balance=0.10,
        lam_compact=0.15,
    ),
    route=RouteDefaults(two_opt=True, two_opt_iter=128),
    pyvrp=PyVRPDefaults(dist_scale=10000, demand_scale=1000),
    neighbours=CompetitiveNeighbourDefaults(
        max_neigh=50,
        geo_core_k=28,
        geo_pool_k=50,
        geom_shortlist=96,
        group_heat_k=0,
        group_heat_k_low=0,
        route_cover_top_slots=6,
        route_cover_top_slots_low=8,
        route_cover_per_slot=3,
        symmetric=True,
        slot_sim_weight=0.0,
        group_sim_weight=0.0,
        dist_penalty=1.0,
    ),
)


HFVRP_READ_PRESET = READPreset(
    projection=ProjectionDefaults(
        topk=4,
        cum_prob=0.95,
        lam_balance=0.0,
        lam_compact=0.0,
    ),
    route=RouteDefaults(two_opt=True, two_opt_iter=128),
    pyvrp=PyVRPDefaults(dist_scale=10000, demand_scale=1000, cost_scale=1000.0),
    neighbours=CompetitiveNeighbourDefaults(
        max_neigh=50,
        geo_core_k=28,
        geo_pool_k=50,
        geom_shortlist=96,
        group_heat_k=12,
        group_heat_k_low=16,
        route_cover_top_slots=5,
        route_cover_top_slots_low=6,
        route_cover_per_slot=2,
        symmetric=True,
        slot_sim_weight=0.0,
        group_sim_weight=0.25,
        dist_penalty=1.0,
    ),
    hf_projector=HFProjectorDefaults(
        lam_prob=1.0,
        lam_fixed=0.05,
        lam_econ=0.10,
        lam_load=0.10,
    ),
)