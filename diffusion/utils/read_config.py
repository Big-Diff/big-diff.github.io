from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .read_competitive_neighbours import HeatmapNeighbourDefaults


@dataclass(frozen=True, slots=True)
class READDecodeCfg:
    """Public READ decoder configuration."""

    pyvrp_budget_ms: float = 1000.0
    top_slot_k: int = 6
    lambda_prob: float = 1.0
    lambda_insert: float = 0.20
    lambda_open: float = 0.05
    lambda_fill: float = 0.05


@dataclass(frozen=True, slots=True)
class PyVRPDefaults:
    dist_scale: int = 10000
    demand_scale: int = 1000
    cost_scale: float = 1000.0


@dataclass(frozen=True, slots=True)
class READPreset:
    pyvrp: PyVRPDefaults
    neighbours: HeatmapNeighbourDefaults


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


CVRP_READ_PRESET = READPreset(
    pyvrp=PyVRPDefaults(dist_scale=10000, demand_scale=1000),
    neighbours=HeatmapNeighbourDefaults(num_neighbours=50),
)


HFVRP_READ_PRESET = READPreset(
    pyvrp=PyVRPDefaults(dist_scale=10000, demand_scale=100000, cost_scale=1000.0),
    neighbours=HeatmapNeighbourDefaults(num_neighbours=50),
)
