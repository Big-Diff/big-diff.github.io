from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _as_dict(ns_or_dict: Any) -> Dict[str, Any]:
    if isinstance(ns_or_dict, dict):
        return dict(ns_or_dict)
    if isinstance(ns_or_dict, Namespace):
        return vars(ns_or_dict).copy()
    raise TypeError(f"Unsupported config source: {type(ns_or_dict)}")


@dataclass
class HFDataConfig:
    storage_path: str
    train_split: str
    validation_split: str
    test_split: Optional[str] = None

    # Strict HFVRP mainline:
    #   - full N x K_all bipartite graph
    #   - slot count comes from len(vehicle_capacity)
    #   - slot ordering is solution-independent
    sparse_factor: int = -1
    dataset_knn_k: Optional[int] = None
    hf_slot_order: str = "attribute"  # "attribute" or "raw"


@dataclass
class HFModelConfig:
    hidden_dim: int = 192
    gnn_layers: int = 4
    biattn_heads: int = 4
    graph_in_dim: int = 10

    dyn_refresh_every: int = 2
    intra_every: int = 2
    n2n_knn_k: int = 8

    use_n2n: bool = True
    use_global: bool = True
    use_adaln: bool = False


@dataclass
class HFTrainConfig:
    consistency: bool = True
    boundary_func: str = "truncate"
    alpha: float = 0.5
    xt_jitter: float = 0.05


@dataclass
class HFEvalConfig:
    parallel_sampling: int = 1
    eval_cost_every: int = 10
    eval_cost_batches: int = 1
    eval_deterministic: bool = False
    eval_seed: int = 12345
    eval_fix_init: bool = False

    refine_threads: int = 1
    hf_log_cost_gap: bool = True

    report_time: bool = False
    report_time_split: str = "test"
    report_time_only_cost: bool = True


@dataclass
class HFDecodeConfig:
    # All defaults below are kept in lock-step with HFDecodeCfg in
    # diffusion/utils/hfvrp_decoder.py. The decoder dataclass is the
    # authoritative source; this config only mirrors it.
    projector_topk: int = 4
    projector_cum_prob: float = 0.95

    route_two_opt: bool = True
    two_opt_iter: int = 128

    use_pyvrp: bool = True
    pyvrp_budget_ms: float = 1000.0
    pyvrp_dist_scale: int = 10000
    pyvrp_demand_scale: int = 1000

    # PyVRP official NeighbourhoodParams
    pyvrp_num_neighbours: int = 50
    pyvrp_weight_wait_time: float = 0.2
    pyvrp_symmetric_proximity: bool = True

    # PyVRP component runner / custom-neighbourhood mode
    pyvrp_neigh_mode: str = "heat"  # "official" | "heat"
    pyvrp_cost_scale: float = 1000.0
    pyvrp_collect_stats: bool = False
    pyvrp_display: bool = False
    pyvrp_canonicalize: bool = True

    # Heat-neighbourhood: geometry trunk + tier heat + route-cover.
    heat_base_geo_k: int = 28
    heat_geo_core_k: int = 28
    heat_geo_pool_k: int = 50
    heat_geom_shortlist: int = 96
    heat_tier_k: int = 12
    heat_tier_k_high: int = 12
    heat_tier_k_low: int = 16
    heat_route_cover_top_slots: int = 5
    heat_route_cover_top_slots_high: int = 5
    heat_route_cover_top_slots_low: int = 6
    heat_route_cover_per_slot: int = 2
    heat_low_conf_bonus_k: int = 0
    heat_max_neigh: int = 50
    heat_symmetric: bool = True
    heat_slot_sim_weight: float = 0.0
    heat_tier_sim_weight: float = 0.25
    heat_dist_penalty: float = 1.0
    heat_conf_high_top1: float = 0.85
    heat_conf_high_margin: float = 0.20
    heat_conf_low_top1: float = 0.60
    heat_conf_low_margin: float = 0.08
    heat_global_conf_adapt: bool = True
    heat_global_tier_p1_min: float = 0.72
    heat_global_tier_margin_min: float = 0.10
    heat_global_low_frac_max: float = 0.35

    # Stage-A projector cost terms
    hf_lam_prob: float = 1.0
    hf_lam_econ: float = 0.10
    hf_lam_load: float = 0.10
    hf_lam_fixed: float = 0.05

@dataclass
class HFVRPStageAConfig:
    data: HFDataConfig
    model: HFModelConfig
    train: HFTrainConfig
    eval: HFEvalConfig
    decode: HFDecodeConfig

    @classmethod
    def from_namespace(cls, ns_or_dict: Any) -> "HFVRPStageAConfig":
        d = _as_dict(ns_or_dict)

        # Strict internal config. Deprecated dataset flags are not used by this mainline.
        slot_order = str(d.get("hf_slot_order", "attribute")).strip().lower()
        if slot_order in {"solution", "reference", "gt"}:
            raise ValueError(
                "hf_slot_order='solution' has been removed from the strict HFVRP mainline. "
                "Use 'attribute' or 'raw'."
            )

        data = HFDataConfig(
            storage_path=str(d["storage_path"]),
            train_split=str(d["train_split"]),
            validation_split=str(d["validation_split"]),
            test_split=d.get("test_split", None),
            sparse_factor=int(d.get("sparse_factor", -1)),
            dataset_knn_k=None if d.get("dataset_knn_k", None) is None else int(d["dataset_knn_k"]),
            hf_slot_order=slot_order,
        )

        model = HFModelConfig(
            hidden_dim=int(d.get("hidden_dim", 192)),
            gnn_layers=int(d.get("gnn_layers", 4)),
            biattn_heads=int(d.get("biattn_heads", 4)),
            graph_in_dim=int(d.get("graph_in_dim", 10)),
            dyn_refresh_every=int(d.get("dyn_refresh_every", 2)),
            intra_every=int(d.get("intra_every", 2)),
            n2n_knn_k=int(d.get("n2n_knn_k", 8)),
            use_n2n=bool(d.get("use_n2n", True)),
            use_global=bool(d.get("use_global", True)),
            use_adaln=bool(d.get("use_adaln", False)),
        )

        train = HFTrainConfig(
            consistency=bool(d.get("consistency", True)),
            boundary_func=str(d.get("boundary_func", "truncate")),
            alpha=float(d.get("alpha", 0.5)),
            xt_jitter=float(d.get("xt_jitter", 0.05)),
        )

        eval_cfg = HFEvalConfig(
            parallel_sampling=int(d.get("parallel_sampling", 1)),
            eval_cost_every=int(d.get("eval_cost_every", 10)),
            eval_cost_batches=int(d.get("eval_cost_batches", 1)),
            eval_deterministic=bool(d.get("eval_deterministic", False)),
            eval_seed=int(d.get("eval_seed", 12345)),
            eval_fix_init=bool(d.get("eval_fix_init", False)),
            refine_threads=int(d.get("refine_threads", 1)),
            hf_log_cost_gap=bool(d.get("hf_log_cost_gap", True)),
            report_time=bool(d.get("report_time", False)),
            report_time_split=str(d.get("report_time_split", "test")),
            report_time_only_cost=bool(d.get("report_time_only_cost", True)),
        )

        decode = HFDecodeConfig(
            projector_topk=int(d.get("read_projector_topk", d.get("projector_topk", 4))),
            projector_cum_prob=float(d.get("read_projector_cum_prob", d.get("projector_cum_prob", 0.95))),

            route_two_opt=bool(d.get("route_two_opt", True)),
            two_opt_iter=int(d.get("two_opt_iter", 128)),

            use_pyvrp=bool(d.get("read_use_pyvrp", d.get("use_pyvrp", True))),
            pyvrp_budget_ms=float(d.get("read_pyvrp_budget_ms", d.get("pyvrp_budget_ms", 1000.0))),
            pyvrp_dist_scale=int(d.get("read_pyvrp_dist_scale", d.get("pyvrp_dist_scale", 10000))),
            pyvrp_demand_scale=int(d.get("read_pyvrp_demand_scale", d.get("pyvrp_demand_scale", 1000))),

            # PyVRP official NeighbourhoodParams
            pyvrp_num_neighbours=int(
                d.get("read_pyvrp_num_neighbours", d.get("pyvrp_num_neighbours", 50))
            ),
            pyvrp_weight_wait_time=float(
                d.get("read_pyvrp_weight_wait_time", d.get("pyvrp_weight_wait_time", 0.2))
            ),
            pyvrp_symmetric_proximity=bool(
                d.get("read_pyvrp_symmetric_proximity", d.get("pyvrp_symmetric_proximity", True))
            ),
            pyvrp_neigh_mode=str(
                d.get("read_pyvrp_neigh_mode", d.get("pyvrp_neigh_mode", "heat"))
            ),
            pyvrp_cost_scale=float(
                d.get("read_pyvrp_cost_scale", d.get("pyvrp_cost_scale", 1000.0))
            ),
            pyvrp_collect_stats=bool(
                d.get("read_pyvrp_collect_stats", d.get("pyvrp_collect_stats", False))
            ),
            pyvrp_display=bool(
                d.get("read_pyvrp_display", d.get("pyvrp_display", False))
            ),
            pyvrp_canonicalize=bool(
                d.get("read_pyvrp_canonicalize", d.get("pyvrp_canonicalize", True))
            ),

            heat_base_geo_k=int(
                d.get("read_heat_base_geo_k", d.get("heat_base_geo_k", 28))
            ),
            heat_geo_core_k=int(
                d.get("read_heat_geo_core_k", d.get("heat_geo_core_k", 28))
            ),
            heat_geo_pool_k=int(
                d.get("read_heat_geo_pool_k", d.get("heat_geo_pool_k", 50))
            ),
            heat_geom_shortlist=int(
                d.get("read_heat_geom_shortlist", d.get("heat_geom_shortlist", 96))
            ),
            heat_tier_k=int(
                d.get("read_heat_tier_k", d.get("heat_tier_k", 12))
            ),
            heat_tier_k_high=int(
                d.get("read_heat_tier_k_high", d.get("heat_tier_k_high", 12))
            ),
            heat_tier_k_low=int(
                d.get("read_heat_tier_k_low", d.get("heat_tier_k_low", 16))
            ),
            heat_route_cover_top_slots=int(
                d.get("read_heat_route_cover_top_slots", d.get("heat_route_cover_top_slots", 5))
            ),
            heat_route_cover_top_slots_high=int(
                d.get("read_heat_route_cover_top_slots_high", d.get("heat_route_cover_top_slots_high", 5))
            ),
            heat_route_cover_top_slots_low=int(
                d.get("read_heat_route_cover_top_slots_low", d.get("heat_route_cover_top_slots_low", 6))
            ),
            heat_route_cover_per_slot=int(
                d.get("read_heat_route_cover_per_slot", d.get("heat_route_cover_per_slot", 2))
            ),
            heat_low_conf_bonus_k=int(
                d.get("read_heat_low_conf_bonus_k", d.get("heat_low_conf_bonus_k", 0))
            ),
            heat_max_neigh=int(
                d.get("read_heat_max_neigh", d.get("heat_max_neigh", 50))
            ),
            heat_symmetric=bool(
                d.get("read_heat_symmetric", d.get("heat_symmetric", True))
            ),
            heat_slot_sim_weight=float(
                d.get("read_heat_slot_sim_weight", d.get("heat_slot_sim_weight", 0.0))
            ),
            heat_tier_sim_weight=float(
                d.get("read_heat_tier_sim_weight", d.get("heat_tier_sim_weight", 0.25))
            ),
            heat_dist_penalty=float(
                d.get("read_heat_dist_penalty", d.get("heat_dist_penalty", 1.0))
            ),
            heat_conf_high_top1=float(
                d.get("read_heat_conf_high_top1", d.get("heat_conf_high_top1", 0.85))
            ),
            heat_conf_high_margin=float(
                d.get("read_heat_conf_high_margin", d.get("heat_conf_high_margin", 0.20))
            ),
            heat_conf_low_top1=float(
                d.get("read_heat_conf_low_top1", d.get("heat_conf_low_top1", 0.60))
            ),
            heat_conf_low_margin=float(
                d.get("read_heat_conf_low_margin", d.get("heat_conf_low_margin", 0.08))
            ),
            heat_global_conf_adapt=bool(
                d.get("read_heat_global_conf_adapt", d.get("heat_global_conf_adapt", True))
            ),
            heat_global_tier_p1_min=float(
                d.get("read_heat_global_tier_p1_min", d.get("heat_global_tier_p1_min", 0.72))
            ),
            heat_global_tier_margin_min=float(
                d.get("read_heat_global_tier_margin_min", d.get("heat_global_tier_margin_min", 0.10))
            ),
            heat_global_low_frac_max=float(
                d.get("read_heat_global_low_frac_max", d.get("heat_global_low_frac_max", 0.35))
            ),

            hf_lam_prob=float(d.get("hf_lam_prob", 1.0)),
            hf_lam_econ=float(d.get("hf_lam_econ", 0.10)),
            hf_lam_load=float(d.get("hf_lam_load", 0.10)),
            hf_lam_fixed=float(d.get("hf_lam_fixed", 0.05)),
        )
        cfg = cls(data=data, model=model, train=train, eval=eval_cfg, decode=decode)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.data.train_split:
            raise ValueError("train_split must be provided")
        if not self.data.validation_split:
            raise ValueError("validation_split must be provided")
        if self.data.sparse_factor > 0:
            raise ValueError("HFVRP strict mode requires sparse_factor=-1; GT-aware sparse edges are forbidden")
        if self.data.hf_slot_order not in {"attribute", "attr", "attribute_only", "raw", "none", "false", "0"}:
            raise ValueError("hf_slot_order must be 'attribute' or 'raw'")
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.model.gnn_layers <= 0:
            raise ValueError("gnn_layers must be > 0")
        if self.eval.parallel_sampling <= 0:
            raise ValueError("parallel_sampling must be >= 1")
        if self.decode.projector_topk <= 0:
            raise ValueError("read_projector_topk must be >= 1")
        if self.decode.pyvrp_budget_ms <= 0:
            raise ValueError("read_pyvrp_budget_ms must be > 0")
        if self.decode.pyvrp_dist_scale <= 0:
            raise ValueError("read_pyvrp_dist_scale must be > 0")
        if self.decode.pyvrp_demand_scale <= 0:
            raise ValueError("read_pyvrp_demand_scale must be > 0")
        if self.decode.pyvrp_num_neighbours <= 0:
            raise ValueError("read_pyvrp_num_neighbours must be >= 1")
        if self.decode.pyvrp_weight_wait_time < 0:
            raise ValueError("read_pyvrp_weight_wait_time must be >= 0")
        if self.decode.pyvrp_neigh_mode not in {"official", "heat"}:
            raise ValueError("pyvrp_neigh_mode must be 'official' or 'heat'")
        if self.decode.pyvrp_cost_scale <= 0:
            raise ValueError("pyvrp_cost_scale must be > 0")
        if self.decode.heat_base_geo_k <= 0:
            raise ValueError("heat_base_geo_k must be >= 1")
        if self.decode.heat_geo_core_k < 0:
            raise ValueError("heat_geo_core_k must be >= 0")
        if self.decode.heat_geo_pool_k < 0:
            raise ValueError("heat_geo_pool_k must be >= 0")
        if self.decode.heat_geo_pool_k < self.decode.heat_geo_core_k:
            raise ValueError("heat_geo_pool_k must be >= heat_geo_core_k")
        if self.decode.heat_geom_shortlist <= 0:
            raise ValueError("heat_geom_shortlist must be >= 1")
        if self.decode.heat_tier_k < 0:
            raise ValueError("heat_tier_k must be >= 0")
        if self.decode.heat_tier_k_high < 0:
            raise ValueError("heat_tier_k_high must be >= 0")
        if self.decode.heat_tier_k_low < 0:
            raise ValueError("heat_tier_k_low must be >= 0")
        if self.decode.heat_route_cover_top_slots < 0:
            raise ValueError("heat_route_cover_top_slots must be >= 0")
        if self.decode.heat_route_cover_top_slots_high < 0:
            raise ValueError("heat_route_cover_top_slots_high must be >= 0")
        if self.decode.heat_route_cover_top_slots_low < 0:
            raise ValueError("heat_route_cover_top_slots_low must be >= 0")
        if self.decode.heat_route_cover_per_slot < 0:
            raise ValueError("heat_route_cover_per_slot must be >= 0")
        if self.decode.heat_low_conf_bonus_k < 0:
            raise ValueError("heat_low_conf_bonus_k must be >= 0")
        if self.decode.heat_max_neigh <= 0:
            raise ValueError("heat_max_neigh must be >= 1")
        if self.decode.heat_slot_sim_weight != 0.0:
            # Not recommended in the first iteration, but not forbidden.
            pass
        if self.decode.heat_tier_sim_weight < 0:
            raise ValueError("heat_tier_sim_weight must be >= 0")
        if self.decode.heat_dist_penalty < 0:
            raise ValueError("heat_dist_penalty must be >= 0")
        if not (0.0 <= self.decode.heat_conf_high_top1 <= 1.0):
            raise ValueError("heat_conf_high_top1 must be in [0, 1]")
        if not (0.0 <= self.decode.heat_conf_low_top1 <= 1.0):
            raise ValueError("heat_conf_low_top1 must be in [0, 1]")
        if self.decode.heat_conf_high_margin < 0:
            raise ValueError("heat_conf_high_margin must be >= 0")
        if self.decode.heat_conf_low_margin < 0:
            raise ValueError("heat_conf_low_margin must be >= 0")
        if not (0.0 <= self.decode.heat_global_tier_p1_min <= 1.0):
            raise ValueError("heat_global_tier_p1_min must be in [0, 1]")
        if self.decode.heat_global_tier_margin_min < 0:
            raise ValueError("heat_global_tier_margin_min must be >= 0")
        if not (0.0 <= self.decode.heat_global_low_frac_max <= 1.0):
            raise ValueError("heat_global_low_frac_max must be in [0, 1]")