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


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(x)


@dataclass
class HFDataConfig:
    storage_path: str
    train_split: str
    validation_split: str
    test_split: Optional[str] = None

    sparse_factor: int = -1
    dataset_knn_k: Optional[int] = None
    hf_slot_order: str = "attribute"


@dataclass
class HFModelConfig:
    hidden_dim: int = 192
    gnn_layers: int = 4
    biattn_heads: int = 4
    graph_in_dim: int = 10

    n2n_knn_k: int = 8
    n2n_attn_heads: int = 4
    n2n_attn_dropout: float = 0.05
    n2n_attn_ffn_mult: int = 2

    v2v_heads: int = 4
    v2v_dropout: float = 0.05
    v2v_ffn_mult: int = 2

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

    hf_log_cost_gap: bool = True


@dataclass
class HFDecodeConfig:
    # Fixed decoder defaults live in diffusion/utils/read_config.py.
    # This config only controls whether Stage-C PyVRP refinement runs.
    use_pyvrp: bool = True
    pyvrp_budget_ms: float = 1000.0


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
            n2n_knn_k=int(d.get("n2n_knn_k", 8)),
            n2n_attn_heads=int(d.get("n2n_attn_heads", 4)),
            n2n_attn_dropout=float(d.get("n2n_attn_dropout", 0.05)),
            n2n_attn_ffn_mult=int(d.get("n2n_attn_ffn_mult", 2)),
            v2v_heads=int(d.get("v2v_heads", 4)),
            v2v_dropout=float(d.get("v2v_dropout", 0.05)),
            v2v_ffn_mult=int(d.get("v2v_ffn_mult", 2)),
        )

        train = HFTrainConfig(
            consistency=_as_bool(d.get("consistency", True)),
            boundary_func=str(d.get("boundary_func", "truncate")),
            alpha=float(d.get("alpha", 0.5)),
            xt_jitter=float(d.get("xt_jitter", 0.05)),
        )

        eval_cfg = HFEvalConfig(
            parallel_sampling=int(d.get("parallel_sampling", 1)),
            eval_cost_every=int(d.get("eval_cost_every", 10)),
            eval_cost_batches=int(d.get("eval_cost_batches", 1)),
            eval_deterministic=_as_bool(d.get("eval_deterministic", False)),
            eval_seed=int(d.get("eval_seed", 12345)),
            eval_fix_init=_as_bool(d.get("eval_fix_init", False)),
            hf_log_cost_gap=_as_bool(d.get("hf_log_cost_gap", True)),
        )

        decode = HFDecodeConfig(
            use_pyvrp=_as_bool(d.get("read_use_pyvrp", d.get("use_pyvrp", True))),
            pyvrp_budget_ms=float(d.get("read_pyvrp_budget_ms", d.get("pyvrp_budget_ms", 1000.0))),
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
        if self.model.biattn_heads <= 0:
            raise ValueError("biattn_heads must be > 0")
        if self.model.graph_in_dim <= 0:
            raise ValueError("graph_in_dim must be > 0")
        if self.model.n2n_knn_k <= 0:
            raise ValueError("n2n_knn_k must be > 0")

        if self.eval.parallel_sampling <= 0:
            raise ValueError("parallel_sampling must be >= 1")
        if self.eval.eval_cost_every <= 0:
            raise ValueError("eval_cost_every must be >= 1")
        if self.eval.eval_cost_batches <= 0:
            raise ValueError("eval_cost_batches must be >= 1")

        if self.decode.pyvrp_budget_ms < 0:
            raise ValueError("read_pyvrp_budget_ms must be >= 0")


__all__ = [
    "HFDataConfig",
    "HFModelConfig",
    "HFTrainConfig",
    "HFEvalConfig",
    "HFDecodeConfig",
    "HFVRPStageAConfig",
]
