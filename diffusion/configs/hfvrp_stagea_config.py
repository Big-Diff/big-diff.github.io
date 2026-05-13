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
class HFVRPStageAConfig:
    data: HFDataConfig
    model: HFModelConfig
    train: HFTrainConfig
    eval: HFEvalConfig

    @classmethod
    def from_namespace(cls, ns_or_dict: Any) -> "HFVRPStageAConfig":
        d = _as_dict(ns_or_dict)

        data = HFDataConfig(
            storage_path=str(d["storage_path"]),
            train_split=str(d["train_split"]),
            validation_split=str(d["validation_split"]),
            test_split=d.get("test_split", None),
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
            xt_jitter=float(d.get("xt_jitter", 0.0)),
        )

        eval_cfg = HFEvalConfig(
            parallel_sampling=int(d.get("parallel_sampling", 1)),
            eval_cost_every=int(d.get("eval_cost_every", 10)),
            eval_cost_batches=int(d.get("eval_cost_batches", 1)),
            eval_deterministic=bool(d.get("eval_deterministic", False)),
            eval_seed=int(d.get("eval_seed", 12345)),
            eval_fix_init=bool(d.get("eval_fix_init", False)),
            refine_threads=int(d.get("refine_threads", 8)),
            hf_log_cost_gap=bool(d.get("hf_log_cost_gap", True)),
            report_time=bool(d.get("report_time", False)),
            report_time_split=str(d.get("report_time_split", "test")),
            report_time_only_cost=bool(d.get("report_time_only_cost", True)),
        )


        cfg = cls(
            data=data,
            model=model,
            train=train,
            eval=eval_cfg,
        )
        cfg.validate()
        return cfg
    def validate(self) -> None:
        # Dataset paths.
        if not self.data.storage_path:
            raise ValueError("storage_path must be provided")
        if not self.data.train_split:
            raise ValueError("train_split must be provided")
        if not self.data.validation_split:
            raise ValueError("validation_split must be provided")

        # Stage-A denoiser.
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.model.gnn_layers <= 0:
            raise ValueError("gnn_layers must be > 0")
        if self.model.biattn_heads <= 0:
            raise ValueError("biattn_heads must be > 0")
        if self.model.graph_in_dim <= 0:
            raise ValueError("graph_in_dim must be > 0")
        if self.model.dyn_refresh_every <= 0:
            raise ValueError("dyn_refresh_every must be > 0")
        if self.model.intra_every <= 0:
            raise ValueError("intra_every must be > 0")
        if self.model.n2n_knn_k < 0:
            raise ValueError("n2n_knn_k must be >= 0")

        # Row-categorical consistency training.
        if not (0.0 <= self.train.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if self.train.xt_jitter < 0:
            raise ValueError("xt_jitter must be >= 0")

        # Evaluation.
        if self.eval.parallel_sampling <= 0:
            raise ValueError("parallel_sampling must be >= 1")
        if self.eval.refine_threads <= 0:
            raise ValueError("refine_threads must be >= 1")
        if self.eval.eval_seed < 0:
            raise ValueError("eval_seed must be >= 0")

