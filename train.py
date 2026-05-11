"""Training / evaluation entrypoint aligned with the slim CVRP/HFVRP mainline.

This version keeps only the CVRP/HFVRP Stage-A training paths and aligns
boolean defaults and decoder arguments with the simplified READ decoder.
"""

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# from diffusion.pl_cvrp_model import CVRPNodeAssignModel as MainCVRPNodeAssignModel
# from diffusion.pl_hfvrp_model import HFVRPNodeAssignModel as MainHFVRPNodeAssignModel

torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# runtime helpers
# -----------------------------------------------------------------------------

def configure_wandb_mode(args):
    if getattr(args, "offline", False):
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ.pop("WANDB_MODE", None)


def configure_wandb_display():
    os.environ.setdefault("WANDB_CONSOLE", "wrap")
    os.environ.setdefault("WANDB_QUIET", "false")
    os.environ.setdefault("WANDB_SILENT", "false")


def build_wandb_settings():
    return wandb.Settings(
        console="wrap",
        quiet=False,
        silent=False,
        show_info=True,
        show_warnings=True,
        show_errors=True,
        max_end_of_run_history_metrics=50,
        max_end_of_run_summary_metrics=50,
    )


def configure_determinism(args):
    seed = int(getattr(args, "seed", 12345))
    pl.seed_everything(seed, workers=True)

    if getattr(args, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _make_ddp_consistent_run_id(explicit_id=None):
    if explicit_id:
        return explicit_id

    env_id = os.environ.get("WANDB_RUN_ID")
    if env_id:
        return env_id

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        obj_list = [None]
        if torch.distributed.get_rank() == 0:
            obj_list[0] = wandb.util.generate_id()
        torch.distributed.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    return wandb.util.generate_id()


# -----------------------------------------------------------------------------
# arguments
# -----------------------------------------------------------------------------

def arg_parser():
    parser = ArgumentParser(description="Train / evaluate slim diffusion models for CVRP and HFVRP.")

    common = parser.add_argument_group("common")
    common.add_argument("--device", default="cuda")
    common.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["cvrp", "cvrp_node", "cvrp_node_assign", "hfvrp", "hfvrp_node", "hfvrp_node_assign"],
    )
    common.add_argument("--storage_path", type=str, required=True)

    data = parser.add_argument_group("data")
    data.add_argument("--training_split", type=str, default=None)
    data.add_argument("--validation_split", type=str, default=None)
    data.add_argument("--test_split", type=str, default=None)
    data.add_argument("--train_split", type=str, default=None, help="VRP training split alias used by model files.")
    data.add_argument("--validation_examples", type=int, default=64)
    data.add_argument("--test_examples", type=int, default=1280)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--batch_size", type=int, default=64)
    optim.add_argument("--num_epochs", type=int, default=50)
    optim.add_argument("--learning_rate", type=float, default=1e-4)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--lr_scheduler", type=str, default="constant")
    optim.add_argument("--num_workers", type=int, default=0)
    optim.add_argument("--fp16", action="store_true")
    optim.add_argument("--use_activation_checkpoint", action="store_true")
    optim.add_argument("--check_val_every_n_epoch", type=int, default=5)

    diffusion = parser.add_argument_group("diffusion")
    diffusion.add_argument("--diffusion_schedule", type=str, default="linear")
    diffusion.add_argument("--diffusion_steps", type=int, default=1000)
    diffusion.add_argument("--inference_diffusion_steps", type=int, default=1000, help="Consistency sampling steps: 1 = single-step generation, >1 = multi-step sampling.")
    diffusion.add_argument("--inference_schedule", type=str, default="cosine")
    diffusion.add_argument("--inference_trick", type=str, default="ddim")
    diffusion.add_argument("--sequential_sampling", type=int, default=1)
    diffusion.add_argument("--parallel_sampling", type=int, default=1)
    diffusion.add_argument("--guided", action="store_true", help="Enable guided x_t sampling during Stage-A inference.")

    model = parser.add_argument_group("shared_model")
    model.add_argument("--n_layers", type=int, default=12)
    model.add_argument("--hidden_dim", type=int, default=192)
    model.add_argument("--sparse_factor", type=int, default=-1)
    model.add_argument("--aggregation", type=str, default="sum")
    model.add_argument("--two_opt_iterations", type=int, default=0)
    model.add_argument("--save_numpy_heatmap", action="store_true")

    run = parser.add_argument_group("run_control")
    run.add_argument("--project_name", type=str, default="vrp_diffusion")
    run.add_argument("--wandb_entity", type=str, default=None)
    run.add_argument("--wandb_logger_name", type=str, default=None)
    run.add_argument("--resume_id", type=str, default=None)
    run.add_argument("--ckpt_path", type=str, default=None)
    run.add_argument("--resume_weight_only", action="store_true")
    run.add_argument("--do_train", action="store_true")
    run.add_argument("--do_test", action="store_true")
    run.add_argument("--ckpt_monitor", type=str, default=None)
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--disable_wandb", action="store_true")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--use_ema", type=int, default=1, choices=[0, 1])
    run.add_argument("--ema_decay", type=float, default=0.999)

    vrp = parser.add_argument_group("vrp_stagea")
    vrp.add_argument("--num_vehicles", type=int, default=0)
    vrp.add_argument("--K_max", type=int, default=None)
    vrp.add_argument("--k_max", type=int, default=None)
    vrp.add_argument("--gnn_layers", type=int, default=4)
    vrp.add_argument("--time_dim", type=int, default=128)
    vrp.add_argument("--node_in_dim", type=int, default=10)
    vrp.add_argument("--veh_in_dim", type=int, default=7)
    vrp.add_argument("--edge_in_dim", type=int, default=8)
    vrp.add_argument("--graph_in_dim", type=int, default=6)
    vrp.add_argument("--dropout", type=float, default=0.0)
    vrp.add_argument("--demand_col", type=int, default=2)
    vrp.add_argument("--n2n_knn_k", type=int, default=8)
    vrp.add_argument("--biattn_heads", type=int, default=4)
    vrp.add_argument("--biattn_dropout", type=float, default=0.0)
    vrp.add_argument("--biattn_head_dim", type=int, default=None)
    vrp.add_argument("--dyn_refresh_every", type=int, default=2)
    vrp.add_argument("--intra_every", type=int, default=2)

    losses = parser.add_argument_group("vrp_losses")

    # Pairwise-partition diffusion objective:
    #   pairwise loss is the main partition supervision;
    #   row CE is only a weak slot anchor;
    #   consistency KL is optional.
    losses.add_argument("--lam_pair", type=float, default=1.0)
    losses.add_argument("--lam_row", type=float, default=0.1)
    losses.add_argument("--lam_cons", type=float, default=0.0)

    losses.add_argument("--pair_pos_samples", type=int, default=128)
    losses.add_argument("--pair_neg_samples", type=int, default=128)
    losses.add_argument("--pair_eps", type=float, default=1e-6)

    evalg = parser.add_argument_group("vrp_eval")
    evalg.add_argument("--eval_cost_every", type=int, default=10)
    evalg.add_argument("--eval_cost_batches", type=int, default=1)
    evalg.add_argument("--eval_fix_init", action="store_true")
    evalg.add_argument("--eval_deterministic", action="store_true")
    evalg.add_argument("--eval_seed", type=int, default=12345)
    evalg.add_argument("--refine_threads", type=int, default=8)

    read = parser.add_argument_group("vrp_read_decoder")
    read.add_argument(
        "--read_pyvrp_budget_ms",
        type=float,
        default=1000.0,
        help=(
            "PyVRP refinement time budget per instance in milliseconds. "
            "Set to 0 to disable PyVRP refinement."
        ),
    )

    hf = parser.add_argument_group("hf_decoder")
    hf.add_argument("--hf_rigid_ce", action="store_true")
    hf.add_argument("--hf_sanity_eval", action="store_true")
    hf.add_argument("--hf_log_cost_gap", dest="hf_log_cost_gap", action="store_true")
    hf.add_argument("--no_hf_log_cost_gap", dest="hf_log_cost_gap", action="store_false")
    hf.add_argument(
        "--hf_slot_order",
        type=str,
        default="attribute",
        choices=["attribute", "raw"],
    )
    hf.set_defaults(hf_log_cost_gap=True)
    guide = parser.add_argument_group("vrp_guidance")

    # CVRP guidance is now sim-only.
    # Capacity/compactness proxy guidance has been removed together with
    # the corresponding training losses.
    guide.add_argument("--c_sim", type=float, default=1.0)
    guide.add_argument("--guide_scale", type=float, default=1.0)
    guide.add_argument("--guided_deterministic", action="store_true")
    guide.add_argument("--cm_guide_skip_first", action="store_true")
    guide.add_argument("--no_cm_guide_skip_first", dest="cm_guide_skip_first", action="store_false")
    guide.add_argument("--cm_guide_schedule", type=str, default="late", choices=["constant", "linear", "late"])
    guide.add_argument("--cm_debug_guidance", action="store_true")

    vrp_model = parser.add_argument_group("vrp_model_toggles")
    vrp_model.add_argument("--use_n2n", action="store_true")
    vrp_model.add_argument("--no_use_n2n", dest="use_n2n", action="store_false")
    parser.add_argument("--n2n_mode", type=str, default="gated", choices=["gated", "attn"])
    parser.add_argument("--n2n_attn_heads", type=int, default=4)
    parser.add_argument("--n2n_attn_dropout", type=float, default=0.05)
    parser.add_argument("--n2n_attn_ffn_mult", type=int, default=2)
    vrp_model.add_argument("--use_global", action="store_true")
    vrp_model.add_argument("--no_use_global", dest="use_global", action="store_false")
    vrp_model.add_argument("--use_adaln", action="store_true")
    vrp_model.add_argument("--no_use_adaln", dest="use_adaln", action="store_false")
    parser.add_argument("--use_v2v", action="store_true")
    parser.add_argument("--v2v_every", type=int, default=2)
    parser.add_argument("--v2v_heads", type=int, default=4)
    parser.add_argument("--v2v_dropout", type=float, default=0.05)
    parser.add_argument("--v2v_ffn_mult", type=int, default=2)
    parser.set_defaults(use_n2n=True, use_v2v=False, use_global=True, use_adaln=False, use_tier_tokens=True)

    parser.add_argument("--hf_lam_type", type=float, default=1.0)
    parser.add_argument("--hf_lam_pair", type=float, default=1.0)
    parser.add_argument("--hf_lam_row", type=float, default=0.10)
    parser.add_argument("--hf_lam_cons", type=float, default=0.0)
    parser.add_argument("--hf_pair_pos_samples", type=int, default=128)
    parser.add_argument("--hf_pair_neg_samples", type=int, default=128)

    vrp_train = parser.add_argument_group("vrp_training_tricks")
    vrp_train.add_argument("--xt_jitter", type=float, default=0.0)
    vrp_train.add_argument("--xt_jitter_warmup", type=int, default=0)
    vrp_train.add_argument("--use_hungarian_ce", action="store_true", default=False)
    vrp_train.add_argument("--veh_shuffle_prob", type=float, default=0.0)
    vrp_train.add_argument("--veh_id_drop_prob", type=float, default=0.0)

    # Optional auxiliary active-K predictor. Kept for checkpoint compatibility.
    parser.add_argument("--use_k_predictor", action="store_true")
    parser.add_argument("--no_use_k_predictor", dest="use_k_predictor", action="store_false")
    parser.set_defaults(use_k_predictor=False)


    consistency = parser.add_argument_group("consistency")
    consistency.add_argument("--consistency", action="store_true")
    consistency.add_argument("--boundary_func", default="truncate")
    consistency.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fast-T2T style time-pair factor: t2 = floor(alpha * t).",
    )


    consistency.add_argument("--hf_curr_align_weight", type=float, default=0.5)
    consistency.add_argument("--hf_curr_type_weight", type=float, default=0.0)
    consistency.add_argument("--hf_curr_raw_start", type=float, default=0.20)
    consistency.add_argument("--hf_curr_raw_end", type=float, default=1.00)
    consistency.add_argument("--hf_curr_raw_ramp_frac", type=float, default=0.50)
    consistency.add_argument("--hf_align_every", type=int, default=4, help="Compute aligned Hungarian loss every k training steps to reduce hot-path overhead.")

    consistency.add_argument("--hf_disable_noise_curriculum", action="store_true")
    consistency.add_argument("--hf_noise_stage1_end", type=float, default=0.20)
    consistency.add_argument("--hf_noise_stage2_end", type=float, default=0.50)
    consistency.add_argument("--hf_noise_stage1_tmax_frac", type=float, default=0.30)
    consistency.add_argument("--hf_noise_stage2_tmax_frac", type=float, default=0.60)


    # Backward-compatible aliases for the CVRP pairwise objective.
    consistency.add_argument("--cm_lam_pair", type=float, default=None)
    consistency.add_argument("--cm_lam_row", type=float, default=None)
    consistency.add_argument("--cm_lam_cons", type=float, default=None)
    consistency.add_argument("--cm_pair_pos_samples", type=int, default=None)
    consistency.add_argument("--cm_pair_neg_samples", type=int, default=None)

    parser.add_argument("--cm_lam_row_consistency", type=float, default=0.0)
    parser.add_argument("--hf_cm_row_ct_mode", type=str, default="sym_kl",
                        choices=["sym_kl", "js", "mse"])
    parser.add_argument("--hf_cm_row_ct_tau", type=float, default=1.0)

    parser.set_defaults(cm_guide_skip_first=True, hf_cm_same_type_hungarian=True)
    parser.add_argument("--hf_expensive_val", action="store_true")
    parser.add_argument('--dataset_knn_k', type=int, default=None)
    parser.add_argument(
        "--assignment_backbone",
        type=str,
        default="hf_full",
        choices=["hf_full", "hf_lite", "hf_lite_edgeupd"],
    )

    timing = parser.add_argument_group("timing")
    timing.add_argument("--report_time", action="store_true")
    timing.add_argument("--report_time_split", type=str, default="test", choices=["test", "val", "all"])
    timing.add_argument("--report_time_only_cost", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()

    if args.train_split is None and args.training_split is not None:
        args.train_split = args.training_split

    return args


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_rank0 = rank == 0

    configure_wandb_mode(args)
    configure_wandb_display()
    configure_determinism(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        _ = torch.empty(1, device="cuda")
        assert torch.cuda.current_device() == local_rank

    if torch.cuda.is_available():
        print("RANK", rank, "LOCAL_RANK", local_rank, "current_device", torch.cuda.current_device())
    else:
        print("RANK", rank, "LOCAL_RANK", local_rank, "current_device", "cpu")
    print(args)

    if args.task in ["cvrp", "cvrp_node", "cvrp_node_assign"]:
        from diffusion.pl_cvrp_model import CVRPNodeAssignModel as model_class

        if args.ckpt_monitor is not None:
            monitor = args.ckpt_monitor
            saving_mode = "max" if "acc" in monitor else "min"
        else:
            monitor = "val/acc"
            saving_mode = "max"

    elif args.task in ["hfvrp", "hfvrp_node", "hfvrp_node_assign"]:
        from diffusion.pl_hfvrp_model import HFVRPNodeAssignModel as model_class

        if args.ckpt_monitor is not None:
            monitor = args.ckpt_monitor
            saving_mode = "max" if "acc" in monitor else "min"
        elif getattr(args, "hf_sanity_eval", False):
            monitor = "val/acc"
            saving_mode = "max"
        else:
            monitor = "val/cost_refined"
            saving_mode = "min"

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    model = model_class(param_args=args)

    log_root = os.path.join(args.storage_path, "models")
    os.makedirs(log_root, exist_ok=True)

    wandb_logger = None
    run_name = args.wandb_logger_name or f"{args.task}"
    if is_rank0 and not getattr(args, "disable_wandb", False):
        run_id = _make_ddp_consistent_run_id(args.resume_id)
        wandb_logger = WandbLogger(
            name=run_name,
            project=args.project_name,
            entity=args.wandb_entity,
            save_dir=log_root,
            id=run_id,
            resume="allow",
            settings=build_wandb_settings(),
        )
        try:
            exp = wandb_logger.experiment
            exp.define_metric("trainer/global_step")
            exp.define_metric("*", step_metric="trainer/global_step")
            exp.config.update(vars(args), allow_val_change=True)
        except Exception:
            pass

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(log_root, run_name, "checkpoints"),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    is_torchrun = ("TORCHELASTIC_RUN_ID" in os.environ) or (world_size > 1)
    n_gpus = torch.cuda.device_count()
    if is_torchrun:
        devices = max(1, n_gpus)
        num_nodes = max(1, world_size // max(1, devices))
        ddp_find_unused = bool(getattr(args, "consistency", False))
        strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=ddp_find_unused)
    else:
        devices = n_gpus if n_gpus > 0 else 1
        num_nodes = 1
        strategy = "ddp" if devices > 1 else "auto"

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=16 if args.fp16 else 32,
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=20), lr_callback, checkpoint_callback],
        max_epochs=args.num_epochs,
        max_steps=-1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        detect_anomaly=False,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=5,
        enable_checkpointing=True,
        inference_mode=False,
        check_val_every_n_epoch=int(args.check_val_every_n_epoch),
    )

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only and ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            drop_prefixes = []

            if not bool(getattr(args, "use_k_predictor", False)):
                drop_prefixes.append("k_predictor.")

            drop_keys = [
                k for k in list(state_dict.keys())
                if any(k.startswith(p) for p in drop_prefixes)
            ]

            for k in drop_keys:
                state_dict.pop(k)

            print(f"[ckpt] weight-only finetune from: {ckpt_path}")
            print(f"[ckpt] dropped {len(drop_keys)} keys: {drop_prefixes}")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            print(f"[ckpt] missing keys: {len(missing)}")
            print(f"[ckpt] unexpected keys: {len(unexpected)}")

            trainer.fit(model, ckpt_path=None)

        else:
            # 只有同构继续训练才走这个，比如 CVRP100 -> CVRP100 恢复训练。
            # CVRP100 -> CVRP200 不要走这里。
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(model, ckpt_path=None)

    elif args.do_test:
        if ckpt_path is None:
            raise ValueError("--ckpt_path must be provided when --do_test is set.")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        if not bool(getattr(args, "use_k_predictor", False)):
            drop_keys = [
                k for k in list(state_dict.keys())
                if k.startswith("k_predictor.")
            ]
            for k in drop_keys:
                state_dict.pop(k)

            print(f"[ckpt] dropped {len(drop_keys)} k_predictor keys")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[ckpt] missing keys: {len(missing)}")
        print(f"[ckpt] unexpected keys: {len(unexpected)}")

        trainer.test(model, ckpt_path=None)
    if trainer.logger is not None:
        trainer.logger.finalize("success")
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main(arg_parser())
