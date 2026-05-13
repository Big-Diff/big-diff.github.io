"""HFVRP Stage-A model with a lean READ-style mainline.

This version keeps the HF decoder aligned with the simplified CVRP mainline:
- consistency-first training,
- minimal validation/test metrics,
- type-aware feasible projection,
- NN + 2-opt warm-start,
- light heterogeneous PyVRP refinement.
"""

import argparse
import os
import time

import numpy as np
import torch

from diffusion.co_datasets.hfvrp_dataset import HFVRPNPZVehNodeDataset
from diffusion.consistency.hfvrp import HFVRPConsistency
from .pl_meta_model import VRPAssignMetaModel
from diffusion.utils.hfvrp_decoder import READDecodeCfg, decode_read_batch_struct
from diffusion.configs.hfvrp_stagea_config import HFVRPStageAConfig
from torch.utils.data import Subset
from torch_geometric.data import Batch as PyGBatch


class HFVRPNodeAssignModel(VRPAssignMetaModel):
    def __init__(self, param_args=None):
        print('[LOAD] HFVRPNodeAssignModel(stage-a) from', __file__)
        if isinstance(param_args, dict):
            param_args = argparse.Namespace(**param_args)
        elif param_args is None:
            param_args = argparse.Namespace()

        param_args.use_pt_bipartite = True

        # Strict HFVRP Stage-A config: full fleet slots, row-categorical diffusion.
        self.cfg = HFVRPStageAConfig.from_namespace(param_args)

        def _resolve(p):
            if p is None:
                return None
            p = str(p)
            if p == '':
                return None
            return p if os.path.isabs(p) else os.path.join(self.cfg.data.storage_path, p)

        train_path = _resolve(self.cfg.data.train_split)
        val_path = _resolve(self.cfg.data.validation_split)
        test_path = _resolve(self.cfg.data.test_split)
        if train_path is None or val_path is None:
            raise ValueError('Need train_split and validation_split')
        if test_path is None:
            test_path = val_path

        super().__init__(param_args)

        def _make_dataset(path):
            if os.path.isdir(path):
                raise ValueError("HFVRP expects NPZ datasets, not memmap directories.")
            return HFVRPNPZVehNodeDataset(path)

        self.train_dataset = _make_dataset(train_path)
        self.validation_dataset = _make_dataset(val_path)
        self.test_dataset = _make_dataset(test_path)

        train_examples = getattr(self.args, "training_examples", None)
        if train_examples is not None and int(train_examples) > 0:
            n = min(int(train_examples), len(self.train_dataset))
            self.train_dataset = Subset(self.train_dataset, range(n))

        val_examples = getattr(self.args, "validation_examples", None)
        if val_examples is not None and int(val_examples) > 0:
            n = min(int(val_examples), len(self.validation_dataset))
            self.validation_dataset = Subset(self.validation_dataset, range(n))

        test_examples = getattr(self.args, "test_examples", None)
        if test_examples is not None and int(test_examples) > 0:
            n = min(int(test_examples), len(self.test_dataset))
            self.test_dataset = Subset(self.test_dataset, range(n))

        self.save_hyperparameters(param_args)

        print("train_dataset:", len(self.train_dataset))
        print("validation_dataset:", len(self.validation_dataset))
        print("test_dataset:", len(self.test_dataset))


        self.consistency_tools = HFVRPConsistency(
            self.args,
            sigma_max=self.diffusion.T,
            boundary_func=self.cfg.train.boundary_func,
        )
        if bool(self.cfg.train.consistency):
            self.consistency_trainer = self.consistency_tools

        self.model = self._build_assignment_model()
        self.decode_cfg = self._build_hf_decode_cfg()

        print('[cfg] train:', train_path)
        print('[cfg] val  :', val_path)
        print('[cfg] test :', test_path)


    def forward(self, graph, xt_edge: torch.Tensor, t_graph: torch.Tensor):
        return self.model(graph, xt_edge, t_graph)

    def categorical_training_step(self, batch, batch_idx):
        raise RuntimeError(
            'HFVRP categorical_training_step() is intentionally disabled. '
            'Set --consistency and train through HFVRPConsistency.consistency_losses().'
        )

    def _build_hf_decode_cfg(self):
        budget_ms = float(
            getattr(
                self.args,
                "read_pyvrp_budget_ms",
                getattr(self.args, "pyvrp_budget_ms", 1000.0),
            )
        )
        return READDecodeCfg(pyvrp_budget_ms=budget_ms)

    def _eval_bipartite_stage_a(self, batch, batch_idx: int, split: str):
        device = self.device
        ec = self.cfg.eval

        S_req = max(1, int(ec.parallel_sampling))
        eval_cost_every = int(ec.eval_cost_every)
        eval_cost_batches = int(ec.eval_cost_batches)
        log_cost_gap = bool(ec.hf_log_cost_gap)

        if split == "val":
            do_cost = True
        else:
            do_cost = (
                    batch_idx % max(1, eval_cost_every) == 0
                    and batch_idx < max(0, eval_cost_batches)
            )

        if not do_cost:
            return

        S = S_req

        if bool(ec.eval_deterministic):
            seed = int(ec.eval_seed) + int(batch_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        graph0 = batch.to(device)
        data_list0 = graph0.to_data_list()
        B0 = len(data_list0)

        gt_cost0 = None
        if hasattr(graph0, "gt_cost") and graph0.gt_cost is not None:
            gt_cost0 = graph0.gt_cost.view(-1).to(device).float().abs()

        if S > 1:
            rep = []
            for d in data_list0:
                rep.extend([d] * S)
            graph = PyGBatch.from_data_list(rep).to(device)
            graph_list = rep
            B = B0 * S
        else:
            graph = graph0
            graph_list = data_list0
            B = B0

        common = self._prepare_graph_common(graph)
        veh_cnt = common["veh_cnt"]
        edge_index = common["edge_index"]
        src, dst = edge_index[0], edge_index[1]

        veh_cnt, active_edge = self._slot_counts_and_active_edges(graph, common)

        y_init = self._sample_initial_row_labels(
            graph,
            common,
            veh_cnt,
            device=device,
            batch_idx=batch_idx,
            deterministic=bool(ec.eval_fix_init),
            seed_base=int(ec.eval_seed),
        )
        xt = self._row_labels_to_edge_state(y_init, active_edge, common).clamp(0.0, 1.0)

        _do_timing = self._timing_enabled(split, do_cost)
        _t0 = None
        if _do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _t0 = time.perf_counter()

        last_p1 = self._run_assignment_diffusion(
            graph,
            batch_idx,
            split,
            xt,
            active_edge,
            common,
        )
        prob_bnK = self._dense_prob_from_edges(
            graph,
            last_p1,
            active_edge,
            common,
            veh_cnt,
        )

        rep_cost_refined = torch.full((B,), float("inf"), device=device)
        jobs = [None] * B

        cfg = self.decode_cfg
        decode_seed_base = int(ec.eval_seed) + int(batch_idx) * 10007

        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(veh_cnt[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            data_g = graph_list[g]
            prob_c_t = prob_bnK[g, :Ng, :Kg].detach().cpu()

            depot_xy = data_g.depot_xy[0, 0].detach().cpu().numpy().astype(np.float32)
            xy_rel = data_g.node_features[:, :2].detach().cpu().numpy().astype(np.float32)
            clients_xy = xy_rel + depot_xy[None, :]

            dem_g = data_g.demand_linehaul.detach().cpu().numpy().astype(np.float32)
            cap_g = data_g.vehicle_capacity[:Kg].detach().cpu().numpy().astype(np.float32)
            fixed_g = data_g.vehicle_fixed_cost[:Kg].detach().cpu().numpy().astype(np.float32)
            unit_g = data_g.vehicle_unit_distance_cost[:Kg].detach().cpu().numpy().astype(np.float32)

            jobs[g] = {
                "prob_c": prob_c_t,
                "dem": dem_g,
                "cap_vec": cap_g,
                "fixed_vec": fixed_g,
                "unit_cost_vec": unit_g,
                "depot_xy": depot_xy,
                "clients_xy": clients_xy,
                "cfg": cfg,
                "seed": int(decode_seed_base + g),
                "tier_vec": data_g.vehicle_tier[:Kg].detach().cpu().numpy().astype(np.int64),
            }

        valid_jobs = [(i, j) for i, j in enumerate(jobs) if j is not None]
        if valid_jobs:
            decode_out = decode_read_batch_struct(
                [j for _, j in valid_jobs],
                max_workers=int(ec.refine_threads),
                return_profile=False,
            )

            for (g, _), out in zip(valid_jobs, decode_out):
                if np.isfinite(float(out.refined_cost)):
                    rep_cost_refined[g] = float(out.refined_cost)

        cost_list = []
        gt_cost_list = []

        def _rep_indices_for_sample(g0: int):
            if S <= 1:
                return torch.tensor([g0], device=device)
            return torch.arange(g0 * S, (g0 + 1) * S, device=device)

        for g0 in range(B0):
            reps = _rep_indices_for_sample(g0)
            rep = int(reps[torch.argmin(rep_cost_refined[reps])].item())

            if torch.isfinite(rep_cost_refined[rep]).item():
                cost_list.append(rep_cost_refined[rep])

                if gt_cost0 is not None and g0 < gt_cost0.numel():
                    gt_val = gt_cost0[g0]
                    if torch.isfinite(gt_val):
                        gt_cost_list.append(gt_val)

        bs_log = int(B0)

        if cost_list:
            cost_mean = torch.stack(cost_list).mean()

            self.log(
                f"{split}/cost",
                cost_mean,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=bs_log,
            )

            if log_cost_gap and gt_cost_list:
                gt_cost_tensor = torch.stack(gt_cost_list)
                gt_cost_tensor = gt_cost_tensor[torch.isfinite(gt_cost_tensor)]

                if gt_cost_tensor.numel() > 0:
                    gt_cost_mean = gt_cost_tensor.mean().clamp_min(1e-8)
                    cost_gap = (cost_mean - gt_cost_mean) / gt_cost_mean

                    self.log(
                        f"{split}/cost_gap",
                        cost_gap,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=bs_log,
                    )

            if split == "val":
                print(f"[HF VAL] cost={float(cost_mean.detach().cpu().item()):.5f}")

        else:
            bad = torch.tensor(float("inf"), device=device)
            self.log(
                f"{split}/cost",
                bad,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=bs_log,
            )

        if _do_timing and _t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            self._solve_time_ms_sum += float(elapsed_ms)
            self._solve_time_inst += int(B0)