"""Unified CVRP Stage-A model on top of the current bipartite heterogeneous backbone.

Key choices:
- keep the current hetero GNN / unified bipartite mainline,
- treat CVRP as a degenerate HF special case,
- use row-softmax assignment semantics throughout eval / decode,
"""

import argparse
import numpy as np
import torch
import os
import time
from torch_geometric.data import Batch as PyGBatch
import torch.nn.functional as F
from diffusion.co_datasets.cvrp_dataset import CVRPNPZVehNodeDataset
from diffusion.co_datasets.memmap_dataset import CVRPMemmapVehNodeDataset
from diffusion.consistency.cvrp import CVRPConsistency
from .pl_meta_model import VRPAssignMetaModel
from .utils.cvrp_decoder import READDecodeCfg, decode_read_batch_struct
from torch.utils.data import Subset




class CVRPNodeAssignModel(VRPAssignMetaModel):
    def __init__(self, param_args=None):
        print('[LOAD] CVRPNodeAssignModel(unified-stage-a) from', __file__)
        if isinstance(param_args, dict):
            param_args = argparse.Namespace(**param_args)
        elif param_args is None:
            param_args = argparse.Namespace()

        param_args.use_pt_bipartite = True

        def _resolve(p):
            if p is None:
                return None
            p = str(p)
            if p == '':
                return None
            return p if os.path.isabs(p) else os.path.join(getattr(param_args, 'storage_path', ''), p)

        train_path = _resolve(getattr(param_args, 'train_split', None))
        val_path = _resolve(getattr(param_args, 'validation_split', None))
        test_path = _resolve(getattr(param_args, 'test_split', None))
        if train_path is None or val_path is None:
            raise ValueError('Need train_split and validation_split')
        if test_path is None:
            test_path = val_path

        super().__init__(param_args)

        def _make_dataset(path):
            kmax_arg = (
                    getattr(self.args, "K_max", None)
                    or getattr(self.args, "k_max", None)
                    or getattr(self.args, "num_vehicles", None)
            )
            kmax_arg = int(kmax_arg) if kmax_arg is not None and int(kmax_arg) > 0 else None

            if kmax_arg is None:
                raise ValueError("K_max/k_max/num_vehicles must be set for strict Kmax CVRP training/evaluation. Refusing to fall back to K_ref_used from actions/best_tour, because that would leak oracle route-count information into the model slot space.")
            print(f"[cfg] using Kmax graph with K_max={kmax_arg}")

            common_kwargs = dict(
                K_max=kmax_arg,
                sparse_factor=int(getattr(self.args, "sparse_factor", -1)),
                seed=int(getattr(self.args, "seed", 12345)),
                keep_raw=bool(getattr(self.args, "keep_raw_instance", False)),
            )

            if os.path.isdir(path):
                return CVRPMemmapVehNodeDataset(path, **common_kwargs)
            else:
                return CVRPNPZVehNodeDataset(path, **common_kwargs)

        self.train_dataset = _make_dataset(train_path)
        self.validation_dataset = _make_dataset(val_path)
        self.test_dataset = _make_dataset(test_path)

        train_examples = getattr(self.args, 'training_examples', None)
        if train_examples is not None and int(train_examples) > 0:
            self.train_dataset = Subset(self.train_dataset, range(min(int(train_examples), len(self.train_dataset))))
        val_examples = getattr(self.args, 'validation_examples', None)
        if val_examples is not None and int(val_examples) > 0:
            self.validation_dataset = Subset(self.validation_dataset, range(min(int(val_examples), len(self.validation_dataset))))
        test_examples = getattr(self.args, 'test_examples', None)
        if test_examples is not None and int(test_examples) > 0:
            self.test_dataset = Subset(self.test_dataset, range(min(int(test_examples), len(self.test_dataset))))

        self.save_hyperparameters(param_args)
        self.consistency_tools = CVRPConsistency(self.args, sigma_max=self.diffusion.T, boundary_func=getattr(self.args, 'boundary_func', 'truncate'))
        if bool(getattr(self.args, 'consistency', True)):
            self.consistency_trainer = self.consistency_tools

        self.model = self._build_assignment_model()
        # pl_cvrp_model.py
        self.decode_cfg = READDecodeCfg(
            pyvrp_budget_ms=(
                float(getattr(self.args, "read_pyvrp_budget_ms", 1000.0))
                if bool(getattr(self.args, "read_use_pyvrp", True))
                else 0.0
            )
        )

        print('train_dataset:', len(self.train_dataset))
        print('validation_dataset:', len(self.validation_dataset))
        print('test_dataset:', len(self.test_dataset))
        print('[cfg] train:', train_path)
        print('[cfg] val  :', val_path)
        print('[cfg] test :', test_path)
    def forward(self, graph, xt_edge: torch.Tensor, t_graph: torch.Tensor):
        return self.model(graph, xt_edge, t_graph)

    def forward_edge(self, graph, xt_edge: torch.Tensor, t_edge: torch.Tensor):
        device = xt_edge.device
        edge_index = graph.edge_index.long().to(device)
        dst = edge_index[1]
        if hasattr(graph, 'edge_graph') and graph.edge_graph is not None:
            edge_graph = graph.edge_graph.long().to(device)
        else:
            edge_graph = graph.node_batch.long().to(device)[dst]
        B = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0
        t_graph = torch.zeros((B,), device=device, dtype=torch.long)
        if edge_graph.numel() > 0:
            t_graph.scatter_reduce_(0, edge_graph, t_edge.long(), reduce='amax', include_self=True)
        return self.forward(graph, xt_edge, t_graph)

    def categorical_training_step(self, batch, batch_idx):
        raise RuntimeError('CVRP categorical_training_step() is intentionally disabled. Set --consistency and train through CVRPConsistency.consistency_losses().')

    def _eval_bipartite_stage_a(self, batch, batch_idx: int, split: str):
        device = self.device

        S_req = max(1, int(getattr(self.args, "parallel_sampling", 1)))
        eval_cost_every = int(getattr(self.args, "eval_cost_every", 10))
        eval_cost_batches = int(getattr(self.args, "eval_cost_batches", 1))
        log_cost_gap = bool(getattr(self.args, "log_cost_gap", getattr(self.args, "hf_log_cost_gap", False)))

        if split == "val":
            do_cost = True
        else:
            do_cost = (
                    batch_idx % max(1, eval_cost_every) == 0
                    and batch_idx < max(0, eval_cost_batches)
            )

        # Final paper path: skipped cost-eval batches produce no public metric.
        # Do not run diffusion, construct jobs, or select replicas by NLL.
        if not do_cost:
            return

        S = S_req

        if bool(getattr(self.args, "eval_deterministic", False)):
            seed = int(getattr(self.args, "eval_seed", 12345)) + int(batch_idx)
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
        elif hasattr(graph0, "gt_cost_linehaul") and graph0.gt_cost_linehaul is not None:
            gt_cost0 = graph0.gt_cost_linehaul.view(-1).to(device).float().abs()

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

        # Strict Kmax evaluation: use the slot semantics defined by CVRPConsistency.
        veh_cnt, active_edge = self._slot_counts_and_active_edges(graph, common)
        common["K_active"] = veh_cnt

        y_init = self._sample_initial_row_labels(
            graph,
            common,
            veh_cnt,
            device=device,
            batch_idx=batch_idx,
            deterministic=bool(getattr(self.args, "eval_fix_init", False)),
            seed_base=int(getattr(self.args, "eval_seed", 12345)),
        )

        xt = self._row_labels_to_edge_state(
            y_init,
            active_edge,
            common,
        ).clamp(0.0, 1.0)

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

        rep_cost = torch.full((B,), float("inf"), device=device)
        jobs = [None] * B

        cfg = self.decode_cfg
        decode_seed_base = int(getattr(self.args, "eval_seed", 12345)) + int(batch_idx) * 10007

        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(veh_cnt[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            data_g = graph_list[g]
            prob_c_t = prob_bnK[g, :Ng, :Kg].detach().cpu()

            if hasattr(data_g, "points") and data_g.points is not None:
                pts = data_g.points.detach().cpu().numpy().astype(np.float32)
                depot_xy = pts[0]
                clients_xy = pts[1:]
            else:
                depot_xy = data_g.depot_xy[0, 0].detach().cpu().numpy().astype(np.float32)
                xy_rel = data_g.node_features[:, :2].detach().cpu().numpy().astype(np.float32)
                clients_xy = xy_rel + depot_xy[None, :]

            dem_g = data_g.demand_linehaul.detach().cpu().numpy().astype(np.float32)

            if hasattr(data_g, "vehicle_capacity") and data_g.vehicle_capacity is not None:
                cap_g = float(data_g.vehicle_capacity.view(-1)[0].item())
            else:
                cap_g = float(data_g.capacity.view(-1)[0].item())

            jobs[g] = {
                "prob_c": prob_c_t,
                "dem": dem_g,
                "cap": cap_g,
                "depot_xy": depot_xy,
                "clients_xy": clients_xy,
                "cfg": cfg,
                "seed": int(decode_seed_base + g),
            }

        valid_jobs = [(i, j) for i, j in enumerate(jobs) if j is not None]
        if valid_jobs:
            decode_out = decode_read_batch_struct(
                [j for _, j in valid_jobs],
                max_workers=int(getattr(self.args, "refine_threads", 1)),
                return_profile=False,
            )

            for (g, _), out in zip(valid_jobs, decode_out):
                cost = float(out.refined_cost)
                if np.isfinite(cost):
                    rep_cost[g] = cost

        cost_list = []
        gt_cost_list = []

        def _rep_indices_for_sample(g0: int):
            if S <= 1:
                return torch.tensor([g0], device=device)
            return torch.arange(g0 * S, (g0 + 1) * S, device=device)

        for g0 in range(B0):
            reps = _rep_indices_for_sample(g0)
            rep = int(reps[torch.argmin(rep_cost[reps])].item())

            if torch.isfinite(rep_cost[rep]).item():
                cost_list.append(rep_cost[rep])

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
                print(f"[CVRP VAL] cost={float(cost_mean.detach().cpu().item()):.5f}")

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