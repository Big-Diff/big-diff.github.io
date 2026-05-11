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

from diffusion.co_datasets.hfvrp_dataset  import VRPNPZVehNodeDataset
from diffusion.consistency.hfvrp import HFVRPConsistency
from .pl_meta_model import VRPAssignMetaModel
from diffusion.utils.hfvrp_decoder import READDecodeCfg, decode_read_batch_struct
from diffusion.configs.hfvrp_stagea_config import HFVRPStageAConfig
from torch.utils.data import Subset

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
            sf = int(self.cfg.data.sparse_factor)

            if os.path.isdir(path):
                raise ValueError('HFVRP currently expects NPZ datasets, not memmap directories.')

            if sf > 0:
                raise ValueError(
                    "HFVRP strict mode requires sparse_factor=-1. "
                    "Positive sparse_factor creates GT-aware sparse edges."
                )

            # Do not precompute or store node-kNN graphs in the dataset.
            # The HF GNN builds batched KNN on GPU from --n2n_knn_k, which keeps
            # training/inference semantics aligned and avoids large CPU-side caches.
            return VRPNPZVehNodeDataset(
                path,
                sparse_factor=sf,
                dataset_knn_k=None,
                keep_raw=bool(getattr(self.args, "save_numpy_heatmap", False)),
                hf_slot_order=str(self.cfg.data.hf_slot_order),
            )

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

    def _vehicle_shuffle_groups(self, graph, g: int, total_count_g: int, common: dict):
        vg0 = int(common['veh_start'][g].item())
        tiers = getattr(graph, 'vehicle_tier', None)
        if tiers is None:
            return super()._vehicle_shuffle_groups(graph, g, total_count_g, common)
        tiers_g = tiers[vg0:vg0 + total_count_g].view(-1).to(self.device)
        groups = []
        for t in torch.unique(tiers_g):
            idx = (tiers_g == t).nonzero(as_tuple=False).view(-1)
            if idx.numel() > 0:
                groups.append(idx)
        return groups if groups else [torch.arange(total_count_g, device=self.device, dtype=torch.long)]

    def _perm_invariant_acc(self, pred_y: torch.Tensor, gt_y: torch.Tensor, tier_vec: torch.Tensor) -> float:
        """Permutation-invariant slot accuracy within each used tier."""
        K = int(tier_vec.numel())
        if K <= 1:
            return float((pred_y == gt_y).float().mean().item())

        used_mask = torch.zeros((K,), device=tier_vec.device, dtype=torch.bool)
        used_mask[torch.unique(gt_y)] = True

        total_match = 0.0
        total_n = max(1, pred_y.numel())
        uniq_tiers = torch.unique(tier_vec[used_mask]) if bool(used_mask.any()) else torch.unique(tier_vec)
        for t in uniq_tiers.tolist():
            ids = ((tier_vec == int(t)) & used_mask).nonzero(as_tuple=False).view(-1)
            if ids.numel() == 0:
                continue
            if ids.numel() == 1:
                only_id = int(ids.item())
                total_match += float(((gt_y == only_id) & (pred_y == only_id)).sum().item())
                continue

            ids_list = ids.tolist()
            m = int(ids.numel())
            conf = torch.zeros((m, m), device=pred_y.device, dtype=torch.float32)
            for a, gt_slot in enumerate(ids_list):
                mask = (gt_y == gt_slot)
                if bool(mask.any()):
                    pred_clip = pred_y[mask]
                    for b, pr_slot in enumerate(ids_list):
                        conf[a, b] = float((pred_clip == pr_slot).sum().item())

            try:
                import scipy.optimize as opt
                row_ind, col_ind = opt.linear_sum_assignment((-conf).detach().cpu().numpy())
                total_match += conf[row_ind, col_ind].sum().item()
            except Exception:
                total_match += conf.max(dim=1).values.sum().item()

        return float(total_match / float(total_n))

    def forward(self, graph, xt_edge: torch.Tensor, t_graph: torch.Tensor):
        return self.model(graph, xt_edge, t_graph)

    def forward_edge(self, graph, xt_edge: torch.Tensor, t_edge: torch.Tensor):
        """
        consistency trainer passes edge-level t_edge (E,)
        denoiser expects graph-level t_graph (B,)
        aggregate t_edge -> t_graph by edge_graph
        """
        device = xt_edge.device
        edge_index = graph.edge_index.long().to(device)
        dst = edge_index[1]

        if hasattr(graph, "edge_graph") and graph.edge_graph is not None:
            edge_graph = graph.edge_graph.long().to(device)
        else:
            edge_graph = graph.node_batch.long().to(device)[dst]

        B = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0
        t_graph = torch.zeros((B,), device=device, dtype=torch.long)

        if edge_graph.numel() > 0:
            t_graph.scatter_reduce_(
                0, edge_graph, t_edge.long(), reduce="amax", include_self=True
            )

        return self.forward(graph, xt_edge, t_graph)

    def q_sample_edge(
            self,
            x0_edge: torch.Tensor,
            t_edge: torch.Tensor,
            active_edge: torch.Tensor = None,
    ):
        raise RuntimeError(
            "HFVRP q_sample_edge() is disabled in strict row-categorical mode. "
            "Use HFVRPConsistency.q_sample_row() through consistency_losses()."
        )

    def _row_labels_to_edge_state(self, y_row_flat: torch.Tensor, active_edge: torch.Tensor, common: dict):
        """Convert row labels y_i into edge-form one-hot x_{k,i}."""
        dst = common["edge_index"][1].long()
        veh_local = common["veh_local"].long()
        return (veh_local == y_row_flat[dst]).float() * active_edge.float()

    def _sample_initial_row_labels(
        self,
        graph,
        common: dict,
        K_per_graph: torch.Tensor,
        *,
        device,
        batch_idx: int,
        deterministic: bool,
        seed_base: int,
    ):
        """Sample a legal row-categorical initial state: one slot per customer."""
        node_batch = graph.node_batch.long().to(device)
        total_nodes = int(node_batch.numel())
        y_init = torch.zeros((total_nodes,), device=device, dtype=torch.long)

        g_cpu = None
        if deterministic:
            g_cpu = torch.Generator(device="cpu")
            g_cpu.manual_seed(int(seed_base) + int(batch_idx))

        B = int(common["B"])

        # Fast path used during normal stochastic evaluation: vectorized over nodes.
        if g_cpu is None:
            K_node = K_per_graph.long().to(device)[node_batch].clamp_min(1)
            y_init = torch.floor(torch.rand((total_nodes,), device=device) * K_node.float()).long()
            return torch.minimum(torch.clamp_min(y_init, 0), K_node - 1)

        # Deterministic fixed-init path keeps an explicit CPU generator for stable
        # comparison across runs. This runs only in evaluation, not training.
        for g in range(B):
            mask = node_batch == int(g)
            Ng = int(mask.sum().item())
            Kg = int(K_per_graph[g].item())
            if Ng <= 0 or Kg <= 0:
                continue
            samp = torch.randint(Kg, (Ng,), generator=g_cpu, dtype=torch.long).to(device)
            y_init[mask] = samp
        return y_init

    def categorical_training_step(self, batch, batch_idx):
        raise RuntimeError(
            'HFVRP categorical_training_step() is intentionally disabled. '
            'Set --consistency and train through HFVRPConsistency.consistency_losses().'
        )

    # pl_hfvrp_model.py
    def _build_hf_decode_cfg(self):
        dc = self.cfg.decode
        use_pyvrp = bool(getattr(dc, "use_pyvrp", True))
        budget_ms = float(getattr(dc, "pyvrp_budget_ms", 1000.0))
        return READDecodeCfg(
            pyvrp_budget_ms=budget_ms if use_pyvrp else 0.0,
        )

    def _eval_bipartite_stage_a(self, batch, batch_idx: int, split: str):
        import time
        import numpy as np
        import torch
        from torch_geometric.data import Batch as PyGBatch

        device = self.device
        ec = self.cfg.eval

        S_req = max(1, int(ec.parallel_sampling))
        eval_cost_every = int(ec.eval_cost_every)
        eval_cost_batches = int(ec.eval_cost_batches)
        log_cost_gap = bool(ec.hf_log_cost_gap)

        if split == "val":
            do_cost = True
        else:
            do_cost = (batch_idx % max(1, eval_cost_every) == 0 and batch_idx < max(0, eval_cost_batches))

        S = S_req if do_cost else 1

        if bool(ec.eval_deterministic):
            base_seed = int(ec.eval_seed)
            seed = base_seed + int(batch_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        graph0 = batch.to(device)
        data_list0 = graph0.to_data_list()
        B0 = len(data_list0)

        # fixed GT cost from original batch, independent of replica selection
        gt_cost0 = None
        if hasattr(graph0, "gt_cost") and graph0.gt_cost is not None:
            gt_cost0 = graph0.gt_cost.view(-1).to(device).float().abs()
        elif hasattr(graph0, "gt_cost_linehaul") and graph0.gt_cost_linehaul is not None:
            gt_cost0 = graph0.gt_cost_linehaul.view(-1).to(device).float().abs()

        # replicate by sample:
        # sample0_rep0, sample0_rep1, ..., sample1_rep0, ...
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
        active_edge = self.consistency_tools._get_active_edge(
            graph, common["edge_graph"], src, dst, B, device
        )

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

        last_p1 = self._run_assignment_diffusion(graph, batch_idx, split, xt, active_edge, common)
        prob_bnK = self._dense_prob_from_edges(graph, last_p1, active_edge, common, veh_cnt)
        logits_lab_all = torch.argmax(prob_bnK, dim=2)

        rep_seed_lab = [None] * B
        rep_final_lab = [None] * B
        rep_cost_stagea = torch.full((B,), float("inf"), device=device)
        rep_cost_refined = torch.full((B,), float("inf"), device=device)
        rep_nll_seed = torch.full((B,), float("inf"), device=device)
        rep_nll_final = torch.full((B,), float("inf"), device=device)
        rep_feas = torch.zeros((B,), device=device, dtype=torch.bool)
        rep_acc_logits_slot = torch.zeros((B,), device=device)
        meta = [None] * B
        jobs = [None] * B

        cfg = self.decode_cfg
        decode_seed_base = int(ec.eval_seed) + int(batch_idx) * 10007

        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(veh_cnt[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            data_g = graph_list[g]
            gold_g = data_g.y.long().to(device).clamp(0, Kg - 1)
            tier_g = (
                data_g.vehicle_tier[:Kg].long().to(device)
                if hasattr(data_g, "vehicle_tier") and data_g.vehicle_tier is not None
                else torch.zeros((Kg,), device=device, dtype=torch.long)
            )

            lab_logits = logits_lab_all[g, :Ng].long()
            rep_acc_logits_slot[g] = (lab_logits == gold_g).float().mean()
            meta[g] = (gold_g, tier_g, data_g, lab_logits)

            if not do_cost:
                rep_seed_lab[g] = lab_logits
                rep_final_lab[g] = lab_logits
                chosen_prob = prob_bnK[g, torch.arange(Ng, device=device), lab_logits].clamp_min(1e-12)
                nll = -torch.log(chosen_prob).sum()
                rep_nll_seed[g] = float(nll.item())
                rep_nll_final[g] = float(nll.item())
                rep_feas[g] = True
                continue

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
            cap_g = data_g.vehicle_capacity[:Kg].detach().cpu().numpy().astype(np.float32)
            fixed_g = (
                data_g.vehicle_fixed_cost[:Kg].detach().cpu().numpy().astype(np.float32)
                if hasattr(data_g, "vehicle_fixed_cost") and data_g.vehicle_fixed_cost is not None
                else np.zeros((Kg,), dtype=np.float32)
            )
            unit_g = (
                data_g.vehicle_unit_distance_cost[:Kg].detach().cpu().numpy().astype(np.float32)
                if hasattr(data_g, "vehicle_unit_distance_cost") and data_g.vehicle_unit_distance_cost is not None
                else np.ones((Kg,), dtype=np.float32)
            )

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
                "slot_mask": (
                    data_g.vehicle_slot_mask[:Kg].detach().cpu().numpy().astype(np.bool_)
                    if hasattr(data_g, "vehicle_slot_mask") and data_g.vehicle_slot_mask is not None
                    else None
                ),
                "tier_vec": (
                    data_g.vehicle_tier[:Kg].detach().cpu().numpy().astype(np.int64)
                    if hasattr(data_g, "vehicle_tier") and data_g.vehicle_tier is not None
                    else None
                ),
            }

        if do_cost:
            valid_jobs = [(i, j) for i, j in enumerate(jobs) if j is not None]
            if valid_jobs:
                decode_out = decode_read_batch_struct(
                    [j for _, j in valid_jobs],
                    max_workers=int(ec.refine_threads),
                    return_profile=False,
                )
                for (g, _), out in zip(valid_jobs, decode_out):
                    rep_seed_lab[g] = out.seed_lab_t.to(device).long()
                    rep_final_lab[g] = out.final_lab_t.to(device).long()
                    rep_cost_stagea[g] = float(out.stagea_cost)
                    rep_cost_refined[g] = float(out.refined_cost)
                    rep_nll_seed[g] = float(out.nll_seed)
                    rep_nll_final[g] = float(out.nll_final)
                    rep_feas[g] = bool(
                        torch.isfinite(rep_cost_stagea[g]).item()
                        and torch.isfinite(rep_cost_refined[g]).item()
                    )

        acc_projected_list = []
        acc_refined_list = []
        acc_logits_slot_list = []
        acc_logits_aligned_list = []
        feas_list = []
        cost_stagea_list = []
        cost_refined_list = []
        gt_cost_list = []
        BIG = 1e9

        def _rep_indices_for_sample(g0: int):
            if S <= 1:
                return torch.tensor([g0], device=device)
            return torch.arange(g0 * S, (g0 + 1) * S, device=device)

        for g0 in range(B0):
            reps = _rep_indices_for_sample(g0)
            score = (rep_cost_refined if do_cost else rep_nll_seed)[reps] + (~rep_feas[reps]).float() * BIG
            rep = int(reps[torch.argmin(score)].item())

            if rep_seed_lab[rep] is None or rep_final_lab[rep] is None or meta[rep] is None:
                continue

            gold_g, tier_g, data_g, lab_logits = meta[rep]
            lab_seed = rep_seed_lab[rep]
            lab_final = rep_final_lab[rep]

            feas_list.append(rep_feas[rep].float())
            acc_projected_list.append(
                torch.tensor(self._perm_invariant_acc(lab_seed, gold_g, tier_g), device=device)
            )
            acc_refined_list.append(
                torch.tensor(self._perm_invariant_acc(lab_final, gold_g, tier_g), device=device)
            )
            acc_logits_slot_list.append(rep_acc_logits_slot[rep])
            acc_logits_aligned_list.append(
                torch.tensor(self._perm_invariant_acc(lab_logits, gold_g, tier_g), device=device)
            )

            if do_cost and torch.isfinite(rep_cost_stagea[rep]).item() and torch.isfinite(rep_cost_refined[rep]).item():
                cost_stagea_list.append(rep_cost_stagea[rep])
                cost_refined_list.append(rep_cost_refined[rep])

            if gt_cost0 is not None and g0 < gt_cost0.numel():
                gt_val = gt_cost0[g0]
                if torch.isfinite(gt_val):
                    gt_cost_list.append(gt_val)

        acc_projected = (
            torch.stack(acc_projected_list).mean()
            if acc_projected_list else torch.tensor(0.0, device=device)
        )
        acc_refined = (
            torch.stack(acc_refined_list).mean()
            if acc_refined_list else torch.tensor(0.0, device=device)
        )
        acc_logits_slot = (
            torch.stack(acc_logits_slot_list).mean()
            if acc_logits_slot_list else torch.tensor(0.0, device=device)
        )
        acc_logits_aligned = (
            torch.stack(acc_logits_aligned_list).mean()
            if acc_logits_aligned_list else torch.tensor(0.0, device=device)
        )
        feas_rate = (
            torch.stack(feas_list).mean()
            if feas_list else torch.tensor(0.0, device=device)
        )
        bs_log = int(B0)

        self.log(f"{split}/acc", acc_projected, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs_log)
        self.log(f"{split}/acc_projected", acc_projected, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs_log)
        self.log(f"{split}/acc_refined", acc_refined, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs_log)
        self.log(f"{split}/acc_logits_slot", acc_logits_slot, prog_bar=False, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=bs_log)
        self.log(f"{split}/acc_logits_aligned", acc_logits_aligned, prog_bar=False, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=bs_log)
        self.log(f"{split}/feasible_rate", feas_rate, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs_log)

        cost_stagea_mean = None
        cost_refined_mean = None
        gt_cost_mean = None

        if do_cost and cost_stagea_list:
            cost_stagea_mean = torch.stack(cost_stagea_list).mean()
            cost_refined_mean = torch.stack(cost_refined_list).mean()

            self.log(f"{split}/cost_stagea", cost_stagea_mean, prog_bar=False, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=bs_log)
            self.log(f"{split}/cost_refined", cost_refined_mean, prog_bar=False, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=bs_log)

            if gt_cost_list:
                gt_cost_tensor = torch.stack(gt_cost_list)
                gt_cost_tensor = gt_cost_tensor[torch.isfinite(gt_cost_tensor)]
                if gt_cost_tensor.numel() > 0:
                    gt_cost_mean = gt_cost_tensor.mean().clamp_min(1e-8)
                    self.log(f"{split}/gt_cost_raw", gt_cost_mean, prog_bar=False, on_step=False, on_epoch=True,
                             sync_dist=True, batch_size=bs_log)

                    if log_cost_gap:
                        gap_stagea = (cost_stagea_mean - gt_cost_mean) / gt_cost_mean
                        gap_refined = (cost_refined_mean - gt_cost_mean) / gt_cost_mean
                        self.log(f"{split}/cost_gap_stagea", gap_stagea, prog_bar=False, on_step=False, on_epoch=True,
                                 sync_dist=True, batch_size=bs_log)
                        self.log(f"{split}/cost_gap_refined", gap_refined, prog_bar=False, on_step=False, on_epoch=True,
                                 sync_dist=True, batch_size=bs_log)

        elif do_cost:
            bad = torch.tensor(float("inf"), device=device)
            self.log(f"{split}/cost_stagea", bad, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=bs_log)
            self.log(f"{split}/cost_refined", bad, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=bs_log)

        if split == "val":
            print(
                f"[HF VAL SUMMARY] "
                f"acc={float(acc_projected.detach().cpu().item()):.5f} "
                f"acc_proj={float(acc_projected.detach().cpu().item()):.5f} "
                f"acc_ref={float(acc_refined.detach().cpu().item()):.5f} "
                f"acc_slot={float(acc_logits_slot.detach().cpu().item()):.5f} "
                f"acc_aligned={float(acc_logits_aligned.detach().cpu().item()):.5f} "
                f"feas={float(feas_rate.detach().cpu().item()):.5f} "
                f"cost_stagea={float(cost_stagea_mean.detach().cpu().item()) if cost_stagea_mean is not None else float('inf'):.5f} "
                f"cost_refined={float(cost_refined_mean.detach().cpu().item()) if cost_refined_mean is not None else float('inf'):.5f}"
            )

        if _do_timing and _t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            self._solve_time_ms_sum += float(elapsed_ms)
            self._solve_time_inst += int(B0)