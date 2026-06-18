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
from torch_geometric.data import Batch as PyGBatch
import torch.nn.functional as F
from diffusion.co_datasets.cvrp_dataset import CVRPNPZVehNodeDataset
from diffusion.co_datasets.memmap_dataset import CVRPMemmapVehNodeDataset
from diffusion.consistency.cvrp import CVRPConsistency
from .pl_meta_model import VRPAssignMetaModel
from .utils.cvrp_decoder import READDecodeCfg, construct_seed_batch, refine_seed_batch
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

    def _perm_invariant_acc(
            self,
            pred_y: torch.Tensor,
            gt_y: torch.Tensor,
            Ku_g: int,
    ) -> float:
        if pred_y.numel() == 0:
            return 0.0

        Ku_g = int(Ku_g)
        if Ku_g <= 1:
            return float((pred_y == gt_y).float().mean().item())

        pred_y = pred_y.long().clamp(0, Ku_g - 1)
        gt_y = gt_y.long().clamp(0, Ku_g - 1)

        conf = torch.zeros((Ku_g, Ku_g), device=pred_y.device, dtype=torch.float32)

        for t in range(Ku_g):
            mask = gt_y == t
            if bool(mask.any()):
                cnt = torch.bincount(pred_y[mask], minlength=Ku_g).float()
                conf[t] = cnt

        try:
            import scipy.optimize as opt

            row_ind, col_ind = opt.linear_sum_assignment(
                (-conf).detach().cpu().numpy()
            )
            row_ind = torch.as_tensor(row_ind, device=conf.device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=conf.device, dtype=torch.long)
            matched = conf[row_ind, col_ind].sum().item()
        except Exception:
            matched = conf.max(dim=1).values.sum().item()

        return float(matched / max(1, int(pred_y.numel())))

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


        S = S_req if do_cost else 1

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
        veh_cnt, active_edge = self._slot_counts_and_active_edges(graph, common)

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

        logits_lab_all = torch.argmax(prob_bnK, dim=2)

        rep_seed_lab = [None] * B
        rep_final_lab = [None] * B
        rep_seed_routes = [None] * B
        rep_cost_stagea = torch.full((B,), float("inf"), device=device)
        rep_cost_refined = torch.full((B,), float("inf"), device=device)
        rep_nll_seed = torch.full((B,), float("inf"), device=device)
        rep_nll_final = torch.full((B,), float("inf"), device=device)
        rep_feas = torch.zeros((B,), device=device, dtype=torch.bool)
        rep_acc_logits_slot = torch.zeros((B,), device=device)
        meta = [None] * B
        jobs = [None] * B

        cfg = self.decode_cfg
        decode_seed_base = int(getattr(self.args, "eval_seed", 12345)) + int(batch_idx) * 10007

        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(veh_cnt[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            data_g = graph_list[g]
            gold_g = data_g.y.long().to(device).clamp(0, Kg - 1)

            lab_logits = logits_lab_all[g, :Ng].long()
            rep_acc_logits_slot[g] = (lab_logits == gold_g).float().mean()
            meta[g] = (gold_g, data_g, lab_logits)

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

        if do_cost:
            valid_jobs = [(i, j) for i, j in enumerate(jobs) if j is not None]
            if valid_jobs:
                seed_out = construct_seed_batch(
                    [j for _, j in valid_jobs],
                    max_workers=1,
                )

                for (g, _), out in zip(valid_jobs, seed_out):
                    rep_seed_lab[g] = out.seed_lab_t.to(device).long()
                    rep_final_lab[g] = out.seed_lab_t.to(device).long()
                    rep_seed_routes[g] = out.seed_routes
                    rep_cost_stagea[g] = float(out.stagea_cost)
                    rep_cost_refined[g] = float(out.stagea_cost)
                    rep_nll_seed[g] = float(out.nll_seed)
                    rep_nll_final[g] = float(out.nll_seed)
                    rep_feas[g] = bool(torch.isfinite(rep_cost_stagea[g]).item())

                selected_reps = []
                BIG_SELECT = 1e9
                for g0 in range(B0):
                    if S <= 1:
                        reps = torch.tensor([g0], device=device)
                    else:
                        reps = torch.arange(g0 * S, (g0 + 1) * S, device=device)

                    score = rep_cost_stagea[reps] + (~rep_feas[reps]).float() * BIG_SELECT
                    rep = int(reps[torch.argmin(score)].item())
                    if (
                        jobs[rep] is not None
                        and rep_seed_lab[rep] is not None
                        and rep_seed_routes[rep] is not None
                        and bool(rep_feas[rep].item())
                    ):
                        selected_reps.append(rep)

                refine_jobs = []
                for rep in selected_reps:
                    job = dict(jobs[rep])
                    job["seed_lab"] = rep_seed_lab[rep].detach().cpu().numpy()
                    job["seed_routes"] = rep_seed_routes[rep]
                    refine_jobs.append((rep, job))

                if refine_jobs:
                    refined_out = refine_seed_batch(
                        [j for _, j in refine_jobs],
                        max_workers=1,
                    )

                    for (rep, _), out in zip(refine_jobs, refined_out):
                        rep_final_lab[rep] = out.final_lab_t.to(device).long()
                        rep_cost_stagea[rep] = float(out.stagea_cost)
                        rep_cost_refined[rep] = float(out.refined_cost)
                        rep_nll_seed[rep] = float(out.nll_seed)
                        rep_nll_final[rep] = float(out.nll_final)
                        rep_feas[rep] = bool(
                            torch.isfinite(rep_cost_stagea[rep]).item()
                            and torch.isfinite(rep_cost_refined[rep]).item()
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

        instance_rows = []
        base_local_index = int(getattr(self, "_test_seen_count", 0))

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

            gold_g, data_g, lab_logits = meta[rep]
            lab_seed = rep_seed_lab[rep]
            lab_final = rep_final_lab[rep]

            Kg = (
                int(
                    max(
                        int(gold_g.max().item()) + 1,
                        int(lab_seed.max().item()) + 1,
                        int(lab_final.max().item()) + 1,
                        int(lab_logits.max().item()) + 1,
                    )
                ) if gold_g.numel() > 0 else 1
            )

            feas_list.append(rep_feas[rep].float())
            acc_projected_list.append(torch.tensor(self._perm_invariant_acc(lab_seed, gold_g, Kg), device=device))
            acc_refined_list.append(torch.tensor(self._perm_invariant_acc(lab_final, gold_g, Kg), device=device))
            acc_logits_slot_list.append(rep_acc_logits_slot[rep])
            acc_logits_aligned_list.append(
                torch.tensor(self._perm_invariant_acc(lab_logits, gold_g, Kg), device=device)
            )

            if do_cost and torch.isfinite(rep_cost_stagea[rep]).item() and torch.isfinite(rep_cost_refined[rep]).item():
                cost_stagea_list.append(rep_cost_stagea[rep])
                cost_refined_list.append(rep_cost_refined[rep])

            gt_item = None
            if gt_cost0 is not None and g0 < gt_cost0.numel():
                gt_val = gt_cost0[g0]
                if torch.isfinite(gt_val):
                    gt_cost_list.append(gt_val)
                    gt_item = float(gt_val.detach().cpu().item())

            if split == "test" and do_cost:
                stagea_item = float(rep_cost_stagea[rep].detach().cpu().item()) if torch.isfinite(
                    rep_cost_stagea[rep]).item() else float("inf")
                refined_item = float(rep_cost_refined[rep].detach().cpu().item()) if torch.isfinite(
                    rep_cost_refined[rep]).item() else float("inf")
                feasible_item = bool(rep_feas[rep].detach().cpu().item())

                if gt_item is not None and np.isfinite(gt_item) and gt_item > 1e-8:
                    gap_stagea_pct = (stagea_item - gt_item) / gt_item * 100.0 if np.isfinite(stagea_item) else float(
                        "inf")
                    gap_refined_pct = (refined_item - gt_item) / gt_item * 100.0 if np.isfinite(
                        refined_item) else float("inf")
                    abs_gap_refined_pct = abs(gap_refined_pct) if np.isfinite(gap_refined_pct) else float("inf")
                else:
                    gap_stagea_pct = None
                    gap_refined_pct = None
                    abs_gap_refined_pct = None

                chosen_rep_local = int(rep - g0 * S) if S > 1 else 0

                instance_rows.append({
                    "rank": int(getattr(self, "global_rank", 0)),
                    "sample_index_local": int(base_local_index + g0),
                    "batch_idx": int(batch_idx),
                    "batch_pos": int(g0),
                    "chosen_rep": int(chosen_rep_local),
                    "gt_cost": gt_item,
                    "stagea_cost": stagea_item,
                    "refined_cost": refined_item,
                    "gap_stagea_pct": gap_stagea_pct,
                    "gap_refined_pct": gap_refined_pct,
                    "abs_gap_refined_pct": abs_gap_refined_pct,
                    "feasible": feasible_item,
                })

        acc_projected = torch.stack(acc_projected_list).mean() if acc_projected_list else torch.tensor(0.0,
                                                                                                       device=device)
        acc_refined = torch.stack(acc_refined_list).mean() if acc_refined_list else torch.tensor(0.0, device=device)
        acc_logits_slot = torch.stack(acc_logits_slot_list).mean() if acc_logits_slot_list else torch.tensor(0.0,
                                                                                                             device=device)
        acc_logits_aligned = torch.stack(acc_logits_aligned_list).mean() if acc_logits_aligned_list else torch.tensor(
            0.0, device=device)
        feas_rate = torch.stack(feas_list).mean() if feas_list else torch.tensor(0.0, device=device)
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
                    gt_cost_mean = gt_cost_tensor.mean()
                    self.log(f"{split}/gt_cost_raw", gt_cost_mean, prog_bar=False, on_step=False, on_epoch=True,
                             sync_dist=True, batch_size=bs_log)

                    if log_cost_gap:
                        denom = gt_cost_mean.clamp_min(1e-8)
                        gap_stagea = (cost_stagea_mean - gt_cost_mean) / denom
                        gap_refined = (cost_refined_mean - gt_cost_mean) / denom
                        self.log(f"{split}/cost_gap_stagea", gap_stagea, prog_bar=False, on_step=False, on_epoch=True,
                                 sync_dist=True, batch_size=bs_log)
                        self.log(f"{split}/cost_gap_refined", gap_refined, prog_bar=False, on_step=False, on_epoch=True,
                                 sync_dist=True, batch_size=bs_log)

        if split == "val":
            print(
                f"[CVRP VAL SUMMARY] "
                f"acc={float(acc_projected.detach().cpu().item()):.5f} "
                f"acc_proj={float(acc_projected.detach().cpu().item()):.5f} "
                f"acc_ref={float(acc_refined.detach().cpu().item()):.5f} "
                f"acc_slot={float(acc_logits_slot.detach().cpu().item()):.5f} "
                f"acc_aligned={float(acc_logits_aligned.detach().cpu().item()):.5f} "
                f"feas={float(feas_rate.detach().cpu().item()):.5f} "
                f"cost_stagea={float(cost_stagea_mean.detach().cpu().item()) if cost_stagea_mean is not None else float('inf'):.5f} "
                f"cost_refined={float(cost_refined_mean.detach().cpu().item()) if cost_refined_mean is not None else float('inf'):.5f}"
            )
