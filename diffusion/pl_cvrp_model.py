"""Unified CVRP Stage-A model on top of the current bipartite heterogeneous backbone.

Key choices:
- keep the current hetero GNN / unified bipartite mainline,
- treat CVRP as a degenerate HF special case,
- use row-softmax assignment semantics throughout eval / decode,
"""

import argparse
import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.co_datasets.cvrp_dataset import CVRPNPZVehNodeDataset
from diffusion.co_datasets.memmap_dataset import CVRPMemmapVehNodeDataset
from diffusion.consistency.cvrp import CVRPConsistency
from .pl_meta_model import VRPAssignMetaModel
from .utils.cvrp_decoder import READDecodeCfg, decode_read_batch_struct
from torch.utils.data import Subset


class ActiveKPredictor(nn.Module):
    """
    Instance-level active-slot predictor.

    It predicts K_ref_used as an auxiliary task. During training it does not
    modify graph.K_used, active_edge, or row diffusion. This keeps the assignment
    denoiser stable while learning a future inference-time active-K estimator.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, max_k: int = 64):
        super().__init__()
        self.max_k = int(max_k)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.max_k + 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat.float())
        # Class 0 is invalid because route count starts from 1.
        logits[:, 0] = -1e9
        return logits


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
                keep_raw=bool(getattr(self.args, "save_numpy_heatmap", False)),
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
        self.use_k_predictor = bool(getattr(self.args, "use_k_predictor", False))
        self.use_k_pred_decode = bool(getattr(self.args, "use_k_pred_decode", False))

        self.use_k_predictor = bool(getattr(self.args, "use_k_predictor", False))

        if self.use_k_predictor:
            self.k_pred_max = int(getattr(self.args, "num_vehicles", 0) or 0)
            self.k_predictor = torch.nn.Sequential(
                torch.nn.Linear(12, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.k_pred_max + 1),
            )
        else:
            self.k_pred_max = None
            self.k_predictor = None

        k_pred_max = (
                getattr(self.args, "K_max", None)
                or getattr(self.args, "k_max", None)
                or getattr(self.args, "num_vehicles", None)
                or 64
        )
        self.k_pred_max = int(k_pred_max)

        self.k_pred_feat_dim = 12
        self.k_pred_loss_weight = float(getattr(self.args, "k_pred_loss_weight", 0.05))
        self.k_pred_decode_slack = int(getattr(self.args, "k_pred_decode_slack", 1))

        self.k_predictor = ActiveKPredictor(
            in_dim=self.k_pred_feat_dim,
            hidden_dim=int(getattr(self.args, "k_pred_hidden_dim", 128)),
            max_k=self.k_pred_max,
        )
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

    def on_test_start(self):
        super_call = getattr(super(), "on_test_start", None)
        if callable(super_call):
            super_call()
        self._test_instance_rows = []
        self._test_seen_count = 0

    def on_test_end(self):
        super_call = getattr(super(), "on_test_end", None)
        if callable(super_call):
            super_call()

        rows = getattr(self, "_test_instance_rows", None)
        if not rows:
            return

        run_name = getattr(self.args, "wandb_logger_name", None) or "cvrp_test"
        outdir = os.path.join(self.args.storage_path, "models", run_name, "instance_logs")
        os.makedirs(outdir, exist_ok=True)

        rank = int(getattr(self, "global_rank", 0))
        csv_path = os.path.join(outdir, f"test_instance_costs_rank{rank}.csv")
        json_path = os.path.join(outdir, f"test_instance_costs_rank{rank}.json")

        fieldnames = [
            "rank",
            "sample_index_local",
            "batch_idx",
            "batch_pos",
            "chosen_rep",
            "gt_cost",
            "stagea_cost",
            "refined_cost",
            "gap_stagea_pct",
            "gap_refined_pct",
            "abs_gap_refined_pct",
            "feasible",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"[test export] saved {len(rows)} rows to {csv_path}")



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

    def q_sample_edge(self, x0_edge: torch.Tensor, t_edge: torch.Tensor, active_edge: torch.Tensor = None):
        return self.consistency_tools.q_sample_edge(x0_edge=x0_edge, t_edge=t_edge, diffusion=self.diffusion, active_edge=active_edge)

    def q_sample_row(self, y0_row: torch.Tensor, t_node: torch.Tensor, k_node: torch.Tensor):
        return self.consistency_tools.q_sample_row(
            y0_row=y0_row,
            t_node=t_node,
            k_node=k_node,
            diffusion=self.diffusion,
        )

    def _get_physical_k(self, graph, B: int, device):
        """Physical candidate slots per graph, normally K_max or veh_batch count."""
        veh_batch = graph.veh_batch.long().to(device)
        veh_cnt = torch.bincount(veh_batch, minlength=max(1, int(B))).long()

        if hasattr(graph, "K_max") and graph.K_max is not None:
            kmax = graph.K_max.view(-1).long().to(device)
            if kmax.numel() == 1 and int(B) > 1:
                kmax = kmax.repeat(int(B))
            kmax = kmax[:int(B)].clamp_min(1)
            return torch.minimum(kmax, veh_cnt.clamp_min(1))

        return veh_cnt.clamp_min(1)

    def _get_graph_k_used(self, graph, B: int, device):
        """
        Strict Kmax mode.

        For now, ignore graph.K_used and always return the physical candidate slot
        count, normally graph.K_max. K_ref_used remains available only for metrics
        or the auxiliary K predictor target.
        """
        return self._get_physical_k(graph, B, device)

    def _capacity_lower_bound_k(self, graph, B: int, device):
        """ceil(total demand / capacity), used as a safety lower bound."""
        node_batch = graph.node_batch.long().to(device)
        demand = graph.demand_linehaul.float().to(device).view(-1)

        total_dem = torch.zeros((B,), device=device, dtype=torch.float32)
        total_dem.index_add_(0, node_batch, demand)

        cap = graph.capacity.float().to(device).view(-1)
        if cap.numel() == 1 and int(B) > 1:
            cap = cap.repeat(int(B))
        cap = cap[:int(B)].clamp_min(1e-6)

        return torch.ceil(total_dem / cap).long().clamp_min(1)

    def _build_k_pred_features(self, graph):
        """
        Static graph-level features for active-K prediction.

        This intentionally does not use GNN hidden states, so K loss does not
        interfere with the denoising GNN representation.
        """
        device = graph.node_features.device
        node_batch = graph.node_batch.long().to(device)
        B = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1

        demand = graph.demand_linehaul.float().to(device).view(-1)
        cap = graph.capacity.float().to(device).view(-1)
        if cap.numel() == 1 and B > 1:
            cap = cap.repeat(B)
        cap = cap[:B].clamp_min(1e-6)

        node_cnt = torch.bincount(node_batch, minlength=B).float().to(device).clamp_min(1.0)

        total_dem = torch.zeros((B,), device=device, dtype=torch.float32)
        total_dem.index_add_(0, node_batch, demand)

        mean_dem = total_dem / node_cnt

        second = torch.zeros((B,), device=device, dtype=torch.float32)
        second.index_add_(0, node_batch, demand * demand)
        var_dem = (second / node_cnt - mean_dem * mean_dem).clamp_min(0.0)
        std_dem = torch.sqrt(var_dem + 1e-12)

        max_dem = torch.zeros((B,), device=device, dtype=torch.float32)
        try:
            max_dem.scatter_reduce_(0, node_batch, demand, reduce="amax", include_self=True)
        except Exception:
            for g in range(B):
                m = node_batch == g
                if bool(m.any()):
                    max_dem[g] = demand[m].max()

        k_phys = self._get_physical_k(graph, B, device).float()
        k_used = self._get_graph_k_used(graph, B, device).float()
        cap_lb = torch.ceil(total_dem / cap).float().clamp_min(1.0)

        # Rel-coordinate statistics from node_features[:, :2].
        xy = graph.node_features[:, :2].float().to(device)
        r = torch.sqrt((xy * xy).sum(dim=-1) + 1e-12)

        mean_r = torch.zeros((B,), device=device)
        mean_r.index_add_(0, node_batch, r)
        mean_r = mean_r / node_cnt

        second_r = torch.zeros((B,), device=device)
        second_r.index_add_(0, node_batch, r * r)
        std_r = torch.sqrt((second_r / node_cnt - mean_r * mean_r).clamp_min(0.0) + 1e-12)

        feat = torch.stack(
            [
                total_dem / cap,
                cap_lb / k_phys.clamp_min(1.0),
                node_cnt / 100.0,
                mean_dem / cap,
                std_dem / cap,
                max_dem / cap,
                k_phys / node_cnt,
                k_used / node_cnt,
                mean_r,
                std_r,
                total_dem / node_cnt.clamp_min(1.0),
                cap_lb / node_cnt.clamp_min(1.0),
            ],
            dim=-1,
        )

        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_k_pred_loss(self, graph):
        """
        Auxiliary K prediction loss.

        Target is K_ref_used. The loss updates only k_predictor because the features
        are static tensors from the batch, not GNN hidden states.
        """
        device = graph.node_features.device

        if (not self.use_k_predictor) or (not hasattr(graph, "K_ref_used")) or graph.K_ref_used is None:
            z = graph.node_features.sum() * 0.0
            return z, {}

        feat = self._build_k_pred_features(graph)
        logits = self.k_predictor(feat)

        target = graph.K_ref_used.view(-1).long().to(device)
        target = target.clamp(1, self.k_pred_max)

        loss = F.cross_entropy(logits, target)

        with torch.no_grad():
            pred = logits.argmax(dim=-1).long()
            acc = (pred == target).float().mean()
            mae = (pred.float() - target.float()).abs().mean()
            over = (pred.float() - target.float()).mean()

        logs = {
            "train/k_loss": loss.detach(),
            "train/k_acc": acc.detach(),
            "train/k_mae": mae.detach(),
            "train/k_bias": over.detach(),
        }
        return loss, logs

    @torch.no_grad()
    def _predict_active_k(self, graph, B: int, device, slack: int = 1):
        """
        Predict inference-time active K.

        Safety:
          K_decode >= ceil(total_demand / capacity)
          K_decode <= physical Kmax
        """
        k_phys = self._get_physical_k(graph, B, device)
        cap_lb = self._capacity_lower_bound_k(graph, B, device)

        feat = self._build_k_pred_features(graph)
        logits = self.k_predictor(feat)
        k_hat = logits.argmax(dim=-1).long().to(device)
        k_hat = k_hat[:B].clamp_min(1)

        k_decode = torch.maximum(k_hat, cap_lb)
        k_decode = torch.minimum(k_decode + int(slack), k_phys)
        return k_decode.clamp_min(1)

    def _active_edge_from_k(self, common: dict, K_active: torch.Tensor):
        """
        Build active_edge from an explicit K_active vector.

        This is needed for fair inference when K_active is predicted rather than
        read from graph.K_used.
        """
        edge_graph = common["edge_graph"].long()
        veh_local = common["veh_local"].long()
        return veh_local < K_active[edge_graph]

    def _row_labels_to_edge_state(self, y_row_flat: torch.Tensor, active_edge: torch.Tensor, common: dict):
        dst = common["dst"]
        veh_local = common["veh_local"]
        xt_edge = (veh_local == y_row_flat[dst]).float()
        return xt_edge * active_edge.float()

    def _dense_logits_from_edges(self, logits: torch.Tensor, active_edge: torch.Tensor, common: dict, K_per_graph: torch.Tensor):
        device = logits.device
        B = int(common["B"])
        Nmax = int(common["node_cnt"].max().item()) if B > 0 else 0
        Kmax = int(K_per_graph.max().item()) if B > 0 else 0

        S = torch.full((B, Nmax, Kmax), -1e9, device=device, dtype=torch.float32)
        if Nmax <= 0 or Kmax <= 0 or not bool(active_edge.any()):
            return S

        emask = active_edge
        g_e = common["edge_graph"][emask].long()
        nl = common["node_local"][emask].long().clamp_min(0)
        vl = common["veh_local"][emask].long().clamp_min(0)
        lg = logits[emask].float()

        S[g_e, nl, vl] = lg

        col_idx = torch.arange(Kmax, device=device).view(1, 1, Kmax)
        col_valid = col_idx < K_per_graph.view(-1, 1, 1)
        S = S.masked_fill(~col_valid, -1e9)
        return S

    def _sample_rows_from_prob(self, prob_bnK: torch.Tensor, common: dict, K_per_graph: torch.Tensor, deterministic: bool = False):
        device = prob_bnK.device
        total_nodes = int(common["node_batch"].numel())
        y_row = torch.zeros((total_nodes,), device=device, dtype=torch.long)

        B = int(common["B"])
        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(K_per_graph[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            start = int(common["node_start"][g].item())
            p = prob_bnK[g, :Ng, :Kg].float()
            p = torch.nan_to_num(p, nan=1.0 / float(Kg), posinf=1.0 / float(Kg), neginf=1.0 / float(Kg))
            p = p.clamp_min(0.0)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if deterministic:
                y_row[start:start + Ng] = torch.argmax(p, dim=-1)
            else:
                y_row[start:start + Ng] = torch.multinomial(p, 1).squeeze(-1)

        return y_row

    def _run_assignment_diffusion_row(self, graph, batch_idx: int, split: str, y_init_row: torch.Tensor, active_edge: torch.Tensor, common: dict):
        from .utils.diffusion_schedulers import InferenceSchedule

        device = self.device
        B = int(common["B"])
        veh_cnt = common.get("K_active", common["veh_cnt"])
        node_batch = common["node_batch"]
        node_start = common["node_start"]

        infer_T = int(getattr(self.args, "inference_diffusion_steps", 50))
        schedule = InferenceSchedule(
            inference_schedule=self.args.inference_schedule,
            T=self.diffusion.T,
            inference_T=infer_T,
        )

        deterministic = bool(getattr(self.args, "eval_deterministic", False))
        y_t = y_init_row.long().clone()
        last_prob_bnK = None

        for i in range(infer_T):
            t1, t2 = schedule(i)
            t1 = int(t1)
            t2 = int(t2)

            t_graph = torch.full((B,), t1, device=device, dtype=torch.long)

            xt_edge = self._row_labels_to_edge_state(y_t, active_edge, common)
            xt_in = torch.nan_to_num(
                xt_edge.float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            logits = self.forward(graph, xt_in, t_graph)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

            S = self._dense_logits_from_edges(logits, active_edge, common, veh_cnt)
            last_prob_bnK = F.softmax(S, dim=2)

            if t2 <= 0:
                break

            t_node = t_graph[node_batch]
            target_t_node = torch.full_like(t_node, t2)
            k_node = veh_cnt[node_batch]

            local_idx = torch.arange(node_batch.numel(), device=device) - node_start[node_batch]
            p0_row = last_prob_bnK[node_batch, local_idx.long(), :]

            y_t = self.consistency_tools.posterior_sample_row(
                y_t_row=y_t,
                t_node=t_node,
                target_t_node=target_t_node,
                p0_row=p0_row,
                k_node=k_node,
                diffusion=self.diffusion,
                deterministic=deterministic,
            )
        return last_prob_bnK

    def _edge_prob_from_logits(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, common: dict):
        device = logits.device
        B = int(common['B'])
        node_cnt = common['node_cnt']
        veh_cnt = common['veh_cnt']
        edge_graph = common['edge_graph']
        node_local = common['node_local']
        veh_local = common['veh_local']

        Nmax = int(node_cnt.max().item()) if B > 0 else 0
        Kmax = int(veh_cnt.max().item()) if B > 0 else 0
        p1 = torch.zeros_like(logits, dtype=torch.float32, device=device)
        if Nmax <= 0 or Kmax <= 0 or not bool(active_edge.any()):
            return p1

        S = torch.full((B, Nmax, Kmax), -10000.0, device=device, dtype=torch.float32)
        emask = active_edge
        g_e = edge_graph[emask].long()
        nl = node_local[emask].long().clamp_min(0)
        vl = veh_local[emask].long().clamp_min(0)
        lg = logits[emask].float()
        S[g_e, nl, vl] = lg

        col_idx = torch.arange(Kmax, device=device).view(1, 1, Kmax)
        col_valid = col_idx < veh_cnt.view(-1, 1, 1)
        S = S.masked_fill(~col_valid, -1e9)
        P = F.softmax(S, dim=2)
        p1[emask] = P[g_e, nl, vl]
        return p1 * active_edge.float()

    def _perm_invariant_acc(self, pred_y: torch.Tensor, gt_y: torch.Tensor, Ku_g: int) -> float:
        if Ku_g <= 1:
            return float((pred_y == gt_y).float().mean().item())
        conf = torch.zeros((Ku_g, Ku_g), device=pred_y.device, dtype=torch.float32)
        for t in range(Ku_g):
            m = gt_y == t
            if bool(m.any()):
                cnt = torch.bincount(pred_y[m].clamp(0, Ku_g - 1), minlength=Ku_g).float()
                conf[t] = cnt
        try:
            import scipy.optimize as opt
            row_ind, col_ind = opt.linear_sum_assignment((-conf).detach().cpu().numpy())
            matched = conf[row_ind, col_ind].sum().item()
        except Exception:
            matched = conf.max(dim=1).values.sum().item()
        return float(matched / max(1, pred_y.numel()))

    def categorical_training_step(self, batch, batch_idx):
        raise RuntimeError('CVRP categorical_training_step() is intentionally disabled. Set --consistency and train through CVRPConsistency.consistency_losses().')

    def _eval_bipartite_stage_a(self, batch, batch_idx: int, split: str):
        import time
        import numpy as np
        import torch
        from torch_geometric.data import Batch as PyGBatch

        device = self.device
        S_req = max(1, int(getattr(self.args, "parallel_sampling", 1)))
        eval_cost_every = int(getattr(self.args, "eval_cost_every", 10))
        eval_cost_batches = int(getattr(self.args, "eval_cost_batches", 1))
        log_cost_gap = bool(getattr(self.args, "hf_log_cost_gap", False))

        do_cost = True if split == "val" else (
                batch_idx % max(1, eval_cost_every) == 0 and batch_idx < max(0, eval_cost_batches)
        )
        S = S_req if do_cost else 1

        if bool(getattr(self.args, "eval_deterministic", False)):
            base_seed = int(getattr(self.args, "eval_seed", 12345))
            seed = base_seed + int(batch_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        graph0 = batch.to(device)
        data_list0 = graph0.to_data_list()
        B0 = len(data_list0)

        # fixed GT cost from the original batch, independent of replica selection
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

        # Physical graph slots are still in common["veh_cnt"].
        # Active slots are either graph.K_used or predicted K_hat.
        # Strict Kmax evaluation:
        # use the full physical candidate slot space. Do not crop by graph.K_used,
        # K_ref_used, or predicted K_hat.
        K_active = self._get_physical_k(graph, B=B, device=device)

        common["K_active"] = K_active

        veh_cnt = K_active
        edge_index = common["edge_index"]
        src, dst = edge_index[0], edge_index[1]
        active_edge = self._active_edge_from_k(common, K_active).to(device).bool()

        if self.use_k_predictor and hasattr(graph, "K_ref_used") and graph.K_ref_used is not None:
            with torch.no_grad():
                feat_k = self._build_k_pred_features(graph)
                logits_k = self.k_predictor(feat_k)
                tgt_k = graph.K_ref_used.view(-1).long().to(device).clamp(1, self.k_pred_max)
                pred_k = logits_k.argmax(dim=-1).long()

                k_acc = (pred_k == tgt_k).float().mean()
                k_mae = (pred_k.float() - tgt_k.float()).abs().mean()
                k_used_mean = K_active.float().mean()
                k_ref_mean = tgt_k.float().mean()
                k_pred_mean = pred_k.float().mean()

            self.log(f"{split}/k_acc", k_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=int(B))
            self.log(f"{split}/k_mae", k_mae, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=int(B))
            self.log(f"{split}/k_used_mean", k_used_mean, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=int(B))
            self.log(f"{split}/k_ref_mean", k_ref_mean, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=int(B))
            self.log(f"{split}/k_pred_mean", k_pred_mean, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=int(B))

        total_nodes = int(common["node_batch"].numel())
        y_init_row = torch.zeros((total_nodes,), device=device, dtype=torch.long)

        g_cpu = None
        if bool(getattr(self.args, "eval_fix_init", False)):
            base_seed = int(getattr(self.args, "eval_seed", 12345))
            init_seed = base_seed + int(batch_idx)
            g_cpu = torch.Generator(device="cpu")
            g_cpu.manual_seed(init_seed)

        for g in range(B):
            Ng = int(common["node_cnt"][g].item())
            Kg = int(veh_cnt[g].item())
            if Ng <= 0 or Kg <= 0:
                continue

            start = int(common["node_start"][g].item())
            if g_cpu is not None:
                samp = torch.randint(Kg, (Ng,), generator=g_cpu, dtype=torch.long)
                y_init_row[start:start + Ng] = samp.to(device)
            else:
                y_init_row[start:start + Ng] = torch.randint(Kg, (Ng,), device=device)

        _do_timing = self._timing_enabled(split, do_cost)
        _t0 = None
        if _do_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _t0 = time.perf_counter()

        prob_bnK = self._run_assignment_diffusion_row(
            graph=graph,
            batch_idx=batch_idx,
            split=split,
            y_init_row=y_init_row,
            active_edge=active_edge,
            common=common,
        )
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
                decode_out = decode_read_batch_struct(
                    [j for _, j in valid_jobs],
                    max_workers=int(getattr(self.args, "refine_threads", 1)),
                    return_profile=False,  # 这里必须改成 True，否则 out.profile 永远是 None
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

        if _do_timing and _t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            self._solve_time_ms_sum += float(elapsed_ms)
            self._solve_time_inst += int(B0)
