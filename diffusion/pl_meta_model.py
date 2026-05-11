import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

# from diffusion.models.gnn_encoder import GNNEncoder
from diffusion.utils.diffusion_schedulers import CategoricalDiffusion
from torch.utils.data import Subset



class COMetaModel(pl.LightningModule):
    def __init__(self,
                 param_args,
                 node_feature_only=False):
        super(COMetaModel, self).__init__()
        self.args = param_args
        self.diffusion_schedule = self.args.diffusion_schedule
        self.diffusion_steps = self.args.diffusion_steps
        self.sparse = self.args.sparse_factor > 0 or node_feature_only

        # out_channels = 2
        # self.diffusion = CategoricalDiffusion(
        #     T=self.diffusion_steps, schedule=self.diffusion_schedule)
        #
        # self.model = GNNEncoder(
        #     n_layers=self.args.n_layers,
        #     hidden_dim=self.args.hidden_dim,
        #     out_channels=out_channels,
        #     aggregation=self.args.aggregation,
        #     sparse=self.sparse,
        #     use_activation_checkpoint=self.args.use_activation_checkpoint,
        #     node_feature_only=node_feature_only,
        # )
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule
        )

        # CVRP/HFVRP Stage-A 子类会在各自 __init__ 中覆盖 self.model
        self.model = None
        self.num_training_steps_cached = None
        # self.output_dir = os.path.join('output', time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
        # os.makedirs(self.output_dir)

    # def test_epoch_end(self, outputs):
    #     unmerged_metrics = {}
    #     for metrics in outputs:
    #         for k, v in metrics.items():
    #             if k not in unmerged_metrics:
    #                 unmerged_metrics[k] = []
    #             unmerged_metrics[k].append(v)
    #
    #     merged_metrics = {}
    #     for k, v in unmerged_metrics.items():
    #         merged_metrics[k] = float(np.mean(v))
    #     self.logger.log_metrics(merged_metrics, step=self.global_step)

    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        return self.num_training_steps_cached

    def configure_optimizers(self):
        main_params = []
        k_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if name.startswith("k_predictor."):
                k_params.append(p)
            else:
                main_params.append(p)

        rank_zero_info('Main parameters: %d' % sum(p.numel() for p in main_params))
        rank_zero_info('K predictor parameters: %d' % sum(p.numel() for p in k_params))
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

        lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)
        k_lr = float(getattr(self.args, "k_pred_lr", lr))

        param_groups = [
            {
                "params": main_params,
                "lr": lr,
                "weight_decay": wd,
            }
        ]

        if len(k_params) > 0:
            param_groups.append(
                {
                    "params": k_params,
                    "lr": k_lr,
                    "weight_decay": wd,
                }
            )

        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(param_groups)

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epochs,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """
        Args:
            target_t: int / tensor-like
            t:        int / tensor-like
            x0_pred_prob: (..., 2)  (dense: B,N,N,2 ; sparse: 1,BN,K,2 等)
            xt:           (...)     (dense: B,N,N ; sparse: E)

        Returns:
            xt_next (same shape as xt, float/binary)
            prob    (same shape as xt, float in [0,1])
        """
        diffusion = self.diffusion
        device = x0_pred_prob.device
        if xt.device != device:
            xt = xt.to(device)

        def _to_int_scalar(v, name: str):
            if v is None:
                return None
            if isinstance(v, (int, np.integer)):
                return int(v)
            if torch.is_tensor(v):
                vv = v.reshape(-1)
                if vv.numel() == 0:
                    raise ValueError(f"{name} is empty tensor")
                if vv.numel() > 1:
                    # 推理阶段一般应该都是同一个 timestep
                    if not torch.all(vv == vv[0]).item():
                        raise ValueError(
                            f"{name} has multiple different values: "
                            f"{vv[:8].detach().cpu().tolist()} ... (need scalar timestep)"
                        )
                return int(vv[0].item())
            try:
                return int(v)
            except Exception as e:
                raise TypeError(f"Cannot convert {name}={type(v)} to int: {e}")

        t_int = _to_int_scalar(t, "t")
        target_t_int = _to_int_scalar(target_t, "target_t")
        if target_t_int is None:
            target_t_int = t_int - 1

        # Q_t
        if target_t_int > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t_int]) @ diffusion.Q_bar[t_int]
            Q_t = torch.from_numpy(Q_t).float().to(device)
        else:
            Q_t = torch.eye(2, device=device, dtype=torch.float32)

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t_int]).float().to(device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t_int]).float().to(device)

        # xt -> onehot -> reshape到x0_pred_prob形状
        xt_oh = F.one_hot(xt.long(), num_classes=2).float()
        xt_oh = xt_oh.reshape(x0_pred_prob.shape)

        # posterior math
        x_t_target_prob_part_1 = torch.matmul(xt_oh, Q_t.t().contiguous())  # (...,2)

        x_t_target_prob_part_2 = Q_bar_t_target[0]  # (2,)
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt_oh).sum(dim=-1, keepdim=True)  # (...,1)
        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / (x_t_target_prob_part_3 + 1e-12)

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt_oh).sum(dim=-1, keepdim=True)
        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / (
                    x_t_target_prob_part_3_new + 1e-12)

        sum_x_t_target_prob = sum_x_t_target_prob + x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]
        prob = torch.nan_to_num(sum_x_t_target_prob, nan=0.5, posinf=1.0, neginf=0.0)
        prob = prob.clamp(0.0, 1.0)

        # sample
        if target_t_int > 0:
            xt_next = torch.bernoulli(prob)
        else:
            xt_next = prob

        return xt_next, prob

    def guided_categorical_posterior(self, target_t, t, x0_pred_prob, xt, grad=None):
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with torch.no_grad():
            diffusion = self.diffusion
            if target_t is None:
                target_t = t - 1
            else:
                target_t = target_t.view(1)

            if target_t > 0:
                Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
                Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)  # [2, 2], transition matrix
            else:
                Q_t = torch.eye(2).float().to(x0_pred_prob.device)
            Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
            Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

            xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2), \
                torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
            xt_grad_zero[..., 0] = (1 - xt) * grad
            xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
            xt_grad_one[..., 1] = xt * grad
            xt_grad_one[..., 0] = -xt_grad_one[..., 1]
            xt_grad = xt_grad_zero + xt_grad_one

            # xt_grad = (
            #     torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
            # )
            # xt_grad[..., 1] = grad
            # xt_grad[..., 0] = -grad
            #
            # torch.set_printoptions(threshold=np.inf)
            # print(xt_grad_fake - xt_grad)
            # input()

            xt = F.one_hot(xt.long(), num_classes=2).float()
            xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

            # q(xt−1|xt,x0=0)pθ(x0=0|xt)
            x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
            x_t_target_prob_part_2 = Q_bar_t_target[0]
            x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

            x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3  # [b, n, n, 2]

            sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

            # q(xt−1|xt,x0=1)pθ(x0=1|xt)
            x_t_target_prob_part_2_new = Q_bar_t_target[1]
            x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

            x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

            sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

            p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim=-1)
            p_phi = torch.exp(-xt_grad)
            if self.sparse:
                p_phi = p_phi.reshape(p_theta.shape)
            posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim=-1, keepdim=True)
            posterior = torch.nan_to_num(posterior, nan=0.5, posinf=0.5, neginf=0.5)
            posterior = posterior.clamp_min(0.0)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if target_t > 0:
                p1 = torch.nan_to_num(posterior[..., 1], nan=0.5, posinf=0.5, neginf=0.5).clamp(0.0, 1.0)
                xt = torch.bernoulli(p1)
            else:
                xt = posterior[..., 1].clamp(min=0.0)

    def duplicate_edge_index(self, parallel_sampling, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def train_dataloader(self):
        batch_size = self.args.batch_size
        persistent = int(self.args.num_workers) > 0
        train_dataloader = GraphDataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.args.num_workers,
            persistent_workers=persistent,
            pin_memory = torch.cuda.is_available(),
            drop_last=True)

        return train_dataloader

    def test_dataloader(self, batch_size=None):
        batch_size = self.args.batch_size if batch_size is None else batch_size
        n = int(getattr(self.args, "test_examples", 0) or 0)
        if n > 0:
            n = min(n, len(self.test_dataset))
            test_dataset = Subset(self.test_dataset, list(range(n)))
        else:
            test_dataset = self.test_dataset
        print("Test dataset size:", len(test_dataset))
        persistent = int(self.args.num_workers) > 0
        return GraphDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=persistent,
            pin_memory=torch.cuda.is_available(),
        )
    def val_dataloader(self, batch_size=None):
        batch_size = self.args.batch_size if batch_size is None else batch_size

        n = int(getattr(self.args, "validation_examples", 0) or 0)
        if n > 0:
            n = min(n, len(self.validation_dataset))
            val_dataset = Subset(self.validation_dataset, list(range(n)))
        else:
            val_dataset = self.validation_dataset

        print("Validation dataset size:", len(val_dataset))

        persistent = int(self.args.num_workers) > 0
        return GraphDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=persistent,
            pin_memory=torch.cuda.is_available(),
        )

    def ema_update(self, source_model, target_model, ema):
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.copy_(target_param * ema + source_param * (1 - ema))





class VRPAssignMetaModel(COMetaModel):
    """Common Stage-A bipartite assignment base for CVRP/HFVRP.

    This class keeps diffusion/training/eval utilities generic, while each
    problem subclass keeps its own dataset semantics, vehicle symmetry, and
    decode logic.
    """

    def __init__(self, param_args=None):
        if isinstance(param_args, dict):
            import argparse
            param_args = argparse.Namespace(**param_args)
        elif param_args is None:
            import argparse
            param_args = argparse.Namespace()
        super().__init__(param_args, node_feature_only=False)
        self._Q_bar_torch = None
        self._Q_torch = None
        self.consistency_tools = None
        self.decode_cfg = None
        self._reset_solve_time_meter()

    # ---------- generic helpers ----------
    @staticmethod
    def _group_starts(batch: torch.Tensor, B: int) -> torch.Tensor:
        cnt = torch.bincount(batch, minlength=B)
        starts = torch.zeros((B,), device=batch.device, dtype=torch.long)
        if B > 1:
            starts[1:] = torch.cumsum(cnt, dim=0)[:-1]
        return starts

    def _timing_enabled(self, split: str, do_cost: bool) -> bool:
        if not bool(getattr(self.args, 'report_time', False)):
            return False
        mode = str(getattr(self.args, 'report_time_split', 'test')).lower().strip()
        if mode not in {'test', 'val', 'all'}:
            mode = 'test'
        if mode != 'all' and str(split).lower() != mode:
            return False
        only_cost = bool(getattr(self.args, 'report_time_only_cost', True))
        if only_cost and (not bool(do_cost)):
            return False
        return True

    def _reset_solve_time_meter(self):
        self._solve_time_ms_sum = 0.0
        self._solve_time_inst = 0

    def on_test_epoch_start(self):
        self._reset_solve_time_meter()

    def on_validation_epoch_start(self):
        self._reset_solve_time_meter()

    def on_test_epoch_end(self):
        if self._solve_time_inst > 0:
            mean_ms = self._solve_time_ms_sum / float(self._solve_time_inst)
            self.log('test/solve_time_ms', torch.tensor(mean_ms, device=self.device), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            print(f'[time][test] solve_time_ms/inst = {mean_ms:.3f}')

    def on_validation_epoch_end(self):
        if self._solve_time_inst > 0:
            mean_ms = self._solve_time_ms_sum / float(self._solve_time_inst)
            self.log('val/solve_time_ms', torch.tensor(mean_ms, device=self.device), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            print(f'[time][val] solve_time_ms/inst = {mean_ms:.3f}')

    def _ensure_diffusion_mats(self, device):
        # Q_bar: (T+1, 2, 2)
        if self._Q_bar_torch is None or self._Q_bar_torch.device != device:
            self._Q_bar_torch = torch.as_tensor(
                self.diffusion.Q_bar, device=device, dtype=torch.float32
            )

        # Qs: (T, 2, 2) or similar
        if self._Q_torch is None or self._Q_torch.device != device:
            if hasattr(self.diffusion, "Qs"):
                self._Q_torch = torch.as_tensor(
                    self.diffusion.Qs, device=device, dtype=torch.float32
                )
            elif hasattr(self.diffusion, "Q"):
                self._Q_torch = torch.as_tensor(
                    self.diffusion.Q, device=device, dtype=torch.float32
                )
            else:
                raise AttributeError(
                    "CategoricalDiffusion has neither 'Qs' nor 'Q'. "
                    f"Available attrs: {dir(self.diffusion)}"
                )

    def q_sample_edge(self, x0_edge: torch.Tensor, t_edge: torch.Tensor,
                      active_edge: torch.Tensor = None) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x0) using Q_bar (binary edge diffusion).
        """
        device = x0_edge.device
        self._ensure_diffusion_mats(device)

        T = int(self.diffusion.T)
        t_edge = t_edge.long().clamp_(0, T)
        x0 = x0_edge.long().clamp_(0, 1)

        x0_oh = F.one_hot(x0, num_classes=2).float()  # (E, 2)
        Qb = self._Q_bar_torch.index_select(0, t_edge)  # (E, 2, 2)
        p = torch.bmm(x0_oh.unsqueeze(1), Qb).squeeze(1)  # (E, 2)

        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        xt = torch.bernoulli(p[:, 1].clamp(0.0, 1.0)).float()
        if active_edge is not None:
            xt = xt * active_edge.float()
        return xt

    def _posterior_sample_binary(self, xt: torch.Tensor, t: int, target_t: int, p0: torch.Tensor):
        xt_next, prob = self.categorical_posterior(target_t=target_t, t=t, x0_pred_prob=torch.stack([1.0 - p0, p0], dim=-1), xt=xt.long())
        if xt_next.dim() > prob.dim():
            xt_next = xt_next[..., 1]
        return xt_next.float(), prob.float()

    def forward(self, graph, xt_edge: torch.Tensor, t_graph: torch.Tensor):
        return self.model(graph, xt_edge, t_graph)

    def forward_edge(self, graph, xt_edge: torch.Tensor, t_edge: torch.Tensor):
        if t_edge.dim() == 1 and t_edge.numel() == graph.edge_index.size(1):
            dst = graph.edge_index[1].long().to(xt_edge.device)
            if hasattr(graph, 'node_batch') and graph.node_batch is not None:
                edge_graph = graph.node_batch[dst].long().to(xt_edge.device)
                B = int(graph.node_batch.max().item()) + 1 if graph.node_batch.numel() else 1
            else:
                edge_graph = torch.zeros_like(dst)
                B = 1

            t_graph = torch.zeros((B,), device=xt_edge.device, dtype=t_edge.dtype)
            t_graph.scatter_(0, edge_graph, t_edge.long())
            return self.model(graph, xt_edge, t_graph)

        return self.model(graph, xt_edge, t_edge)

    # def _build_assignment_model(self):
    #     backbone_name = str(getattr(self.args, "assignment_backbone", "hf_full")).lower()
    #
    #     if backbone_name == "hf_lite":
    #         from .models.gnn import EdgeBipartiteDenoiserHF_Lite as Backbone
    #     elif backbone_name == "hf_lite_edgeupd":
    #         from .models.gnn import EdgeBipartiteDenoiserHF_Lite_EdgeUpd as Backbone
    #     else:
    #         from .models.gnn import EdgeBipartiteDenoiser as Backbone

    def _build_assignment_model(self):
        import inspect

        task_name = str(getattr(self.args, "task", "")).lower()
        backbone_name = str(getattr(self.args, "assignment_backbone", "hf_full")).lower()

        # -------- choose backbone class --------
        if task_name in ["hfvrp", "hfvrp_node", "hfvrp_node_assign"]:
            if backbone_name == "hf_lite":
                from .models.gnn_HF import EdgeBipartiteDenoiserV4_HF_Lite as Backbone
            elif backbone_name == "hf_lite_edgeupd":
                from .models.gnn_HF import EdgeBipartiteDenoiserV4_HF_Lite_EdgeUpd as Backbone
            else:
                # keep original A-full HF backbone
                from .models.gnn import EdgeBipartiteDenoiser as Backbone
        else:
            # CVRP and other old paths keep using original gnn.py
            from .models.gnn import EdgeBipartiteDenoiser as Backbone

        hidden = int(getattr(self.args, "hidden_dim", 192))
        n_layers = int(getattr(self.args, "gnn_layers", 4))
        time_dim = int(getattr(self.args, "time_dim", 128))
        dropout = float(getattr(self.args, "dropout", 0.0))

        biattn_heads = int(getattr(self.args, "biattn_heads", 4))
        biattn_dropout = float(getattr(self.args, "biattn_dropout", 0.0))
        biattn_head_dim = getattr(self.args, "biattn_head_dim", None)
        if biattn_head_dim is not None:
            biattn_head_dim = int(biattn_head_dim)

        node_in = int(getattr(self.args, "node_in_dim", 10))
        veh_in = int(getattr(self.args, "veh_in_dim", 7))
        edge_in = int(getattr(self.args, "edge_in_dim", 8))

        # CVRP current gnn.py / dataset graph_feat is 6-dim.
        # HF-lite variants may use richer graph features, so keep at least 10 there.
        if task_name in ["hfvrp", "hfvrp_node", "hfvrp_node_assign"] and backbone_name in ["hf_lite",
                                                                                           "hf_lite_edgeupd"]:
            graph_in = int(getattr(self.args, "graph_in_dim", 10))
            graph_in = max(graph_in, 10)
        else:
            graph_in = int(getattr(self.args, "graph_in_dim", 6))

        use_n2n = bool(getattr(self.args, "use_n2n", True))
        use_global = bool(getattr(self.args, "use_global", True))
        use_adaln = bool(getattr(self.args, "use_adaln", False))

        dyn_refresh_every = int(getattr(self.args, "dyn_refresh_every", 2))
        intra_every = int(getattr(self.args, "intra_every", 2))

        kwargs = dict(
            node_in_dim=node_in,
            veh_in_dim=veh_in,
            edge_in_dim=edge_in,
            graph_in_dim=graph_in,
            hidden_dim=hidden,
            n_layers=n_layers,
            time_dim=time_dim,
            dropout=dropout,
            biattn_heads=biattn_heads,
            biattn_dropout=biattn_dropout,
            biattn_head_dim=biattn_head_dim,
            use_n2n=use_n2n,
            use_global=use_global,
            use_adaln=use_adaln,
            n2n_knn_k=int(getattr(self.args, "n2n_knn_k", 8)),
            dyn_refresh_every=dyn_refresh_every,
            intra_every=intra_every,
        )

        # Optional v2v slot-attention. These keys are filtered below, so older
        # HF backbones that do not support them will not crash.
        kwargs.update(
            dict(
                use_v2v=bool(getattr(self.args, "use_v2v", False)),
                v2v_every=int(getattr(self.args, "v2v_every", 2)),
                v2v_heads=int(getattr(self.args, "v2v_heads", 4)),
                v2v_dropout=float(getattr(self.args, "v2v_dropout", 0.05)),
                v2v_ffn_mult=int(getattr(self.args, "v2v_ffn_mult", 2)),
                n2n_mode=str(getattr(self.args, "n2n_mode", "gated")),
                n2n_attn_heads=int(getattr(self.args, "n2n_attn_heads", 4)),
                n2n_attn_dropout=float(getattr(self.args, "n2n_attn_dropout", 0.05)),
                n2n_attn_ffn_mult=int(getattr(self.args, "n2n_attn_ffn_mult", 2)),
            )
        )

        # Keep compatibility with HF/CVRP backbones that do not yet expose the new
        # v2v keyword arguments.
        sig = inspect.signature(Backbone.__init__)
        valid_keys = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return Backbone(**kwargs)


        # hidden = int(getattr(self.args, 'hidden_dim', 192))
        # n_layers = int(getattr(self.args, 'gnn_layers', 4))
        # time_dim = int(getattr(self.args, 'time_dim', 128))
        # dropout = float(getattr(self.args, 'dropout', 0.0))
        #
        # biattn_heads = int(getattr(self.args, 'biattn_heads', 4))
        # biattn_dropout = float(getattr(self.args, 'biattn_dropout', 0.0))
        # biattn_head_dim = getattr(self.args, 'biattn_head_dim', None)
        # if biattn_head_dim is not None:
        #     biattn_head_dim = int(biattn_head_dim)
        #
        # node_in = int(getattr(self.args, 'node_in_dim', 10))
        # veh_in = int(getattr(self.args, 'veh_in_dim', 7))
        # edge_in = int(getattr(self.args, 'edge_in_dim', 8))
        # graph_in = int(getattr(self.args, 'graph_in_dim', 10))
        #
        # use_n2n = bool(getattr(self.args, 'use_n2n', True))
        # use_global = bool(getattr(self.args, 'use_global', True))
        # use_adaln = bool(getattr(self.args, 'use_adaln', False))
        #
        # dyn_refresh_every = int(getattr(self.args, 'dyn_refresh_every', 2))
        # intra_every = int(getattr(self.args, 'intra_every', 2))
        #
        # return Backbone(
        #     node_in_dim=node_in,
        #     veh_in_dim=veh_in,
        #     edge_in_dim=edge_in,
        #     graph_in_dim=graph_in,
        #     hidden_dim=hidden,
        #     n_layers=n_layers,
        #     time_dim=time_dim,
        #     dropout=dropout,
        #     biattn_heads=biattn_heads,
        #     biattn_dropout=biattn_dropout,
        #     biattn_head_dim=biattn_head_dim,
        #     use_n2n=use_n2n,
        #     use_global=use_global,
        #     use_adaln=use_adaln,
        #     n2n_knn_k=int(getattr(self.args, 'n2n_knn_k', 8)),
        #     dyn_refresh_every=dyn_refresh_every,
        #     intra_every=intra_every,
        # )
    def _prepare_graph_common(self, graph):
        device = self.device
        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        B = int(node_batch.max().item()) + 1 if node_batch.numel() else 1
        veh_cnt = torch.bincount(veh_batch, minlength=B)
        node_cnt = torch.bincount(node_batch, minlength=B)
        veh_start = self._group_starts(veh_batch, B)
        node_start = self._group_starts(node_batch, B)
        edge_graph = node_batch[dst]
        veh_local = src - veh_start[veh_batch[src]]
        node_local = dst - node_start[edge_graph]
        demands = getattr(graph, 'demand_linehaul', None)
        if demands is None:
            demand_col = int(getattr(self.args, 'demand_col', 2))
            demands = graph.node_features[:, demand_col]
        demands = demands.to(device).float()
        return {
            'device': device,
            'node_batch': node_batch,
            'veh_batch': veh_batch,
            'edge_index': edge_index,
            'src': src,
            'dst': dst,
            'B': B,
            'veh_cnt': veh_cnt,
            'node_cnt': node_cnt,
            'veh_start': veh_start,
            'node_start': node_start,
            'edge_graph': edge_graph,
            'veh_local': veh_local,
            'node_local': node_local,
            'demands': demands,
        }




    def _edge_prob_from_logits(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, common: dict):
        return torch.sigmoid(logits.float()) * active_edge.float()

    def _run_assignment_diffusion(self, graph, batch_idx: int, split: str, xt: torch.Tensor, active_edge: torch.Tensor, common: dict):
        from .utils.diffusion_schedulers import InferenceSchedule
        device = self.device
        B = common['B']
        infer_T = int(getattr(self.args, 'inference_diffusion_steps', 50))
        schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule, T=self.diffusion.T, inference_T=infer_T)
        last_p1 = None
        cm_debug = bool(getattr(self.args, 'cm_debug_guidance', False))
        guide_total_sum = 0.0
        guide_cap_sum = 0.0
        guide_sim_sum = 0.0
        guide_steps = 0
        for i in range(infer_T):
            t1, t2 = schedule(i)
            t1 = int(t1)
            t2 = int(t2)
            t_graph = torch.full((B,), t1, device=device, dtype=torch.long)
            if self.consistency_tools is not None and hasattr(self.consistency_tools, 'cm_project_resample_step'):
                xt, p1, cm_info = self.consistency_tools.cm_project_resample_step(
                    self, graph, xt, t_graph, t2, active_edge, step_idx=i, total_steps=infer_T
                )
                guide_losses = cm_info.get('guide_losses', None) if isinstance(cm_info, dict) else None
                if cm_debug and guide_losses is not None:
                    guide_total_sum += float(guide_losses['total'].item())
                    guide_cap_sum += float(guide_losses['cap'].item())
                    guide_sim_sum += float(guide_losses['sim'].item())
                    if float(cm_info.get('guide_scale', 0.0)) > 0:
                        guide_steps += 1
                last_p1 = p1
            else:
                xt_in = xt * 2.0 - 1.0
                logits = self.forward(graph, xt_in, t_graph)
                p1 = self._edge_prob_from_logits(graph, logits, active_edge, common)
                last_p1 = p1
                if t2 > 0:
                    xt, _ = self._posterior_sample_binary(xt, t1, t2, p1)
                    xt = xt * active_edge.float()
                else:
                    xt = p1
            if t2 <= 0:
                break
        if last_p1 is None:
            last_p1 = xt.float() * active_edge.float()
            denom = torch.zeros((graph.node_features.size(0),), device=device, dtype=last_p1.dtype)
            denom.index_add_(0, common['dst'], last_p1)
            last_p1 = last_p1 / denom[common['dst']].clamp_min(1e-12)
            last_p1 = torch.nan_to_num(last_p1, nan=0.0, posinf=0.0, neginf=0.0)
        if cm_debug and guide_steps > 0:
            self.log(f'{split}/cm_guide_total', torch.tensor(guide_total_sum / guide_steps, device=device), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'{split}/cm_guide_cap', torch.tensor(guide_cap_sum / guide_steps, device=device), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'{split}/cm_guide_sim', torch.tensor(guide_sim_sum / guide_steps, device=device), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return last_p1

    def _dense_prob_from_edges(self, graph, last_p1: torch.Tensor, active_edge: torch.Tensor, common: dict, K_per_graph: torch.Tensor):
        device = self.device
        B = common['B']
        Nmax = int(common['node_cnt'].max().item()) if B > 0 else 0
        Kmax = int(K_per_graph.max().item()) if B > 0 else 0
        prob_bnK = torch.zeros((B, Nmax, Kmax), device=device, dtype=torch.float32)
        e_idx = active_edge.nonzero(as_tuple=False).view(-1)
        if e_idx.numel() > 0:
            g_e = common['edge_graph'][e_idx].long()
            nl_e = (common['dst'][e_idx] - common['node_start'][g_e]).long().clamp_min(0)
            vl_e = common['veh_local'][e_idx].long().clamp_min(0)
            nl_e = torch.minimum(nl_e, (common['node_cnt'][g_e] - 1).clamp_min(0))
            vl_e = torch.minimum(vl_e, (K_per_graph[g_e] - 1).clamp_min(0))
            prob_bnK.index_put_((g_e, nl_e, vl_e), last_p1[e_idx].float(), accumulate=False)
        row_sum = prob_bnK.sum(dim=2, keepdim=True).clamp_min(1e-12)
        prob_bnK = prob_bnK / row_sum
        return prob_bnK

    # ---------- training/eval wrappers ----------
    def consistency_training_step(self, batch, batch_idx):
        graph = batch.to(self.device)
        if self.consistency_tools is None or not hasattr(self.consistency_tools, 'consistency_losses'):
            return self.categorical_training_step(batch, batch_idx)
        loss = self.consistency_tools.consistency_losses(self, graph)
        if loss is None:
            raise RuntimeError('consistency_training_step got loss=None.')
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        if getattr(self.args, 'consistency', False) and self.consistency_tools is not None:
            return self.consistency_training_step(batch, batch_idx)
        return self.categorical_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._eval_bipartite_stage_a(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx, split='test'):
        return self._eval_bipartite_stage_a(batch, batch_idx, split=split)

    # ---------- subclass API ----------
    def categorical_training_step(self, batch, batch_idx):
        raise NotImplementedError

    def _eval_bipartite_stage_a(self, batch, batch_idx: int, split: str):
        raise NotImplementedError
