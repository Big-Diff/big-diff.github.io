import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.loader import DataLoader as GraphDataLoader
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
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule
        )

        # CVRP/HFVRP Stage-A 子类会在各自 __init__ 中覆盖 self.model
        self.model = None
        self.num_training_steps_cached = None


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
        params = [p for p in self.parameters() if p.requires_grad]

        rank_zero_info(
            "Trainable parameters: %d" % sum(p.numel() for p in params)
        )
        rank_zero_info(
            "Training steps: %d" % self.get_total_num_training_steps()
        )

        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.args.learning_rate),
            weight_decay=float(self.args.weight_decay),
        )

        if self.args.lr_scheduler == "constant":
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.args.num_epochs),
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }



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


    def _build_assignment_model(self):
        import inspect

        task_name = str(getattr(self.args, "task", "")).lower()
        backbone_name = str(getattr(self.args, "assignment_backbone", "hf_full")).lower()

        # -------- choose backbone class --------
        if task_name in ["hfvrp", "hfvrp_node", "hfvrp_node_assign"]:
            if backbone_name == "hf_lite":
                from .models.gnn_HF import EdgeBipartiteDenoiser_HF as Backbone
            elif backbone_name == "hf_lite_edgeupd":
                from .models.gnn_HF import EdgeBipartiteDenoiser_HF_EdgeUpd as Backbone
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

    def _slot_counts_and_active_edges(self, graph, common: dict):
        if self.consistency_tools is None:
            raise RuntimeError("consistency_tools is required for row-categorical evaluation.")

        _, _, _, _, active_edge, k_graph, _ = self.consistency_tools._batch_structure(
            graph,
            common["src"],
            common["dst"],
            int(common["B"]),
            self.device,
        )
        return k_graph.long(), active_edge.bool()

    def _row_labels_to_edge_state(
            self,
            y_row_flat: torch.Tensor,
            active_edge: torch.Tensor,
            common: dict,
    ):
        dst = common["dst"].long()
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
        node_batch = graph.node_batch.long().to(device)
        total_nodes = int(node_batch.numel())
        y_init = torch.zeros((total_nodes,), device=device, dtype=torch.long)

        if not deterministic:
            K_node = K_per_graph.long().to(device)[node_batch].clamp_min(1)
            y = torch.floor(torch.rand((total_nodes,), device=device) * K_node.float()).long()
            return torch.minimum(torch.clamp_min(y, 0), K_node - 1)

        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(int(seed_base) + int(batch_idx))

        B = int(common["B"])
        for g in range(B):
            mask = node_batch == int(g)
            Ng = int(mask.sum().item())
            Kg = int(K_per_graph[g].item())
            if Ng <= 0 or Kg <= 0:
                continue
            y_init[mask] = torch.randint(Kg, (Ng,), generator=g_cpu, dtype=torch.long).to(device)

        return y_init

    @staticmethod
    def _rep_indices_for_sample(g0: int, S: int, device):
        if S <= 1:
            return torch.tensor([g0], device=device)
        return torch.arange(g0 * S, (g0 + 1) * S, device=device)




    def _edge_prob_from_logits(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, common: dict):
        return torch.sigmoid(logits.float()) * active_edge.float()

    def _run_assignment_diffusion(
            self,
            graph,
            batch_idx: int,
            split: str,
            xt: torch.Tensor,
            active_edge: torch.Tensor,
            common: dict,
    ):
        from .utils.diffusion_schedulers import InferenceSchedule

        del batch_idx, split

        device = self.device
        B = int(common["B"])

        if self.consistency_tools is None or not hasattr(
                self.consistency_tools,
                "cm_project_resample_step",
        ):
            raise RuntimeError(
                "Row-categorical assignment diffusion requires consistency_tools.cm_project_resample_step()."
            )

        infer_T = int(getattr(self.args, "inference_diffusion_steps", 50))
        schedule = InferenceSchedule(
            inference_schedule=self.args.inference_schedule,
            T=self.diffusion.T,
            inference_T=infer_T,
        )

        last_p1 = None

        for i in range(infer_T):
            t1, t2 = schedule(i)
            t_graph = torch.full((B,), int(t1), device=device, dtype=torch.long)

            xt, p1, _ = self.consistency_tools.cm_project_resample_step(
                self,
                graph,
                xt,
                t_graph,
                int(t2),
                active_edge,
                step_idx=i,
                total_steps=infer_T,
            )
            last_p1 = p1

            if int(t2) <= 0:
                break

        if last_p1 is None:
            raise RuntimeError("Assignment diffusion produced no probability output.")

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
