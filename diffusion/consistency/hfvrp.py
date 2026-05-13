import torch
import torch.nn.functional as F

from .meta import RowCategoricalConsistencyBase


class HFVRPConsistency(RowCategoricalConsistencyBase):
    """Row-categorical consistency utilities for strict HFVRP Stage-A.

    Mainline semantics:
      - diffusion variable: customer-row categorical slot label y_i in {0, ..., K_i - 1};
      - GNN input: edge-form one-hot state x_{k,i} = 1[k == y_i];
      - GNN output: edge logits, interpreted row-wise with softmax over vehicle slots;
      - training objective: type CE + within-type pairwise partition loss + weak row CE
        + optional two-time consistency KL;
      - sampler: row-categorical posterior / re-noise.
    """
    def __init__(
        self,
        args,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        super().__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )
        self.args = args
    # ---------------------------------------------------------------------
    # Common row/edge utilities
    # ---------------------------------------------------------------------
    def _slot_count_per_graph(self, graph, veh_cnt, bsz, device):
        if hasattr(graph, "num_vehicle_slots") and graph.num_vehicle_slots is not None:
            ku = graph.num_vehicle_slots.view(-1).long().to(device)
        elif hasattr(graph, "K_used") and graph.K_used is not None:
            ku = graph.K_used.view(-1).long().to(device)
        else:
            ku = veh_cnt.long().to(device)

        if ku.numel() == 1 and bsz > 1:
            ku = ku.repeat(bsz)

        return torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))

    def _get_batched_vehicle_tier(self, graph, B: int, Kmax: int, device):
        """Return vehicle tier ids with shape [B, Kmax]."""
        tier = graph.vehicle_tier.view(-1).long().to(device)

        if tier.numel() != B * Kmax:
            raise RuntimeError(
                f"vehicle_tier size mismatch: got {tier.numel()}, "
                f"expected B*Kmax={B * Kmax}."
            )

        return tier.view(B, Kmax)
    # ---------------------------------------------------------------------
    # Row-wise training objective
    # ---------------------------------------------------------------------
    def _type_ce_from_row_prob(
        self,
        graph,
        row_prob: torch.Tensor,
        k_node: torch.Tensor,
        bsz: int,
        device,
    ):
        """Vehicle-type CE induced by row-wise slot probabilities."""
        del k_node

        if row_prob.numel() == 0:
            return row_prob.sum() * 0.0

        B = int(bsz)
        total_nodes = int(row_prob.size(0))
        Kmax = int(row_prob.size(-1))

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0 or Kmax <= 0:
            return row_prob.sum() * 0.0

        N = total_nodes // B

        P = torch.nan_to_num(
            row_prob[:, :Kmax].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P.view(B, N, Kmax)

        tier = self._get_batched_vehicle_tier(graph, B, Kmax, device)

        uniq = torch.unique(tier, sorted=True)
        G = int(uniq.numel())

        if G <= 0:
            return row_prob.sum() * 0.0

        tier_remap = torch.searchsorted(uniq, tier)

        y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)
        batch_ids = torch.arange(B, device=device).view(B, 1)
        target_tier = tier_remap[batch_ids, y_slot]

        P_type = P.new_zeros((B, N, G))
        type_idx = tier_remap.view(B, 1, Kmax).expand(B, N, Kmax)
        P_type.scatter_add_(2, type_idx, P)
        P_type = P_type / P_type.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        logp = torch.log(P_type.clamp_min(1e-12)).view(B * N, G)
        target = target_tier.reshape(B * N).long()

        return F.nll_loss(logp, target)

    def _sampled_pairwise_partition_loss_within_type(
        self,
        graph,
        row_prob: torch.Tensor,
        k_node: torch.Tensor,
        bsz: int,
        device,
    ):
        """Sampled same-slot partition loss inside each vehicle type."""
        del k_node

        if row_prob.numel() == 0:
            return row_prob.sum() * 0.0

        B = int(bsz)
        total_nodes = int(row_prob.size(0))
        Kmax = int(row_prob.size(-1))

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0 or Kmax <= 0:
            return row_prob.sum() * 0.0

        N = total_nodes // B
        if N < 2:
            return row_prob.sum() * 0.0

        pos_samples = int(getattr(self.args, "hf_pair_pos_samples", 128))
        neg_samples = int(getattr(self.args, "hf_pair_neg_samples", 128))
        pos_samples = max(0, pos_samples)
        neg_samples = max(0, neg_samples)

        if pos_samples <= 0 and neg_samples <= 0:
            return row_prob.sum() * 0.0

        eps = float(getattr(self.args, "pair_eps", 1e-6))

        P = torch.nan_to_num(
            row_prob[:, :Kmax].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P.view(B, N, Kmax)

        y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)

        tier = self._get_batched_vehicle_tier(graph, B, Kmax, device)
        uniq = torch.unique(tier, sorted=True)
        tier = torch.searchsorted(uniq, tier)

        batch_ids_full = torch.arange(B, device=device).view(B, 1)
        target_tier = tier[batch_ids_full, y_slot]

        pair_i_all, pair_j_all = torch.triu_indices(N, N, offset=1, device=device)

        same_type = target_tier[:, pair_i_all].eq(target_tier[:, pair_j_all])
        same_slot = y_slot[:, pair_i_all].eq(y_slot[:, pair_j_all])

        pos_w = (same_type & same_slot).float()
        neg_w = (same_type & (~same_slot)).float()

        valid_graph = torch.ones((B,), dtype=torch.bool, device=device)
        if pos_samples > 0:
            valid_graph = valid_graph & (pos_w.sum(dim=-1) > 0)
        if neg_samples > 0:
            valid_graph = valid_graph & (neg_w.sum(dim=-1) > 0)

        if not bool(valid_graph.any()):
            return row_prob.sum() * 0.0

        P = P[valid_graph]
        target_tier = target_tier[valid_graph]
        tier = tier[valid_graph]
        pos_w = pos_w[valid_graph]
        neg_w = neg_w[valid_graph]

        Bv = int(P.size(0))
        batch_ids = torch.arange(Bv, device=device).view(Bv, 1)

        def _same_prob_for_selected(sel: torch.Tensor) -> torch.Tensor:
            ii = pair_i_all[sel]
            jj = pair_j_all[sel]

            Pi = P[batch_ids, ii]
            Pj = P[batch_ids, jj]

            type_ij = target_tier[batch_ids, ii]
            mask_k = tier.unsqueeze(1).eq(type_ij.unsqueeze(-1))

            Pi_t = Pi.masked_fill(~mask_k, 0.0)
            Pj_t = Pj.masked_fill(~mask_k, 0.0)

            Pi_t = Pi_t / Pi_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            Pj_t = Pj_t / Pj_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            return (Pi_t * Pj_t).sum(dim=-1)

        prob_chunks = []
        target_chunks = []

        if pos_samples > 0:
            pos_sel = torch.multinomial(pos_w, pos_samples, replacement=True)
            pos_prob = _same_prob_for_selected(pos_sel)
            prob_chunks.append(pos_prob)
            target_chunks.append(torch.ones_like(pos_prob))

        if neg_samples > 0:
            neg_sel = torch.multinomial(neg_w, neg_samples, replacement=True)
            neg_prob = _same_prob_for_selected(neg_sel)
            prob_chunks.append(neg_prob)
            target_chunks.append(torch.zeros_like(neg_prob))

        same_prob = torch.cat(prob_chunks, dim=1)
        target = torch.cat(target_chunks, dim=1)

        same_prob = torch.nan_to_num(
            same_prob,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(eps, 1.0 - eps)

        loss = -(
            target * torch.log(same_prob)
            + (1.0 - target) * torch.log1p(-same_prob)
        )

        return loss.mean()

    def consistency_losses(self, model, batch):
        device = model.device
        graph = batch.to(device)

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]

        bsz = int(getattr(graph, "num_graphs", 0) or 0)
        if bsz <= 0:
            bsz = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0

        if bsz <= 0:
            return graph.node_features.sum() * 0.0

        _, _, edge_graph, veh_local, active_edge, _, k_node = self._batch_structure(
            graph,
            src,
            dst,
            bsz,
            device,
        )

        if not bool(active_edge.any()):
            return graph.node_features.sum() * 0.0

        y0_row = graph.y.view(-1).long().to(device)
        y0_row = torch.minimum(torch.clamp_min(y0_row, 0), k_node - 1)

        t_max = int(model.diffusion.T)

        t_graph = torch.randint(1, t_max + 1, (bsz,), device=device)
        t2_graph = (
            float(getattr(model.args, "alpha", 0.5))
            * t_graph.float()
        ).long().clamp(min=0)

        t_node = t_graph[node_batch]
        t2_node = t2_graph[node_batch]

        y_t = self.q_sample_row(
            y0_row=y0_row,
            t_node=t_node,
            k_node=k_node,
            diffusion=model.diffusion,
        )
        y_t2 = self.q_sample_row(
            y0_row=y0_row,
            t_node=t2_node,
            k_node=k_node,
            diffusion=model.diffusion,
        )

        x_t_edge = self._edge_onehot_from_row(
            y_t,
            veh_local,
            dst,
            active_edge,
        )
        x_t2_edge = self._edge_onehot_from_row(
            y_t2,
            veh_local,
            dst,
            active_edge,
        )

        xt_in = self._edge_logits_to_input(x_t_edge)
        xt2_in = self._edge_logits_to_input(x_t2_edge)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits_t = model.forward(graph, xt_in.float(), t_graph).float()
            logits_t2 = model.forward(graph, xt2_in.float(), t2_graph).float()

        logits_t = torch.nan_to_num(
            logits_t,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        logits_t2 = torch.nan_to_num(
            logits_t2,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        row_ce_t = self._row_ce_from_logits(
            graph,
            logits_t,
            active_edge,
            src,
            dst,
            edge_graph,
            bsz,
            device,
        )
        row_ce_t2 = self._row_ce_from_logits(
            graph,
            logits_t2,
            active_edge,
            src,
            dst,
            edge_graph,
            bsz,
            device,
        )
        row_ce = 0.5 * (row_ce_t + row_ce_t2)

        _, row_prob_t, _ = self._row_prob_from_logits(
            graph,
            logits_t,
            active_edge,
            src,
            dst,
            veh_local,
            k_node,
            bsz,
        )
        _, row_prob_t2, _ = self._row_prob_from_logits(
            graph,
            logits_t2,
            active_edge,
            src,
            dst,
            veh_local,
            k_node,
            bsz,
        )

        type_loss_t = self._type_ce_from_row_prob(
            graph,
            row_prob_t,
            k_node,
            bsz,
            device,
        )
        type_loss_t2 = self._type_ce_from_row_prob(
            graph,
            row_prob_t2,
            k_node,
            bsz,
            device,
        )
        type_loss = 0.5 * (type_loss_t + type_loss_t2)

        pair_loss_t = self._sampled_pairwise_partition_loss_within_type(
            graph,
            row_prob_t,
            k_node,
            bsz,
            device,
        )
        pair_loss_t2 = self._sampled_pairwise_partition_loss_within_type(
            graph,
            row_prob_t2,
            k_node,
            bsz,
            device,
        )
        pair_loss = 0.5 * (pair_loss_t + pair_loss_t2)

        lam_type = float(getattr(model.args, "hf_lam_type", 1.0))
        lam_pair = float(getattr(model.args, "hf_lam_pair", 1.0))
        lam_row = float(getattr(model.args, "hf_lam_row", 0.10))
        lam_cons = float(getattr(model.args, "hf_lam_cons", 0.0))

        if lam_cons > 0:
            cons_kl = self._row_consistency_kl(
                graph,
                logits_t,
                logits_t2,
                active_edge,
                src,
                dst,
                veh_local,
                k_node,
                bsz,
            )
        else:
            cons_kl = logits_t.sum() * 0.0

        loss = (
            lam_type * type_loss
            + lam_pair * pair_loss
            + lam_row * row_ce
            + lam_cons * cons_kl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "HFVRP loss became non-finite: "
                f"type_loss={float(type_loss.detach().cpu())}, "
                f"pair_loss={float(pair_loss.detach().cpu())}, "
                f"row_ce={float(row_ce.detach().cpu())}, "
                f"cons_kl={float(cons_kl.detach().cpu())}"
            )

        if hasattr(model, "log_dict"):
            model.log_dict(
                {"train/loss": loss.detach()},
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return loss