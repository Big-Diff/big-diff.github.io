import torch
from diffusion.consistency.meta import  RowCategoricalConsistencyBase


class CVRPConsistency(RowCategoricalConsistencyBase):
    """Row-categorical consistency utilities for the current CVRP Stage-A model.

    Design goal:
      - diffusion object: customer-row categorical slot label y_i in {0, ..., K_i-1}
      - backbone input: edge-form one-hot state x_{k,i} = 1[k == y_i]
      - model output: edge logits, interpreted row-wise with a softmax over slots per customer
      - training target: weak row-wise CE anchor + sampled pairwise partition loss
      - multi-step sampler: row-categorical posterior / re-noise, not independent edge binary re-noise

    This keeps the current edge-level GNN interface, but aligns the diffusion and
    sampling semantics with the one-of-K assignment problem.
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

    def _slot_count_per_graph(self, graph, veh_cnt: torch.Tensor, bsz: int, device):
        """Return physical Kmax candidate slots for strict CVRP."""
        if hasattr(graph, "K_max") and graph.K_max is not None:
            ku = graph.K_max.view(-1).long().to(device)
            if ku.numel() == 1 and bsz > 1:
                ku = ku.repeat(bsz)
            return torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))

        return veh_cnt.clamp_min(1)

    def _sampled_pairwise_partition_loss(
            self,
            graph,
            row_prob: torch.Tensor,
            k_node: torch.Tensor,
            bsz: int,
            device,
    ):
        """Batched sampled pairwise partition loss.

        Assumption:
            The batch contains fixed-size CVRP instances, e.g. all CVRP100.
            Therefore row_prob can be reshaped as [B, N, K].

        Slot-permutation invariant objective:
            s_ij = sum_k P[i, k] P[j, k]

        The target only says whether two customers are in the same GT route.
        It does not supervise a specific slot id.
        """
        if (not hasattr(graph, "y")) or graph.y is None or row_prob.numel() == 0:
            return row_prob.sum() * 0.0

        B = int(bsz)
        total_nodes = int(row_prob.size(0))

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0:
            return row_prob.sum() * 0.0

        N = total_nodes // B
        if N < 2:
            return row_prob.sum() * 0.0

        pos_samples = int(getattr(self.args, "pair_pos_samples", getattr(self.args, "cm_pair_pos_samples", 128)))
        neg_samples = int(getattr(self.args, "pair_neg_samples", getattr(self.args, "cm_pair_neg_samples", 128)))
        pos_samples = max(0, pos_samples)
        neg_samples = max(0, neg_samples)
        eps = float(getattr(self.args, "pair_eps", 1e-6))

        if pos_samples <= 0 and neg_samples <= 0:
            return row_prob.sum() * 0.0

        # For fixed CVRP100, K is normally constant in the whole batch.
        # This keeps compatibility with the old code that sliced by k_node.
        K_max = int(row_prob.size(-1))
        if k_node is not None and k_node.numel() > 0:
            K = int(k_node.view(-1)[0].detach().item())
            K = max(1, min(K, K_max))
        else:
            K = K_max

        P = row_prob[:, :K].reshape(B, N, K).float()
        Y = graph.y.view(-1).long().to(device)
        Y = torch.clamp_min(Y, 0)
        Y = torch.minimum(Y, torch.full_like(Y, K - 1))
        Y = Y.reshape(B, N)

        # Optional debug check. Keep it off during normal training.
        if bool(getattr(self.args, "pair_debug_assert_batch_order", False)):
            if hasattr(graph, "node_batch") and graph.node_batch is not None:
                expected = torch.arange(B, device=device).repeat_interleave(N)
                actual = graph.node_batch.view(-1).long().to(device)
                assert actual.numel() == expected.numel()
                assert torch.equal(actual, expected), (
                    "Batched pairwise loss assumes PyG-style contiguous graph order: "
                    "[graph0 nodes][graph1 nodes]..."
                )

        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # All unordered customer pairs inside one CVRP100 instance.
        pair_i_all, pair_j_all = torch.triu_indices(N, N, offset=1, device=device)
        M = pair_i_all.numel()

        # same_mask: [B, M]
        same_mask = Y[:, pair_i_all].eq(Y[:, pair_j_all])
        pos_w = same_mask.float()
        neg_w = (~same_mask).float()

        valid = torch.ones((B,), dtype=torch.bool, device=device)
        if pos_samples > 0:
            valid = valid & (pos_w.sum(dim=-1) > 0)
        if neg_samples > 0:
            valid = valid & (neg_w.sum(dim=-1) > 0)

        if not bool(valid.any()):
            return row_prob.sum() * 0.0

        P = P[valid]
        pos_w = pos_w[valid]
        neg_w = neg_w[valid]

        Bv = int(P.size(0))
        batch_arange = torch.arange(Bv, device=device).view(Bv, 1)

        prob_chunks = []
        target_chunks = []

        if pos_samples > 0:
            # [Bv, pos_samples], sampled uniformly from positive pair pool per graph.
            pos_sel = torch.multinomial(pos_w, pos_samples, replacement=True)

            pos_i = pair_i_all[pos_sel]
            pos_j = pair_j_all[pos_sel]

            Pi = P[batch_arange, pos_i]
            Pj = P[batch_arange, pos_j]

            pos_prob = (Pi * Pj).sum(dim=-1)
            prob_chunks.append(pos_prob)
            target_chunks.append(torch.ones_like(pos_prob))

        if neg_samples > 0:
            # [Bv, neg_samples], sampled uniformly from negative pair pool per graph.
            neg_sel = torch.multinomial(neg_w, neg_samples, replacement=True)

            neg_i = pair_i_all[neg_sel]
            neg_j = pair_j_all[neg_sel]

            Pi = P[batch_arange, neg_i]
            Pj = P[batch_arange, neg_j]

            neg_prob = (Pi * Pj).sum(dim=-1)
            prob_chunks.append(neg_prob)
            target_chunks.append(torch.zeros_like(neg_prob))

        same_prob = torch.cat(prob_chunks, dim=1)
        target = torch.cat(target_chunks, dim=1)

        same_prob = torch.nan_to_num(same_prob, nan=0.5, posinf=1.0, neginf=0.0)
        same_prob = same_prob.clamp(eps, 1.0 - eps)

        # AMP-safe BCE on probabilities.
        loss_mat = -(
                target * torch.log(same_prob)
                + (1.0 - target) * torch.log1p(-same_prob)
        )

        loss = loss_mat.mean()

        return loss


    def consistency_losses(self, model, batch):
        device = model.device
        graph = batch.to(device)

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]

        bsz = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0
        if bsz <= 0:
            return graph.node_features.sum() * 0.0

        _, _, edge_graph, veh_local, active_edge, ku, k_node = self._batch_structure(graph, src, dst, bsz, device)
        if not bool(active_edge.any()):
            return graph.node_features.sum() * 0.0

        y0_row = graph.y.view(-1).long().to(device)
        y0_row = torch.minimum(torch.clamp_min(y0_row, 0), k_node - 1)

        t_max = int(model.diffusion.T)
        t_graph = torch.randint(1, t_max + 1, (bsz,), device=device)
        t2_graph = (float(getattr(model.args, "alpha", 0.5)) * t_graph.float()).long().clamp(min=0)

        t_node = t_graph[node_batch]
        t2_node = t2_graph[node_batch]

        y_t = self.q_sample_row(y0_row=y0_row, t_node=t_node, k_node=k_node, diffusion=model.diffusion)
        y_t2 = self.q_sample_row(y0_row=y0_row, t_node=t2_node, k_node=k_node, diffusion=model.diffusion)

        x_t_edge = self._edge_onehot_from_row(y_t, veh_local, dst, active_edge)
        x_t2_edge = self._edge_onehot_from_row(y_t2, veh_local, dst, active_edge)

        xt_in = self._edge_logits_to_input(x_t_edge)
        xt2_in = self._edge_logits_to_input(x_t2_edge)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits_t = model.forward(graph, xt_in.float(), t_graph).float()
            logits_t2 = model.forward(graph, xt2_in.float(), t2_graph).float()

        logits_t = torch.nan_to_num(logits_t, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        logits_t2 = torch.nan_to_num(logits_t2, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        row_ce_t = self._row_ce_from_logits(
            graph, logits_t, active_edge, src, dst, edge_graph, bsz, device
        )
        row_ce_t2 = self._row_ce_from_logits(
            graph, logits_t2, active_edge, src, dst, edge_graph, bsz, device
        )

        _, row_prob_t, _ = self._row_prob_from_logits(
            graph, logits_t, active_edge, src, dst, veh_local, k_node, bsz
        )
        _, row_prob_t2, _ = self._row_prob_from_logits(
            graph, logits_t2, active_edge, src, dst, veh_local, k_node, bsz
        )
        pair_loss_t = self._sampled_pairwise_partition_loss(
            graph, row_prob_t, k_node, bsz, device
        )
        pair_loss_t2 = self._sampled_pairwise_partition_loss(
            graph, row_prob_t2, k_node, bsz, device
        )

        # New objective semantics:
        #   - row CE is only a weak slot-anchor, preventing early slot drift;
        #   - pairwise partition loss is the main permutation-invariant signal;
        #   - consistency KL is optional and remains disabled by default.
        lam_row = float(getattr(model.args, "lam_row", getattr(model.args, "cm_lam_row", 0.1)) or 0.0)
        lam_pair = float(getattr(model.args, "lam_pair", getattr(model.args, "cm_lam_pair", 1.0)) or 0.0)
        lam_cons = float(getattr(model.args, "lam_cons", getattr(model.args, "cm_lam_cons", 0.0)) or 0.0)

        cons_kl = self._row_consistency_kl(
            graph, logits_t, logits_t2, active_edge, src, dst, veh_local, k_node, bsz
        ) if lam_cons > 0 else logits_t.sum() * 0.0

        row_loss = 0.5 * (row_ce_t + row_ce_t2)
        pair_loss = 0.5 * (pair_loss_t + pair_loss_t2)

        loss = (
            lam_pair * pair_loss
            + lam_row * row_loss
            + lam_cons * cons_kl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "pairwise-partition consistency loss became non-finite: "
                f"row_loss={row_loss.item()}, pair_loss={pair_loss.item()}, cons_kl={cons_kl.item()}"
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
