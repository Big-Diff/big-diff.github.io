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

    # def _sampled_pairwise_partition_loss(
    #         self,
    #         graph,
    #         row_prob: torch.Tensor,
    #         k_node: torch.Tensor,
    #         bsz: int,
    #         device,
    # ):
    #     """Batched sampled pairwise partition loss.
    #
    #     Assumption:
    #         The batch contains fixed-size CVRP instances, e.g. all CVRP100.
    #         Therefore row_prob can be reshaped as [B, N, K].
    #
    #     Slot-permutation invariant objective:
    #         s_ij = sum_k P[i, k] P[j, k]
    #
    #     The target only says whether two customers are in the same GT route.
    #     It does not supervise a specific slot id.
    #     """
    #     if (not hasattr(graph, "y")) or graph.y is None or row_prob.numel() == 0:
    #         return row_prob.sum() * 0.0
    #
    #     B = int(bsz)
    #     total_nodes = int(row_prob.size(0))
    #
    #     if B <= 0 or total_nodes <= 0 or total_nodes % B != 0:
    #         return row_prob.sum() * 0.0
    #
    #     N = total_nodes // B
    #     if N < 2:
    #         return row_prob.sum() * 0.0
    #
    #     pos_samples = int(getattr(self.args, "pair_pos_samples", getattr(self.args, "cm_pair_pos_samples", 128)))
    #     neg_samples = int(getattr(self.args, "pair_neg_samples", getattr(self.args, "cm_pair_neg_samples", 128)))
    #     pos_samples = max(0, pos_samples)
    #     neg_samples = max(0, neg_samples)
    #     eps = float(getattr(self.args, "pair_eps", 1e-6))
    #
    #     if pos_samples <= 0 and neg_samples <= 0:
    #         return row_prob.sum() * 0.0
    #
    #     # For fixed CVRP100, K is normally constant in the whole batch.
    #     # This keeps compatibility with the old code that sliced by k_node.
    #     K_max = int(row_prob.size(-1))
    #     if k_node is not None and k_node.numel() > 0:
    #         K = int(k_node.view(-1)[0].detach().item())
    #         K = max(1, min(K, K_max))
    #     else:
    #         K = K_max
    #
    #     P = row_prob[:, :K].reshape(B, N, K).float()
    #     Y = graph.y.view(-1).long().to(device)
    #     Y = torch.clamp_min(Y, 0)
    #     Y = torch.minimum(Y, torch.full_like(Y, K - 1))
    #     Y = Y.reshape(B, N)
    #
    #     # Optional debug check. Keep it off during normal training.
    #     if bool(getattr(self.args, "pair_debug_assert_batch_order", False)):
    #         if hasattr(graph, "node_batch") and graph.node_batch is not None:
    #             expected = torch.arange(B, device=device).repeat_interleave(N)
    #             actual = graph.node_batch.view(-1).long().to(device)
    #             assert actual.numel() == expected.numel()
    #             assert torch.equal(actual, expected), (
    #                 "Batched pairwise loss assumes PyG-style contiguous graph order: "
    #                 "[graph0 nodes][graph1 nodes]..."
    #             )
    #
    #     P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    #     P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    #
    #     # All unordered customer pairs inside one CVRP100 instance.
    #     pair_i_all, pair_j_all = torch.triu_indices(N, N, offset=1, device=device)
    #
    #     # same_mask: [B, M]
    #     same_mask = Y[:, pair_i_all].eq(Y[:, pair_j_all])
    #     pos_w = same_mask.float()
    #     neg_w = (~same_mask).float()
    #
    #     valid = torch.ones((B,), dtype=torch.bool, device=device)
    #     if pos_samples > 0:
    #         valid = valid & (pos_w.sum(dim=-1) > 0)
    #     if neg_samples > 0:
    #         valid = valid & (neg_w.sum(dim=-1) > 0)
    #
    #     if not bool(valid.any()):
    #         return row_prob.sum() * 0.0
    #
    #     P = P[valid]
    #     pos_w = pos_w[valid]
    #     neg_w = neg_w[valid]
    #
    #     Bv = int(P.size(0))
    #     batch_arange = torch.arange(Bv, device=device).view(Bv, 1)
    #
    #     prob_chunks = []
    #     target_chunks = []
    #
    #     if pos_samples > 0:
    #         # [Bv, pos_samples], sampled uniformly from positive pair pool per graph.
    #         pos_sel = torch.multinomial(pos_w, pos_samples, replacement=True)
    #
    #         pos_i = pair_i_all[pos_sel]
    #         pos_j = pair_j_all[pos_sel]
    #
    #         Pi = P[batch_arange, pos_i]
    #         Pj = P[batch_arange, pos_j]
    #
    #         pos_prob = (Pi * Pj).sum(dim=-1)
    #         prob_chunks.append(pos_prob)
    #         target_chunks.append(torch.ones_like(pos_prob))
    #
    #     if neg_samples > 0:
    #         # [Bv, neg_samples], sampled uniformly from negative pair pool per graph.
    #         neg_sel = torch.multinomial(neg_w, neg_samples, replacement=True)
    #
    #         neg_i = pair_i_all[neg_sel]
    #         neg_j = pair_j_all[neg_sel]
    #
    #         Pi = P[batch_arange, neg_i]
    #         Pj = P[batch_arange, neg_j]
    #
    #         neg_prob = (Pi * Pj).sum(dim=-1)
    #         prob_chunks.append(neg_prob)
    #         target_chunks.append(torch.zeros_like(neg_prob))
    #
    #     same_prob = torch.cat(prob_chunks, dim=1)
    #     target = torch.cat(target_chunks, dim=1)
    #
    #     same_prob = torch.nan_to_num(same_prob, nan=0.5, posinf=1.0, neginf=0.0)
    #     same_prob = same_prob.clamp(eps, 1.0 - eps)
    #
    #     # AMP-safe BCE on probabilities.
    #     loss_mat = -(
    #             target * torch.log(same_prob)
    #             + (1.0 - target) * torch.log1p(-same_prob)
    #     )
    #
    #     loss = loss_mat.mean()
    #
    #     return loss
    def _sampled_pairwise_partition_loss(self, graph, row_prob, k_node, bsz, device):
        """Anchor-based InfoNCE pair loss; slot-permutation invariant."""
        if (not hasattr(graph, "y")) or graph.y is None or row_prob.numel() == 0:
            return row_prob.sum() * 0.0

        B = int(bsz)
        total_nodes = int(row_prob.size(0))
        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0:
            return row_prob.sum() * 0.0

        N = total_nodes // B
        if N < 2:
            return row_prob.sum() * 0.0

        anchor_samples = int(getattr(self.args, "pair_nce_anchor_samples", 128))
        pos_per_anchor = int(getattr(self.args, "pair_nce_pos_per_anchor", 1))
        neg_per_anchor = int(getattr(self.args, "pair_nce_negatives", 16))
        tau = max(float(getattr(self.args, "pair_nce_tau", 0.1)), 1e-6)

        if anchor_samples <= 0 or pos_per_anchor <= 0 or neg_per_anchor <= 0:
            return row_prob.sum() * 0.0

        Kmax = int(row_prob.size(-1))
        if k_node is not None and k_node.numel() > 0:
            K = int(k_node.view(-1)[0].detach().item())
            K = max(1, min(K, Kmax))
        else:
            K = Kmax

        P = row_prob[:, :K].reshape(B, N, K).float()
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        Y = graph.y.view(-1).long().to(device).clamp(0, K - 1).view(B, N)

        same_route = Y.unsqueeze(2).eq(Y.unsqueeze(1))
        eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)

        pos_mask = same_route & ~eye
        neg_mask = ~same_route

        anchor_w = (pos_mask.any(dim=-1) & neg_mask.any(dim=-1)).float()
        valid = anchor_w.sum(dim=-1) > 0

        if not bool(valid.any()):
            return row_prob.sum() * 0.0

        P = P[valid]
        pos_mask = pos_mask[valid]
        neg_mask = neg_mask[valid]
        anchor_w = anchor_w[valid]

        Bv = int(P.size(0))
        batch2 = torch.arange(Bv, device=device).view(Bv, 1)
        batch3 = torch.arange(Bv, device=device).view(Bv, 1, 1)

        anchor_idx = torch.multinomial(anchor_w, anchor_samples, replacement=True)

        pos_w = pos_mask[batch2, anchor_idx].float()
        neg_w = neg_mask[batch2, anchor_idx].float()

        pos_idx = torch.multinomial(
            pos_w.reshape(-1, N),
            pos_per_anchor,
            replacement=True,
        ).view(Bv, anchor_samples, pos_per_anchor)

        neg_idx = torch.multinomial(
            neg_w.reshape(-1, N),
            neg_per_anchor,
            replacement=True,
        ).view(Bv, anchor_samples, neg_per_anchor)

        Pi = P[batch2, anchor_idx]
        Ppos = P[batch3, pos_idx]
        Pneg = P[batch3, neg_idx]

        pos_logits = (Pi.unsqueeze(2) * Ppos).sum(dim=-1)
        neg_logits = (Pi.unsqueeze(2) * Pneg).sum(dim=-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1) / tau
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        log_pos = torch.logsumexp(logits[:, :, :pos_per_anchor], dim=-1)
        log_all = torch.logsumexp(logits, dim=-1)

        return -(log_pos - log_all).mean()

    def consistency_losses(self, model, batch):
        ctx = self._two_time_row_outputs(model, batch)
        if ctx is None:
            return batch.node_features.sum() * 0.0

        graph = ctx["graph"]
        device = ctx["device"]
        bsz = ctx["bsz"]

        pair_loss_t = self._sampled_pairwise_partition_loss(
            graph, ctx["row_prob_t"], ctx["k_node"], bsz, device
        )
        pair_loss_t2 = self._sampled_pairwise_partition_loss(
            graph, ctx["row_prob_t2"], ctx["k_node"], bsz, device
        )

        row_loss = ctx["row_ce"]
        pair_loss = 0.5 * (pair_loss_t + pair_loss_t2)

        lam_row = float(getattr(model.args, "lam_row", getattr(model.args, "cm_lam_row", 0.1)) or 0.0)
        lam_pair = float(getattr(model.args, "lam_pair", getattr(model.args, "cm_lam_pair", 1.0)) or 0.0)
        lam_cons = float(getattr(model.args, "lam_cons", getattr(model.args, "cm_lam_cons", 0.0)) or 0.0)

        cons_kl = (
            self._two_time_consistency_kl(ctx)
            if lam_cons > 0
            else ctx["logits_t"].sum() * 0.0
        )

        loss = lam_pair * pair_loss + lam_row * row_loss + lam_cons * cons_kl

        if not torch.isfinite(loss):
            raise RuntimeError(
                "CVRP consistency loss became non-finite: "
                f"row_loss={float(row_loss.detach().cpu())}, "
                f"pair_loss={float(pair_loss.detach().cpu())}, "
                f"cons_kl={float(cons_kl.detach().cpu())}"
            )

        if hasattr(model, "log_dict"):
            model.log_dict(
                {
                    "train/cm_loss_pair_main": pair_loss.detach(),
                    "train/cm_loss_row_anchor": row_loss.detach(),
                    "train/cm_row_ce": row_loss.detach(),
                    "train/cm_cons_kl": cons_kl.detach(),
                    "train/cm_lam_pair": torch.tensor(lam_pair, device=device),
                    "train/cm_lam_row": torch.tensor(lam_row, device=device),
                    "train/cm_lam_cons": torch.tensor(lam_cons, device=device),
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(bsz),
            )
            model.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                      batch_size=int(bsz))

        return loss