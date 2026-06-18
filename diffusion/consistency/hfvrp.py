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
        elif hasattr(graph, "max_vehicles") and graph.max_vehicles is not None:
            ku = graph.max_vehicles.view(-1).long().to(device)
        else:
            ku = veh_cnt.long().to(device)

        if ku.numel() == 1 and bsz > 1:
            ku = ku.repeat(bsz)

        return torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))

    def _get_batched_slot_group(self, graph, B: int, Kmax: int, device):
        if hasattr(graph, "slot_group") and graph.slot_group is not None:
            group = graph.slot_group.view(-1).long().to(device)
        elif hasattr(graph, "vehicle_tier") and graph.vehicle_tier is not None:
            group = graph.vehicle_tier.view(-1).long().to(device)
        else:
            raise RuntimeError("HFVRP type-aware loss requires slot_group or vehicle_tier.")

        if group.numel() == B * Kmax:
            return group.view(B, Kmax)

        if group.numel() >= Kmax:
            return group[:Kmax].view(1, Kmax).expand(B, Kmax)

        raise RuntimeError(
            f"slot group size mismatch: got {group.numel()}, expected {Kmax} or {B * Kmax}."
        )

    def _active_edge_mask(
            self,
            graph,
            src: torch.Tensor,
            dst: torch.Tensor,
            edge_graph: torch.Tensor,
            veh_local: torch.Tensor,
            bsz: int,
            device,
    ) -> torch.Tensor:
        del dst, edge_graph, veh_local, bsz

        if hasattr(graph, "vehicle_available_mask") and graph.vehicle_available_mask is not None:
            mask_v = graph.vehicle_available_mask.view(-1).to(device).bool()
            return mask_v[src.long().to(device)]

        return torch.ones_like(src, dtype=torch.bool, device=device)

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

        slot_group = self._get_batched_slot_group(graph, B, Kmax, device)
        uniq = torch.unique(slot_group, sorted=True)
        G = int(uniq.numel())

        if G <= 0:
            return row_prob.sum() * 0.0

        slot_group_remap = torch.searchsorted(uniq, slot_group)

        if hasattr(graph, "y_group") and graph.y_group is not None:
            target_group_raw = graph.y_group.view(-1).long().to(device).view(B, N)
            target_group = torch.searchsorted(uniq, target_group_raw)
        else:
            y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)
            batch_ids = torch.arange(B, device=device).view(B, 1)
            target_group = slot_group_remap[batch_ids, y_slot]

        P_type = P.new_zeros((B, N, G))
        type_idx = slot_group_remap.view(B, 1, Kmax).expand(B, N, Kmax)
        P_type.scatter_add_(2, type_idx, P)
        P_type = P_type / P_type.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        logp = torch.log(P_type.clamp_min(1e-12)).view(B * N, G)
        target = target_group.reshape(B * N).long()

        return F.nll_loss(logp, target)

    def _sampled_pairwise_partition_loss_within_type(self, graph, row_prob, k_node, bsz, device):
        """Anchor-based InfoNCE pair loss inside each vehicle type."""
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

        anchor_samples = int(getattr(self.args, "hf_pair_nce_anchor_samples", 128))
        pos_per_anchor = int(getattr(self.args, "hf_pair_nce_pos_per_anchor", 1))
        neg_per_anchor = int(getattr(self.args, "hf_pair_nce_negatives", 16))
        tau = max(float(getattr(self.args, "hf_pair_nce_tau", 0.1)), 1e-6)

        if anchor_samples <= 0 or pos_per_anchor <= 0 or neg_per_anchor <= 0:
            return row_prob.sum() * 0.0

        P = torch.nan_to_num(row_prob[:, :Kmax].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P.view(B, N, Kmax)

        y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)

        slot_group = self._get_batched_slot_group(graph, B, Kmax, device)
        uniq = torch.unique(slot_group, sorted=True)
        slot_group_remap = torch.searchsorted(uniq, slot_group)

        if hasattr(graph, "y_group") and graph.y_group is not None:
            target_group_raw = graph.y_group.view(-1).long().to(device).view(B, N)
            target_group = torch.searchsorted(uniq, target_group_raw)
        else:
            batch_ids_full = torch.arange(B, device=device).view(B, 1)
            target_group = slot_group_remap[batch_ids_full, y_slot]

        same_type = target_group.unsqueeze(2).eq(target_group.unsqueeze(1))
        same_slot = y_slot.unsqueeze(2).eq(y_slot.unsqueeze(1))
        eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)

        pos_mask = same_type & same_slot & ~eye
        neg_mask = same_type & ~same_slot

        anchor_w = (pos_mask.any(dim=-1) & neg_mask.any(dim=-1)).float()
        valid = anchor_w.sum(dim=-1) > 0

        if not bool(valid.any()):
            return row_prob.sum() * 0.0

        P = P[valid]
        pos_mask = pos_mask[valid]
        neg_mask = neg_mask[valid]
        anchor_w = anchor_w[valid]
        target_group = target_group[valid]
        slot_group_remap = slot_group_remap[valid]

        Bv = int(P.size(0))
        batch2 = torch.arange(Bv, device=device).view(Bv, 1)
        batch3 = torch.arange(Bv, device=device).view(Bv, 1, 1)

        anchor_idx = torch.multinomial(anchor_w, anchor_samples, replacement=True)

        pos_w = pos_mask[batch2, anchor_idx].float()
        neg_w = neg_mask[batch2, anchor_idx].float()

        pos_idx = torch.multinomial(pos_w.reshape(-1, N), pos_per_anchor, replacement=True).view(Bv, anchor_samples,
                                                                                                 pos_per_anchor)
        neg_idx = torch.multinomial(neg_w.reshape(-1, N), neg_per_anchor, replacement=True).view(Bv, anchor_samples,
                                                                                                 neg_per_anchor)

        Pi = P[batch2, anchor_idx]
        Ppos = P[batch3, pos_idx]
        Pneg = P[batch3, neg_idx]

        anchor_group = target_group[batch2, anchor_idx]
        group_mask = slot_group_remap.unsqueeze(1).eq(anchor_group.unsqueeze(-1))

        Pi = Pi.masked_fill(~group_mask, 0.0)
        Pi = Pi / Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        Ppos = Ppos.masked_fill(~group_mask.unsqueeze(2), 0.0)
        Ppos = Ppos / Ppos.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        Pneg = Pneg.masked_fill(~group_mask.unsqueeze(2), 0.0)
        Pneg = Pneg / Pneg.sum(dim=-1, keepdim=True).clamp_min(1e-12)

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

        type_loss_t = self._type_ce_from_row_prob(
            graph, ctx["row_prob_t"], ctx["k_node"], bsz, device
        )
        type_loss_t2 = self._type_ce_from_row_prob(
            graph, ctx["row_prob_t2"], ctx["k_node"], bsz, device
        )
        type_loss = 0.5 * (type_loss_t + type_loss_t2)

        pair_loss_t = self._sampled_pairwise_partition_loss_within_type(
            graph, ctx["row_prob_t"], ctx["k_node"], bsz, device
        )
        pair_loss_t2 = self._sampled_pairwise_partition_loss_within_type(
            graph, ctx["row_prob_t2"], ctx["k_node"], bsz, device
        )
        pair_loss = 0.5 * (pair_loss_t + pair_loss_t2)

        row_ce = ctx["row_ce"]

        lam_type = float(getattr(model.args, "hf_lam_type", 1.0))
        lam_pair = float(getattr(model.args, "hf_lam_pair", 1.0))
        lam_row = float(getattr(model.args, "hf_lam_row", 0.10))
        lam_cons = float(getattr(model.args, "hf_lam_cons", 0.0))

        cons_kl = (
            self._two_time_consistency_kl(ctx)
            if lam_cons > 0
            else ctx["logits_t"].sum() * 0.0
        )

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
                {
                    "train/hf_loss": loss.detach(),
                    "train/hf_type_loss": type_loss.detach(),
                    "train/hf_pair_loss": pair_loss.detach(),
                    "train/hf_row_ce": row_ce.detach(),
                    "train/hf_cons_kl": cons_kl.detach(),
                    "train/hf_lam_type": torch.tensor(lam_type, device=device),
                    "train/hf_lam_pair": torch.tensor(lam_pair, device=device),
                    "train/hf_lam_row": torch.tensor(lam_row, device=device),
                    "train/hf_lam_cons": torch.tensor(lam_cons, device=device),
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