import torch
import torch.nn.functional as F
from .meta import MetaConsistency


class HFVRPConsistency(MetaConsistency):
    """Row-categorical consistency utilities for the current HFVRP Stage-A model.

    Design goal:
      - diffusion object: customer-row categorical slot label y_i in {0, ..., K_i-1}
      - backbone input: edge-form one-hot state x_{k,i} = 1[k == y_i]
      - model output: edge logits, interpreted row-wise with a softmax over slots per customer
      - training target: row-wise categorical CE, with optional two-time consistency KL
      - multi-step sampler: row-categorical posterior / re-noise, not independent edge binary re-noise

    This keeps the current edge-level GNN interface, but aligns the diffusion and
    sampling semantics with the one-of-K assignment problem.
    """

    def __init__(self, args, sigma_max=1000, sigma_min=0, weight_schedule="uniform", boundary_func="truncate"):
        super().__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )
        self.args = args
        self._Q_bar_torch = None
        self._Q_torch = None

    # ---------------------------------------------------------------------
    # Diffusion schedule helpers
    # ---------------------------------------------------------------------
    def _ensure_diffusion_mats(self, device, diffusion):
        if self._Q_bar_torch is None or self._Q_bar_torch.device != device:
            self._Q_bar_torch = torch.as_tensor(diffusion.Q_bar, device=device, dtype=torch.float32)
        if self._Q_torch is None or self._Q_torch.device != device:
            self._Q_torch = torch.as_tensor(diffusion.Qs, device=device, dtype=torch.float32)

    def _row_alpha_bar_vec(self, t: torch.Tensor, diffusion, device) -> torch.Tensor:
        """Vectorized cumulative alpha_bar(t) for row-categorical uniform mixing.

        We reuse the shape of the existing binary cumulative schedule and rescale it so:
            alpha_bar(0) = 1, alpha_bar(T) = 0.
        """
        self._ensure_diffusion_mats(device, diffusion)
        T = int(diffusion.T)
        t = t.long().to(device).clamp(0, T)

        raw_t = self._Q_bar_torch[t, 1, 1]
        raw_T = self._Q_bar_torch[T, 1, 1]
        alpha = (raw_t - raw_T) / (1.0 - raw_T).clamp_min(1e-12)
        alpha = alpha.clamp(0.0, 1.0)
        alpha = torch.where(t <= 0, torch.ones_like(alpha), alpha)
        return alpha

    def _row_alpha_step_vec(self, t_prev: torch.Tensor, t_cur: torch.Tensor, diffusion, device) -> torch.Tensor:
        """Vectorized alpha for the transition from t_prev to t_cur, t_prev <= t_cur.

        For uniform-mixing kernels, alpha_bar(t_cur) = alpha_bar(t_prev) * alpha_step.
        """
        t_prev = t_prev.long().to(device)
        t_cur = t_cur.long().to(device)
        a_prev = self._row_alpha_bar_vec(t_prev, diffusion, device)
        a_cur = self._row_alpha_bar_vec(t_cur, diffusion, device)
        step = a_cur / a_prev.clamp_min(1e-12)
        step = step.clamp(0.0, 1.0)
        step = torch.where(t_cur <= t_prev, torch.ones_like(step), step)
        step = torch.where(a_prev <= 1e-12, torch.zeros_like(step), step)
        return step


    # ---------------------------------------------------------------------
    # Row-categorical corruption and posterior sampler
    # ---------------------------------------------------------------------
    def q_sample_row(self, y0_row: torch.Tensor, t_node: torch.Tensor, k_node: torch.Tensor, diffusion) -> torch.Tensor:
        """Vectorized row-categorical corruption.

        q(y_t | y_0) = alpha_bar(t) * onehot(y_0) + (1 - alpha_bar(t)) * Uniform(K_i).

        Equivalent sampler:
          keep y0 with probability alpha_bar(t), otherwise draw uniformly from [0, K_i - 1].
        """
        device = y0_row.device
        y0 = y0_row.long().to(device)
        K = k_node.long().to(device).clamp_min(1)
        alpha = self._row_alpha_bar_vec(t_node, diffusion, device)

        y0_clip = torch.minimum(torch.clamp_min(y0, 0), K - 1)
        rand_cls = torch.floor(torch.rand_like(alpha) * K.float()).long()
        rand_cls = torch.minimum(torch.clamp_min(rand_cls, 0), K - 1)

        keep = torch.rand_like(alpha) < alpha
        out = torch.where(keep, y0_clip, rand_cls)
        out = torch.where(K <= 1, torch.zeros_like(out), out)
        return out

    def posterior_sample_row(
        self,
        y_t_row: torch.Tensor,
        t_node: torch.Tensor,
        target_t_node: torch.Tensor,
        p0_row: torch.Tensor,
        k_node: torch.Tensor,
        diffusion,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Vectorized row posterior sampler.

        Samples y_{target_t} from q(y_{target_t} | y_t, p_theta(y_0 | y_t))
        under the uniform-mixing categorical kernel.

        Args:
            y_t_row:       (N,) current noisy row label.
            t_node:        (N,) current time per node.
            target_t_node: (N,) target previous time per node.
            p0_row:        (N, Kmax) predicted row distribution for clean assignment.
            k_node:        (N,) active K for each node.
        """
        device = y_t_row.device
        N, Kmax = p0_row.shape
        if N == 0 or Kmax == 0:
            return y_t_row.long()

        K = k_node.long().to(device).clamp_min(1)
        cols = torch.arange(Kmax, device=device).view(1, Kmax)
        valid = cols < K.view(-1, 1)

        p0 = torch.nan_to_num(p0_row.float(), nan=0.0, posinf=0.0, neginf=0.0)
        p0 = p0.masked_fill(~valid, 0.0).clamp_min(0.0)
        p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Direct projection to x0-space.
        direct = (target_t_node.long().to(device) <= 0) | (t_node.long().to(device) <= 0)

        alpha_prev = self._row_alpha_bar_vec(target_t_node, diffusion, device)
        alpha_step = self._row_alpha_step_vec(target_t_node, t_node, diffusion, device)
        invK = (1.0 / K.float()).view(-1, 1)

        # prior_prev[j] = sum_c p0[c] q(y_prev=j | y0=c)
        prior_prev = alpha_prev.view(-1, 1) * p0 + (1.0 - alpha_prev).view(-1, 1) * invK
        prior_prev = prior_prev.masked_fill(~valid, 0.0)

        # likelihood[j] = q(y_t=obs | y_prev=j)
        obs = torch.minimum(torch.clamp_min(y_t_row.long().to(device), 0), K - 1)
        like = ((1.0 - alpha_step).view(-1, 1) * invK).expand(N, Kmax).clone()
        like = like.masked_fill(~valid, 0.0)
        like.scatter_add_(1, obs.view(-1, 1), alpha_step.view(-1, 1))

        post = prior_prev * like
        post = torch.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        post = post.masked_fill(~valid, 0.0)
        post = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # For target_t <= 0, use p0 directly.
        sample_prob = torch.where(direct.view(-1, 1), p0, post)
        sample_prob = sample_prob / sample_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if deterministic:
            out = sample_prob.argmax(dim=-1).long()
        else:
            out = torch.multinomial(sample_prob, 1).squeeze(-1).long()

        out = torch.where(K <= 1, torch.zeros_like(out), out)
        return out




    def q_sample_edge(self, *args, **kwargs):
        raise RuntimeError(
            "HFVRP q_sample_edge() is disabled in strict row-categorical mode. "
            "Use q_sample_row() inside consistency_losses()."
        )

    # ---------------------------------------------------------------------
    # Common row/edge utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _safe_row_prob(prob: torch.Tensor, default: float = 0.5) -> torch.Tensor:
        prob = torch.nan_to_num(prob, nan=default, posinf=default, neginf=default)
        prob = prob.clamp_min(0.0)
        row_sum = prob.sum(dim=-1, keepdim=True)
        bad = (~torch.isfinite(row_sum)) | (row_sum <= 1e-12)
        if bad.any():
            k = prob.size(-1)
            fill = torch.full_like(prob, 1.0 / float(k))
            prob = torch.where(bad.expand_as(prob), fill, prob)
            row_sum = prob.sum(dim=-1, keepdim=True)
        return prob / row_sum.clamp_min(1e-12)

    @staticmethod
    def _group_softmax(logits: torch.Tensor, group: torch.Tensor, eps: float = 1e-12):
        logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
        try:
            from torch_geometric.utils import softmax as pyg_softmax
            out = pyg_softmax(logits, group)
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        g = int(group.max().item()) + 1 if group.numel() > 0 else 0
        if g <= 0:
            return torch.zeros_like(logits)
        max_per = torch.full((g,), -1e30, device=logits.device, dtype=logits.dtype)
        max_per.scatter_reduce_(0, group, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - max_per[group])
        exp = torch.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
        sum_per = torch.zeros((g,), device=logits.device, dtype=logits.dtype)
        sum_per.scatter_add_(0, group, exp)
        out = exp / (sum_per[group] + eps)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_active_edge(self, graph, edge_graph, src, dst, bsz, device):
        """Return active vehicle-customer edges.

        Strict HFVRP solver semantics: all fleet slots provided by the instance are
        valid candidate slots. We do not use used_vehicle_mask or oracle-derived
        vehicle_slot_mask to restrict the model's slot space during train/eval.

        If a future dataset provides a genuine problem-level availability mask
        called vehicle_available_mask, use it; otherwise all graph vehicle slots
        are active.
        """
        del edge_graph, dst, bsz
        if hasattr(graph, "vehicle_available_mask") and graph.vehicle_available_mask is not None:
            mask_v = graph.vehicle_available_mask.view(-1).to(device).bool()
            return mask_v[src.long().to(device)]
        return torch.ones_like(src, dtype=torch.bool, device=device)

    def _batch_structure(self, graph, src: torch.Tensor, dst: torch.Tensor, bsz: int, device):
        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_graph = node_batch[dst]

        veh_cnt = torch.bincount(veh_batch, minlength=max(1, bsz))
        # In aligned HFVRP, graph.K_used denotes the active visible slot count
        # (K_all/Kmax), not the reference number of used routes.
        if hasattr(graph, "K_used") and graph.K_used is not None:
            ku = graph.K_used.view(-1).long().to(device)
            if ku.numel() == 1 and bsz > 1:
                ku = ku.repeat(bsz)
            ku = torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))
        elif hasattr(graph, "max_vehicles") and graph.max_vehicles is not None:
            ku = graph.max_vehicles.view(-1).long().to(device)
            if ku.numel() == 1 and bsz > 1:
                ku = ku.repeat(bsz)
            ku = torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))
        else:
            ku = veh_cnt.clamp_min(1)

        starts = torch.cumsum(veh_cnt, 0) - veh_cnt
        veh_local = src.long() - starts[veh_batch[src.long()]]
        active_edge = self._get_active_edge(graph, edge_graph, src, dst, bsz, device) & (veh_local < ku[edge_graph])
        k_node = ku[node_batch]
        return node_batch, veh_batch, edge_graph, veh_local, active_edge, ku, k_node

    def _edge_onehot_from_row(self, y_row: torch.Tensor, veh_local: torch.Tensor, dst: torch.Tensor, active_edge: torch.Tensor) -> torch.Tensor:
        return ((veh_local.long() == y_row.long()[dst]).float() * active_edge.float())

    def _row_scores_from_edge_scores(
            self,
            graph,
            edge_scores: torch.Tensor,
            active_edge: torch.Tensor,
            src: torch.Tensor,
            dst: torch.Tensor,
            veh_local: torch.Tensor,
            k_node: torch.Tensor,
            bsz: int,
            fill: float = -10000.0,
    ):
        """Convert edge-form scores to dense row-wise scores.

        Returns:
            S:     [num_nodes, Kmax]
            valid: [num_nodes, Kmax]

        HFVRP strict semantics:
            Kmax is the full visible fleet-slot count per graph, not GT-used routes.
        """
        del src

        device = edge_scores.device
        num_nodes = int(graph.node_features.size(0))
        bsz = int(bsz)

        if num_nodes <= 0 or bsz <= 0:
            return (
                edge_scores.new_empty((num_nodes, 0)),
                edge_scores.new_empty((num_nodes, 0), dtype=torch.bool),
            )

        # Prefer shape-derived Kmax under strict full-fleet batching.
        # This avoids k_node.max().item() GPU synchronization in the normal path.
        total_slots = int(graph.veh_features.size(0))
        if total_slots > 0 and total_slots % bsz == 0:
            Kmax = total_slots // bsz
        else:
            # Conservative fallback for irregular batches.
            Kmax = int(k_node.max().detach().item()) if k_node.numel() > 0 else 0

        if Kmax <= 0:
            return (
                edge_scores.new_empty((num_nodes, 0)),
                edge_scores.new_empty((num_nodes, 0), dtype=torch.bool),
            )

        S = edge_scores.new_full((num_nodes, Kmax), float(fill))

        emask = active_edge.bool() & (veh_local >= 0) & (veh_local < Kmax)

        # Empty advanced indexing is safe in PyTorch.
        # Do not guard it with bool(emask.any()), because that synchronizes GPU -> CPU.
        S[dst[emask].long(), veh_local[emask].long()] = edge_scores[emask].float()

        cols = torch.arange(Kmax, device=device).view(1, Kmax)
        valid = cols < k_node.long().view(-1, 1).clamp_min(1)

        S = S.masked_fill(~valid, float(fill))
        return S, valid

    def _row_prob_from_logits(
        self,
        graph,
        logits: torch.Tensor,
        active_edge: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        veh_local: torch.Tensor,
        k_node: torch.Tensor,
        bsz: int,
    ):
        S, valid = self._row_scores_from_edge_scores(
            graph, logits.float(), active_edge, src, dst, veh_local, k_node, bsz, fill=-10000.0
        )
        if S.numel() == 0:
            return S, S, valid
        prob = torch.softmax(S, dim=-1).masked_fill(~valid, 0.0)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return S, prob, valid

    def _edge_prob_from_logits(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, eps: float = 1e-12):
        device = logits.device
        edge_index = graph.edge_index.long().to(device)
        dst = edge_index[1]
        logits_m = torch.nan_to_num(logits.float(), nan=-1e30, posinf=1e30, neginf=-1e30)
        logits_m = logits_m.masked_fill(~active_edge, -1e30)
        p = self._group_softmax(logits_m, dst, eps=eps)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return p * active_edge.float()

    def _edge_prob_from_row_prob(
            self,
            row_prob: torch.Tensor,
            dst: torch.Tensor,
            veh_local: torch.Tensor,
            active_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Gather row-wise probabilities back to edge form.

        row_prob:
            [num_nodes, Kmax]

        Returns:
            edge probability aligned with graph.edge_index.
        """
        if row_prob.numel() == 0:
            return active_edge.float() * 0.0

        Kmax = int(row_prob.size(1))
        out = row_prob.new_zeros((dst.numel(),))

        emask = active_edge.bool() & (veh_local >= 0) & (veh_local < Kmax)

        # Empty advanced indexing is valid; no bool(emask.any()) needed.
        out[emask] = row_prob[dst[emask].long(), veh_local[emask].long()]

        return out * active_edge.float()

    def _row_from_edge_state(
            self,
            graph,
            xt_edge: torch.Tensor,
            active_edge: torch.Tensor,
            src: torch.Tensor,
            dst: torch.Tensor,
            veh_local: torch.Tensor,
            k_node: torch.Tensor,
            bsz: int,
    ) -> torch.Tensor:
        """Recover row labels from edge-form [0, 1] state by row-wise argmax."""
        x = torch.nan_to_num(
            xt_edge.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        S, valid = self._row_scores_from_edge_scores(
            graph, x, active_edge, src, dst, veh_local, k_node, bsz, fill=-1.0
        )
        if S.numel() == 0:
            return torch.zeros(
                (int(graph.node_features.size(0)),),
                device=xt_edge.device,
                dtype=torch.long,
            )

        S = S.masked_fill(~valid, -1.0)
        return S.argmax(dim=-1).long()

    def _sample_row_from_prob(self, row_prob: torch.Tensor, k_node: torch.Tensor, deterministic: bool = False):
        if row_prob.numel() == 0:
            return torch.empty((row_prob.size(0),), device=row_prob.device, dtype=torch.long)
        Kmax = row_prob.size(1)
        cols = torch.arange(Kmax, device=row_prob.device).view(1, Kmax)
        valid = cols < k_node.long().view(-1, 1).clamp_min(1)
        prob = torch.nan_to_num(row_prob.float(), nan=0.0, posinf=0.0, neginf=0.0)
        prob = prob.masked_fill(~valid, 0.0).clamp_min(0.0)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if deterministic:
            return prob.argmax(dim=-1).long()
        return torch.multinomial(prob, 1).squeeze(-1).long()

    # ---------------------------------------------------------------------
    # Guidance and regularizers
    # ---------------------------------------------------------------------

    def _guidance_scale_at_step(self, step_idx: int, total_steps: int) -> float:
        base = float(getattr(self.args, "guide_scale", 1.0))
        if base <= 0:
            return 0.0
        skip_first = bool(getattr(self.args, "cm_guide_skip_first", True))
        mode = str(getattr(self.args, "cm_guide_schedule", "late")).lower().strip()
        if total_steps <= 1:
            return base
        if skip_first and step_idx == 0:
            return 0.0
        if mode == "constant":
            return base
        if mode == "linear":
            frac = float(step_idx + 1) / float(total_steps)
            return base * frac
        if mode == "late":
            frac = float(step_idx) / float(max(1, total_steps - 1))
            return base * frac
        return base

    def guided_logits_step(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, x0_ref_edge: torch.Tensor = None, guide_scale: float = None):
        device = logits.device
        c_sim = float(getattr(self.args, "c_sim", getattr(self.args, "c1", 1.0)))
        c_cap = float(getattr(self.args, "c_cap", getattr(self.args, "c2", 1.0)))
        c_compact = float(getattr(self.args, "c_compact", 0.0))
        guide_scale = float(self._guidance_scale_at_step(1, 1) if guide_scale is None else guide_scale)
        if guide_scale <= 0 or (c_sim <= 0 and c_cap <= 0 and c_compact <= 0):
            p1 = self._edge_prob_from_logits(graph, logits, active_edge)
            zero = logits.sum() * 0.0
            return logits, p1, {"sim": zero.detach(), "cap": zero.detach(), "compact": zero.detach(), "total": zero.detach()}

        with torch.enable_grad():
            logits_g = logits.detach().float().clone().requires_grad_(True)

            sim_loss = logits_g.sum() * 0.0
            if x0_ref_edge is not None and c_sim > 0 and bool(active_edge.any()):
                sim_loss = F.binary_cross_entropy_with_logits(
                    logits_g[active_edge],
                    x0_ref_edge.float().to(device)[active_edge],
                )

            cap_loss = logits_g.sum() * 0.0
            compact_loss = logits_g.sum() * 0.0
            if c_cap > 0:
                cap_loss = self._capacity_proxy_loss(graph, logits_g, active_edge)
            if c_compact > 0:
                compact_loss = self._compactness_proxy_loss(graph, logits_g, active_edge)

            total_loss = c_sim * sim_loss + c_cap * cap_loss + c_compact * compact_loss
            grad = torch.autograd.grad(
                total_loss,
                logits_g,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        denom = grad[active_edge].norm().clamp_min(1e-12) if bool(active_edge.any()) else grad.norm().clamp_min(1e-12)
        grad = grad / denom

        with torch.no_grad():
            guided_logits = logits.detach().float() - guide_scale * grad
            guided_logits = torch.nan_to_num(guided_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            p1_guided = self._edge_prob_from_logits(graph, guided_logits, active_edge)

        return guided_logits, p1_guided, {
            "sim": sim_loss.detach(),
            "cap": cap_loss.detach(),
            "compact": compact_loss.detach(),
            "total": total_loss.detach(),
        }

    def _edge_logits_to_input(self, xt_prob_or_hard: torch.Tensor) -> torch.Tensor:
        """Convert edge-form row-categorical state to GNN input.

        Current convention:
            GNN expects xt_edge in [0, 1].
        """
        if xt_prob_or_hard.ndim == 2:
            xt = xt_prob_or_hard[:, 1].float()
        else:
            xt = xt_prob_or_hard.float()

        xt = torch.nan_to_num(
            xt,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        jitter = float(getattr(self.args, "xt_jitter", 0.0))
        if jitter > 0:
            mult = 1.0 + jitter * torch.randn_like(xt)
            mult = mult.clamp(0.9, 1.1)
            xt = (xt * mult).clamp(0.0, 1.0)

        return xt

    def _get_batched_slot_group(self, graph, B: int, Kmax: int, device):
        """Return slot_group with shape [B, Kmax].

        slot_group[k] denotes the vehicle type/tier of slot k.
        Under PyG batching, per-instance slot_group is usually concatenated as [B*K].
        """
        if hasattr(graph, "slot_group") and graph.slot_group is not None:
            sg = graph.slot_group.view(-1).long().to(device)
        elif hasattr(graph, "vehicle_tier") and graph.vehicle_tier is not None:
            sg = graph.vehicle_tier.view(-1).long().to(device)
        else:
            raise RuntimeError(
                "Type-aware HFVRP loss requires graph.slot_group or graph.vehicle_tier."
            )

        if sg.numel() == B * Kmax:
            return sg.view(B, Kmax)

        if sg.numel() >= Kmax:
            return sg[:Kmax].view(1, Kmax).expand(B, Kmax)

        raise RuntimeError(
            f"slot_group size mismatch: got {sg.numel()}, "
            f"expected Kmax={Kmax} or B*Kmax={B * Kmax}."
        )

    def _type_ce_and_acc_from_row_prob(
            self,
            graph,
            row_prob: torch.Tensor,
            k_node: torch.Tensor,
            bsz: int,
            device,
    ):
        """Type-level assignment CE.

        P_type[i, g] = sum_{k: slot_group[k] == g} P[i, k].

        This supervises which vehicle type each customer should use.
        """
        if row_prob.numel() == 0:
            zero = row_prob.sum() * 0.0
            return zero, zero

        B = int(bsz)
        total_nodes = int(row_prob.size(0))
        Kmax = int(row_prob.size(-1))

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0 or Kmax <= 0:
            zero = row_prob.sum() * 0.0
            return zero, zero

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

        # Remap group ids to 0..G-1 in case tier ids are not contiguous.
        uniq = torch.unique(slot_group, sorted=True)
        G = int(uniq.numel())
        if G <= 0:
            zero = row_prob.sum() * 0.0
            return zero, zero

        slot_group_remap = torch.searchsorted(uniq, slot_group)

        if hasattr(graph, "y_group") and graph.y_group is not None:
            y_group = graph.y_group.view(-1).long().to(device).view(B, N)
            y_group = torch.searchsorted(uniq, y_group)
        else:
            y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)
            batch_arange = torch.arange(B, device=device).view(B, 1)
            y_group = slot_group_remap[batch_arange, y_slot]

        P_type = P.new_zeros((B, N, G))
        group_idx = slot_group_remap.view(B, 1, Kmax).expand(B, N, Kmax)
        P_type.scatter_add_(2, group_idx, P)
        P_type = P_type / P_type.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        logp = torch.log(P_type.clamp_min(1e-12)).view(B * N, G)
        target = y_group.reshape(B * N).long()

        loss = F.nll_loss(logp, target)
        acc = (P_type.argmax(dim=-1) == y_group).float().mean()

        return loss, acc

    def _sampled_pairwise_partition_loss_within_type(
            self,
            graph,
            row_prob: torch.Tensor,
            k_node: torch.Tensor,
            bsz: int,
            device,
    ):
        """Type-aware sampled pairwise partition loss for HFVRP.

        Only same-type customer pairs are sampled.

        Positive pair:
            same vehicle type and same slot.

        Negative pair:
            same vehicle type but different slot.

        same_prob is computed only inside the corresponding vehicle-type slot subset.
        """
        if row_prob.numel() == 0 or (not hasattr(graph, "y")) or graph.y is None:
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

        B = int(bsz)
        total_nodes = int(row_prob.size(0))
        Kmax = int(row_prob.size(-1))

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0 or Kmax <= 0:
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

        N = total_nodes // B
        if N < 2:
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

        pos_samples = int(getattr(self.args, "hf_pair_pos_samples", getattr(self.args, "pair_pos_samples", 128)))
        neg_samples = int(getattr(self.args, "hf_pair_neg_samples", getattr(self.args, "pair_neg_samples", 128)))
        pos_samples = max(0, pos_samples)
        neg_samples = max(0, neg_samples)
        eps = float(getattr(self.args, "pair_eps", 1e-6))

        if pos_samples <= 0 and neg_samples <= 0:
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

        P = torch.nan_to_num(
            row_prob[:, :Kmax].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P.view(B, N, Kmax)

        y_slot = graph.y.view(-1).long().to(device).clamp(0, Kmax - 1).view(B, N)

        slot_group = self._get_batched_slot_group(graph, B, Kmax, device)
        uniq = torch.unique(slot_group, sorted=True)
        slot_group = torch.searchsorted(uniq, slot_group)

        if hasattr(graph, "y_group") and graph.y_group is not None:
            y_group = graph.y_group.view(-1).long().to(device).view(B, N)
            y_group = torch.searchsorted(uniq, y_group)
        else:
            batch_arange_full = torch.arange(B, device=device).view(B, 1)
            y_group = slot_group[batch_arange_full, y_slot]

        pair_i_all, pair_j_all = torch.triu_indices(N, N, offset=1, device=device)

        same_group = y_group[:, pair_i_all].eq(y_group[:, pair_j_all])
        same_slot = y_slot[:, pair_i_all].eq(y_slot[:, pair_j_all])

        pos_w = (same_group & same_slot).float()
        neg_w = (same_group & (~same_slot)).float()

        valid = torch.ones((B,), dtype=torch.bool, device=device)
        if pos_samples > 0:
            valid = valid & (pos_w.sum(dim=-1) > 0)
        if neg_samples > 0:
            valid = valid & (neg_w.sum(dim=-1) > 0)

        if not bool(valid.any()):
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

        P = P[valid]
        y_group = y_group[valid]
        slot_group = slot_group[valid]
        pos_w = pos_w[valid]
        neg_w = neg_w[valid]

        Bv = int(P.size(0))
        batch_arange = torch.arange(Bv, device=device).view(Bv, 1)

        def _same_prob_for_selected(sel: torch.Tensor) -> torch.Tensor:
            ii = pair_i_all[sel]
            jj = pair_j_all[sel]

            Pi = P[batch_arange, ii]
            Pj = P[batch_arange, jj]

            gij = y_group[batch_arange, ii]
            mask_k = slot_group.unsqueeze(1).eq(gij.unsqueeze(-1))

            Pi_g = Pi.masked_fill(~mask_k, 0.0)
            Pj_g = Pj.masked_fill(~mask_k, 0.0)

            Pi_g = Pi_g / Pi_g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            Pj_g = Pj_g / Pj_g.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            return (Pi_g * Pj_g).sum(dim=-1)

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

        loss_mat = -(
                target * torch.log(same_prob)
                + (1.0 - target) * torch.log1p(-same_prob)
        )
        loss = loss_mat.mean()

        with torch.no_grad():
            pred = same_prob >= 0.5
            acc = (pred == target.bool()).float().mean()

            pos_mask = target > 0.5
            neg_mask = ~pos_mask

            pos_score = same_prob[pos_mask].mean() if bool(pos_mask.any()) else row_prob.sum().detach() * 0.0
            neg_score = same_prob[neg_mask].mean() if bool(neg_mask.any()) else row_prob.sum().detach() * 0.0

            pair_count_t = torch.tensor(
                float(Bv * same_prob.size(1)),
                device=device,
                dtype=row_prob.dtype,
            )

        return loss, {
            "loss": loss.detach(),
            "acc": acc.detach(),
            "pos_score": pos_score.detach(),
            "neg_score": neg_score.detach(),
            "num_pairs": pair_count_t.detach(),
        }


    def _capacity_proxy_loss(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        """Optional guidance-only capacity proxy with per-slot HF capacities."""
        device = logits.device
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]

        masked_logits = logits.float().clone()
        masked_logits[~active_edge] = -1e30
        p_assign = self._group_softmax(masked_logits, dst)

        demands = getattr(graph, "demand_linehaul", None)
        if demands is None:
            demand_col = int(getattr(self.args, "demand_col", 2))
            demands = graph.node_features[:, demand_col]
        demands = demands.float().to(device)
        dem_e = demands[dst]

        load = torch.zeros((graph.veh_features.size(0),), device=device, dtype=torch.float32)
        load.index_add_(0, src, p_assign * dem_e)

        if hasattr(graph, "vehicle_capacity") and graph.vehicle_capacity is not None:
            cap_v = graph.vehicle_capacity.float().to(device).view(-1).clamp_min(1e-9)
        elif hasattr(graph, "veh_features") and graph.veh_features.size(1) >= 1:
            cap_v = graph.veh_features[:, 0].float().to(device).clamp_min(1e-9)
        else:
            cap_v = torch.ones_like(load)

        over = F.relu(load - cap_v)
        return ((over / cap_v.clamp_min(1e-9)) ** 2).mean()

    def _compactness_proxy_loss(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        device = logits.device
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        masked_logits = logits.float().clone()
        masked_logits[~active_edge] = -1e30
        p_assign = self._group_softmax(masked_logits, dst)

        xy = graph.node_features[:, :2].float().to(device)
        xyd = xy[dst]

        denom = torch.zeros((graph.veh_features.size(0),), device=device, dtype=torch.float32)
        denom.index_add_(0, src, p_assign)
        denom = denom.clamp_min(1e-12)

        mu = torch.zeros((graph.veh_features.size(0), 2), device=device, dtype=torch.float32)
        mu.index_add_(0, src, p_assign.unsqueeze(-1) * xyd)
        mu = mu / denom.unsqueeze(-1)

        diff = xyd - mu[src]
        return (p_assign * (diff * diff).sum(dim=-1)).mean()

    def regularizer_losses(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, lam_cap: float = 0.0, lam_compact: float = 0.0):
        zero = logits.sum() * 0.0
        cap_loss = self._capacity_proxy_loss(graph, logits, active_edge) if lam_cap > 0 else zero
        compact_loss = self._compactness_proxy_loss(graph, logits, active_edge) if lam_compact > 0 else zero
        return cap_loss, compact_loss


    # ---------------------------------------------------------------------
    # Row-categorical CM sampler
    # ---------------------------------------------------------------------
    def cm_project_resample_step(
        self,
        model,
        graph,
        xt_edge: torch.Tensor,
        t_graph: torch.Tensor,
        t_next: int,
        active_edge: torch.Tensor,
        step_idx: int,
        total_steps: int,
    ):
        """One row-categorical CM inference step.

        Input/output remains edge-form for compatibility with the existing caller.
        Internally, the current edge state is projected to a row label, the model
        predicts p_theta(y0 | y_t, t), and re-sampling to t_next is performed with
        a row-categorical posterior.
        """
        device = xt_edge.device
        deterministic = bool(getattr(self.args, "eval_deterministic", False)) or bool(getattr(self.args, "guided_deterministic", False))

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        bsz = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 0
        if bsz <= 0:
            zero = xt_edge.sum() * 0.0
            return xt_edge, xt_edge, {"logits": zero.detach(), "x0_hat": xt_edge.detach()}

        _, _, edge_graph, veh_local, active_calc, ku, k_node = self._batch_structure(graph, src, dst, bsz, device)
        active_edge = active_edge.to(device).bool() & active_calc.bool()

        xt_in = self._edge_logits_to_input(xt_edge)
        with torch.cuda.amp.autocast(enabled=False):
            logits = model.forward(graph, xt_in.float(), t_graph).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # Row-wise x0 distribution from current projection.
        _, p0_row_unguided, _ = self._row_prob_from_logits(graph, logits, active_edge, src, dst, veh_local, k_node, bsz)
        p1_unguided = self._edge_prob_from_row_prob(p0_row_unguided, dst, veh_local, active_edge)

        guidance_enabled = bool(getattr(self.args, "guided", False))
        guide_scale = self._guidance_scale_at_step(step_idx, total_steps) if guidance_enabled else 0.0
        if guide_scale > 0:
            x0_ref_row = self._sample_row_from_prob(p0_row_unguided, k_node, deterministic=deterministic)
            x0_ref_edge = self._edge_onehot_from_row(x0_ref_row, veh_local, dst, active_edge)
            logits_used, _, guide_losses = self.guided_logits_step(
                graph,
                logits,
                active_edge,
                x0_ref_edge=x0_ref_edge,
                guide_scale=guide_scale,
            )
            _, p0_row, _ = self._row_prob_from_logits(graph, logits_used, active_edge, src, dst, veh_local, k_node, bsz)
        else:
            logits_used = logits
            p0_row = p0_row_unguided
            z = logits.sum() * 0.0
            guide_losses = {"sim": z.detach(), "cap": z.detach(), "compact": z.detach(), "total": z.detach()}

        p1 = self._edge_prob_from_row_prob(p0_row, dst, veh_local, active_edge)
        x0_hat_row = self._sample_row_from_prob(p0_row, k_node, deterministic=deterministic)
        x0_hat_edge = self._edge_onehot_from_row(x0_hat_row, veh_local, dst, active_edge)

        if int(t_next) <= 0:
            return x0_hat_edge, p1, {
                "logits": logits_used.detach(),
                "p0_row": p0_row.detach(),
                "x0_hat": x0_hat_edge.detach(),
                "x0_hat_row": x0_hat_row.detach(),
                "guide_scale": float(guide_scale),
                "guide_losses": guide_losses,
            }

        # Current observed row state from edge-form xt, then row posterior to t_next.
        y_t_row = self._row_from_edge_state(graph, xt_edge, active_edge, src, dst, veh_local, k_node, bsz)
        t_node = t_graph.long().to(device)[node_batch]
        t_next_node = torch.full_like(t_node, int(t_next))

        y_next = self.posterior_sample_row(
            y_t_row=y_t_row,
            t_node=t_node,
            target_t_node=t_next_node,
            p0_row=p0_row,
            k_node=k_node,
            diffusion=model.diffusion,
            deterministic=deterministic,
        )
        xt_next = self._edge_onehot_from_row(y_next, veh_local, dst, active_edge)

        return xt_next, p1, {
            "logits": logits_used.detach(),
            "p0_row": p0_row.detach(),
            "x0_hat": x0_hat_edge.detach(),
            "x0_hat_row": x0_hat_row.detach(),
            "y_t_row": y_t_row.detach(),
            "y_next_row": y_next.detach(),
            "guide_scale": float(guide_scale),
            "guide_losses": guide_losses,
        }

    # ---------------------------------------------------------------------
    # Row-wise training objective
    # ---------------------------------------------------------------------
    def _row_ce_and_acc_from_logits(
        self,
        graph,
        logits: torch.Tensor,
        active_edge: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_graph: torch.Tensor,
        bsz: int,
        device,
    ):
        del edge_graph
        if (not hasattr(graph, "y")) or graph.y is None:
            raise RuntimeError("Row CE needs graph.y canonical slot labels.")

        _, _, _, veh_local, active_calc, ku, k_node = self._batch_structure(graph, src, dst, bsz, device)
        active_edge = active_edge.to(device).bool() & active_calc.bool()

        S, valid = self._row_scores_from_edge_scores(
            graph, logits.float(), active_edge, src, dst, veh_local, k_node, bsz, fill=-10000.0
        )
        if S.numel() == 0:
            zero = logits.sum() * 0.0
            return zero, zero

        y = graph.y.view(-1).long().to(device)
        y = torch.minimum(torch.clamp_min(y, 0), k_node - 1)

        S = S.masked_fill(~valid, -10000.0)
        ce = F.cross_entropy(S, y)
        acc = (S.argmax(dim=1) == y).float().mean()
        return ce, acc

    def _row_consistency_kl(
        self,
        graph,
        logits_t: torch.Tensor,
        logits_t2: torch.Tensor,
        active_edge: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        veh_local: torch.Tensor,
        k_node: torch.Tensor,
        bsz: int,
    ):
        S_t, valid = self._row_scores_from_edge_scores(
            graph, logits_t.float(), active_edge, src, dst, veh_local, k_node, bsz, fill=-10000.0
        )
        S_t2, _ = self._row_scores_from_edge_scores(
            graph, logits_t2.float(), active_edge, src, dst, veh_local, k_node, bsz, fill=-10000.0
        )
        if S_t.numel() == 0:
            return logits_t.sum() * 0.0

        logp_t = F.log_softmax(S_t.masked_fill(~valid, -10000.0), dim=-1)
        p_t2 = torch.softmax(S_t2.masked_fill(~valid, -10000.0), dim=-1).detach()
        kl = F.kl_div(logp_t, p_t2, reduction="none").sum(dim=-1)
        return kl.mean()

    def consistency_losses(self, model, batch):
        device = model.device
        graph = batch.to(device)

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]

        # Prefer PyG batch metadata to avoid a GPU -> CPU max().item() sync.
        bsz = int(getattr(graph, "num_graphs", 0) or 0)
        if bsz <= 0:
            bsz = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0

        if bsz <= 0:
            return graph.node_features.sum() * 0.0

        _, _, edge_graph, veh_local, active_edge, ku, k_node = self._batch_structure(
            graph, src, dst, bsz, device
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

        x_t_edge = self._edge_onehot_from_row(y_t, veh_local, dst, active_edge)
        x_t2_edge = self._edge_onehot_from_row(y_t2, veh_local, dst, active_edge)

        # GNN convention: xt input is [0, 1], not [-1, 1].
        xt_in = self._edge_logits_to_input(x_t_edge)
        xt2_in = self._edge_logits_to_input(x_t2_edge)

        with torch.cuda.amp.autocast(enabled=False):
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

        # ------------------------------------------------------------------
        # Row CE remains as a weak slot anchor.
        # ------------------------------------------------------------------
        row_ce_t, row_acc_t = self._row_ce_and_acc_from_logits(
            graph, logits_t, active_edge, src, dst, edge_graph, bsz, device
        )
        row_ce_t2, row_acc_t2 = self._row_ce_and_acc_from_logits(
            graph, logits_t2, active_edge, src, dst, edge_graph, bsz, device
        )

        row_ce = 0.5 * (row_ce_t + row_ce_t2)
        row_acc = 0.5 * (row_acc_t + row_acc_t2)

        # ------------------------------------------------------------------
        # Convert edge logits to row probabilities for type-aware losses.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Type-level CE: vehicle type is not exchangeable.
        # ------------------------------------------------------------------
        type_loss_t, type_acc_t = self._type_ce_and_acc_from_row_prob(
            graph,
            row_prob_t,
            k_node,
            bsz,
            device,
        )
        type_loss_t2, type_acc_t2 = self._type_ce_and_acc_from_row_prob(
            graph,
            row_prob_t2,
            k_node,
            bsz,
            device,
        )

        type_loss = 0.5 * (type_loss_t + type_loss_t2)
        type_acc = 0.5 * (type_acc_t + type_acc_t2)

        # ------------------------------------------------------------------
        # Within-type pairwise partition loss:
        # same/different route is supervised only inside the same vehicle type.
        # ------------------------------------------------------------------
        pair_loss_t, pair_logs_t = self._sampled_pairwise_partition_loss_within_type(
            graph,
            row_prob_t,
            k_node,
            bsz,
            device,
        )
        pair_loss_t2, pair_logs_t2 = self._sampled_pairwise_partition_loss_within_type(
            graph,
            row_prob_t2,
            k_node,
            bsz,
            device,
        )

        pair_loss = 0.5 * (pair_loss_t + pair_loss_t2)
        pair_acc = 0.5 * (pair_logs_t["acc"] + pair_logs_t2["acc"])
        pair_pos = 0.5 * (pair_logs_t["pos_score"] + pair_logs_t2["pos_score"])
        pair_neg = 0.5 * (pair_logs_t["neg_score"] + pair_logs_t2["neg_score"])
        pair_num = 0.5 * (pair_logs_t["num_pairs"] + pair_logs_t2["num_pairs"])

        lam_type = float(getattr(model.args, "hf_lam_type", 1.0))
        lam_pair = float(getattr(model.args, "hf_lam_pair", 1.0))
        lam_row = float(getattr(model.args, "hf_lam_row", 0.10))
        lam_cons = float(
            getattr(
                model.args,
                "hf_lam_cons",
                getattr(model.args, "lam_cons", getattr(model.args, "cm_lam_cons", 0.0)),
            )
        )

        cons_kl = (
            self._row_consistency_kl(
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
            if lam_cons > 0
            else logits_t.sum() * 0.0
        )

        loss = (
                lam_type * type_loss
                + lam_pair * pair_loss
                + lam_row * row_ce
                + lam_cons * cons_kl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "type-aware HFVRP loss became non-finite: "
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
                    "train/hf_type_acc": type_acc.detach(),

                    "train/hf_pair_loss": pair_loss.detach(),
                    "train/hf_pair_acc": pair_acc.detach(),
                    "train/hf_pair_pos": pair_pos.detach(),
                    "train/hf_pair_neg": pair_neg.detach(),

                    "train/hf_row_ce": row_ce.detach(),
                    "train/hf_row_acc": row_acc.detach(),

                    "train/hf_cons_kl": cons_kl.detach(),
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        return loss
