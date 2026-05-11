import torch
import torch.nn.functional as F
from diffusion.consistency.meta import MetaConsistency


class CVRPConsistency(MetaConsistency):
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
    # Legacy binary edge corruption kept only for compatibility with old calls.
    # The aligned training/sampling path below does not use it.
    # ---------------------------------------------------------------------
    def q_sample_edge(self, x0_edge: torch.Tensor, t_edge: torch.Tensor, diffusion, active_edge: torch.Tensor = None) -> torch.Tensor:
        device = x0_edge.device
        self._ensure_diffusion_mats(device, diffusion)

        T = int(diffusion.T)
        t_edge = t_edge.long().clamp(0, T)
        x0 = x0_edge.long().clamp(0, 1)
        x0_oh = F.one_hot(x0, num_classes=2).float()

        Qb = self._Q_bar_torch.index_select(0, t_edge)
        p = torch.bmm(x0_oh.unsqueeze(1), Qb).squeeze(1)
        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        xt = torch.bernoulli(p[:, 1].clamp(0.0, 1.0)).float()
        if active_edge is not None:
            xt = xt * active_edge.float()
        return xt

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
            p0_row: torch.Tensor = None,
            k_node: torch.Tensor = None,
            diffusion=None,
            deterministic: bool = False,
            *,
            # Backward-compatible dense batched interface used by pl_cvrp_model.py:
            p0_bnK: torch.Tensor = None,
            node_batch: torch.Tensor = None,
            node_start: torch.Tensor = None,
    ) -> torch.Tensor:
        """Vectorized row posterior sampler.

        Supports two interfaces:

        New interface:
            p0_row: (sumN, Kmax)

        Old/eval interface:
            p0_bnK:    (B, Nmax, Kmax)
            node_batch: (sumN,)
            node_start: (B,)

        Samples y_{target_t} from q(y_{target_t} | y_t, p_theta(y_0 | y_t))
        under the uniform-mixing categorical kernel.
        """
        device = y_t_row.device

        if p0_row is None:
            if p0_bnK is None or node_batch is None or node_start is None:
                raise ValueError(
                    "posterior_sample_row needs either p0_row or "
                    "(p0_bnK, node_batch, node_start)."
                )
            node_batch = node_batch.long().to(device)
            node_start = node_start.long().to(device)
            local_idx = torch.arange(node_batch.numel(), device=device) - node_start[node_batch]
            p0_row = p0_bnK.to(device)[node_batch, local_idx.long(), :]

        if k_node is None:
            raise ValueError("posterior_sample_row requires k_node.")

        p0_row = p0_row.to(device)
        N, Kmax = p0_row.shape
        if N == 0 or Kmax == 0:
            return y_t_row.long()

        K = k_node.long().to(device).clamp_min(1)
        cols = torch.arange(Kmax, device=device).view(1, Kmax)
        valid = cols < K.view(-1, 1)

        p0 = torch.nan_to_num(p0_row.float(), nan=0.0, posinf=0.0, neginf=0.0)
        p0 = p0.masked_fill(~valid, 0.0).clamp_min(0.0)
        p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(1e-12)

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

        sample_prob = torch.where(direct.view(-1, 1), p0, post)
        sample_prob = sample_prob / sample_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if deterministic:
            out = sample_prob.argmax(dim=-1).long()
        else:
            out = torch.multinomial(sample_prob, 1).squeeze(-1).long()

        out = torch.where(K <= 1, torch.zeros_like(out), out)
        return out

    # ---------------------------------------------------------------------
    # Common row/edge utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _safe_prob01(p: torch.Tensor, default: float = 0.5) -> torch.Tensor:
        p = torch.nan_to_num(p, nan=default, posinf=default, neginf=default)
        return p.clamp(0.0, 1.0)

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
        """
        Strict Kmax mode:
          active vehicle-customer edges are determined only by the physical vehicle
          slots present in the graph, normally K_max.

        graph.K_used / graph.K_ref_used are deliberately ignored here.
        """
        del dst  # kept for call-site compatibility

        veh_batch = graph.veh_batch.long().to(device)
        edge_graph = edge_graph.long().to(device)

        veh_cnt = torch.bincount(veh_batch, minlength=max(1, int(bsz))).long()

        if hasattr(graph, "K_max") and graph.K_max is not None:
            ku = graph.K_max.view(-1).long().to(device)
            if ku.numel() == 1 and int(bsz) > 1:
                ku = ku.repeat(int(bsz))
            ku = torch.minimum(ku[:int(bsz)].clamp_min(1), veh_cnt.clamp_min(1))
        else:
            ku = veh_cnt.clamp_min(1)

        starts = torch.cumsum(veh_cnt, 0) - veh_cnt
        veh_local = src.long().to(device) - starts[veh_batch[src.long().to(device)]]

        active_edge = veh_local < ku[edge_graph]
        return active_edge.bool()

    def _batch_structure(self, graph, src: torch.Tensor, dst: torch.Tensor, bsz: int, device):
        """
        Return batch structure under strict Kmax semantics.

        ku / k_node are the full physical candidate slot counts, not oracle used-route
        counts. This makes row diffusion, row CE, and active_edge all operate over
        the same Kmax slot space.
        """
        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_graph = node_batch[dst]

        veh_cnt = torch.bincount(veh_batch, minlength=max(1, bsz)).long()

        if hasattr(graph, "K_max") and graph.K_max is not None:
            ku = graph.K_max.view(-1).long().to(device)
            if ku.numel() == 1 and bsz > 1:
                ku = ku.repeat(bsz)
            ku = torch.minimum(ku[:bsz].clamp_min(1), veh_cnt.clamp_min(1))
        else:
            ku = veh_cnt.clamp_min(1)

        starts = torch.cumsum(veh_cnt, 0) - veh_cnt
        veh_local = src.long().to(device) - starts[veh_batch[src.long().to(device)]]

        active_edge = veh_local < ku[edge_graph]
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
        del bsz, src

        device = edge_scores.device
        num_nodes = int(graph.node_features.size(0))

        # Shape construction still needs a Python int.
        # Under strict fixed-K CVRP, this is acceptable, but the cleaner long-term
        # version is to pass Kmax from args/config as a Python int.
        Kmax = int(k_node.max().detach().item()) if k_node.numel() > 0 else 0

        if num_nodes <= 0 or Kmax <= 0:
            return (
                edge_scores.new_empty((num_nodes, 0)),
                edge_scores.new_empty((num_nodes, 0), dtype=torch.bool),
            )

        S = edge_scores.new_full((num_nodes, Kmax), float(fill))

        emask = active_edge & (veh_local >= 0) & (veh_local < Kmax)

        # No bool(emask.any()) here.
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
        if row_prob.numel() == 0:
            return active_edge.float() * 0.0

        Kmax = row_prob.size(1)
        out = row_prob.new_zeros((dst.numel(),))

        emask = active_edge & (veh_local >= 0) & (veh_local < Kmax)
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
        """Recover a row label from an edge-form [0, 1] state by row-wise argmax."""
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
            return torch.zeros((int(graph.node_features.size(0)),), device=xt_edge.device, dtype=torch.long)

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
    # Guidance
    # ---------------------------------------------------------------------
    def _sample_group_onehot(self, prob1: torch.Tensor, group: torch.Tensor, active_edge: torch.Tensor, deterministic: bool = False):
        prob1 = self._safe_prob01(prob1, default=0.5) * active_edge.float()
        if prob1.numel() == 0 or group.numel() == 0:
            return torch.zeros_like(prob1)
        n_group = int(group.max().item()) + 1
        if n_group <= 0:
            return torch.zeros_like(prob1)

        scores = torch.full_like(prob1, -1e30)
        if deterministic:
            scores = torch.where(active_edge, prob1, scores)
        else:
            u = torch.rand_like(prob1).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            scores = torch.where(active_edge, torch.log(prob1.clamp_min(1e-12)) + gumbel, scores)

        best = torch.full((n_group,), -1e30, device=prob1.device, dtype=prob1.dtype)
        best.scatter_reduce_(0, group, scores, reduce="amax", include_self=True)
        is_best = active_edge & (scores >= (best[group] - 1e-12))

        order = torch.arange(prob1.numel(), device=prob1.device, dtype=torch.long)
        inf = torch.full((prob1.numel(),), prob1.numel(), device=prob1.device, dtype=torch.long)
        chosen_ord = torch.where(is_best, order, inf)
        first = torch.full((n_group,), prob1.numel(), device=prob1.device, dtype=torch.long)
        first.scatter_reduce_(0, group, chosen_ord, reduce="amin", include_self=True)
        picked = is_best & (order == first[group])
        return picked.float()

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
        """Optional inference-time similarity guidance.

        Capacity and compactness guidance are intentionally removed. The training
        objective now uses only a weak row CE anchor plus sampled pairwise
        partition supervision, so this guidance path is kept sim-only for
        backward compatibility with existing guided inference calls.
        """
        device = logits.device
        c_sim = float(getattr(self.args, "c_sim", getattr(self.args, "c1", 1.0)))
        guide_scale = float(self._guidance_scale_at_step(1, 1) if guide_scale is None else guide_scale)
        if guide_scale <= 0 or c_sim <= 0 or x0_ref_edge is None or not bool(active_edge.any()):
            p1 = self._edge_prob_from_logits(graph, logits, active_edge)
            zero = logits.sum() * 0.0
            return logits, p1, {"sim": zero.detach(), "total": zero.detach()}

        with torch.enable_grad():
            logits_g = logits.detach().float().clone().requires_grad_(True)
            sim_loss = F.binary_cross_entropy_with_logits(
                logits_g[active_edge],
                x0_ref_edge.float().to(device)[active_edge],
            )
            total_loss = c_sim * sim_loss
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
            "total": total_loss.detach(),
        }

    def _edge_logits_to_input(self, xt_prob_or_hard: torch.Tensor) -> torch.Tensor:
        """Convert edge-form diffusion state to GNN input.

        Current project convention:
            GNN expects edge state in [0, 1].

        For row-categorical diffusion, xt is normally a hard one-hot edge state.
        This function intentionally does not map it to [-1, 1].
        """
        xt = torch.nan_to_num(
            xt_prob_or_hard.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        return xt.clamp(0.0, 1.0)

    def denoise_edge(self, model, graph, x_t_edge: torch.Tensor, t_edge: torch.Tensor, x0_onehot: torch.Tensor):
        """Legacy helper kept for old callers; aligned training does not use x0 skip injection."""
        xt_in = self._edge_logits_to_input(x_t_edge)
        with torch.cuda.amp.autocast(enabled=False):
            logit1 = model.forward_edge(graph, xt_in.float(), t_edge).float()
        logit1 = torch.nan_to_num(logit1, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        model_output = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
        c_skip, c_out = [
            self.append_dims(x, model_output.ndim).float()
            for x in self.get_scalings_for_boundary_condition(t_edge)
        ]
        c_skip = torch.nan_to_num(c_skip, nan=0.0, posinf=1.0, neginf=0.0)
        c_out = torch.nan_to_num(c_out, nan=1.0, posinf=1.0, neginf=0.0)
        denoise = c_out * model_output + c_skip * x0_onehot.float()
        denoise = torch.nan_to_num(denoise, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        return model_output, denoise

    def guided_xt_step(self, model, graph, xt_prob: torch.Tensor, x0_ref_edge: torch.Tensor, t_graph: torch.Tensor, active_edge: torch.Tensor):
        del model, xt_prob, t_graph, x0_ref_edge
        device = active_edge.device
        zero = torch.zeros((active_edge.numel(),), device=device, dtype=torch.float32)
        xt_u = zero
        xt_prob_u = torch.stack([1.0 - zero, zero], dim=-1)
        if active_edge.numel() == 0:
            return xt_u, xt_prob_u
        raise RuntimeError(
            "guided_xt_step is deprecated for row-CM inference; use guided_logits_step inside cm_project_resample_step."
        )

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
            guide_losses = {"sim": z.detach(), "total": z.detach()}

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

        if B <= 0 or total_nodes <= 0 or total_nodes % B != 0:
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

        pos_samples = int(getattr(self.args, "pair_pos_samples", getattr(self.args, "cm_pair_pos_samples", 128)))
        neg_samples = int(getattr(self.args, "pair_neg_samples", getattr(self.args, "cm_pair_neg_samples", 128)))
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
            zero = row_prob.sum() * 0.0
            return zero, {
                "loss": zero.detach(),
                "acc": zero.detach(),
                "pos_score": zero.detach(),
                "neg_score": zero.detach(),
                "num_pairs": torch.zeros((), device=device),
            }

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

        with torch.no_grad():
            pred = same_prob >= 0.5
            acc = (pred == target.bool()).float().mean()

            pos_mask = target > 0.5
            neg_mask = ~pos_mask

            if pos_samples > 0 and bool(pos_mask.any()):
                pos_score = same_prob[pos_mask].mean()
            else:
                pos_score = row_prob.sum().detach() * 0.0

            if neg_samples > 0 and bool(neg_mask.any()):
                neg_score = same_prob[neg_mask].mean()
            else:
                neg_score = row_prob.sum().detach() * 0.0

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

        with torch.cuda.amp.autocast(enabled=False):
            logits_t = model.forward(graph, xt_in.float(), t_graph).float()
            logits_t2 = model.forward(graph, xt2_in.float(), t2_graph).float()

        logits_t = torch.nan_to_num(logits_t, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        logits_t2 = torch.nan_to_num(logits_t2, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        row_ce_t, row_acc_t = self._row_ce_and_acc_from_logits(
            graph, logits_t, active_edge, src, dst, edge_graph, bsz, device
        )
        row_ce_t2, row_acc_t2 = self._row_ce_and_acc_from_logits(
            graph, logits_t2, active_edge, src, dst, edge_graph, bsz, device
        )

        _, row_prob_t, _ = self._row_prob_from_logits(
            graph, logits_t, active_edge, src, dst, veh_local, k_node, bsz
        )
        _, row_prob_t2, _ = self._row_prob_from_logits(
            graph, logits_t2, active_edge, src, dst, veh_local, k_node, bsz
        )
        pair_loss_t, pair_stats_t = self._sampled_pairwise_partition_loss(
            graph, row_prob_t, k_node, bsz, device
        )
        pair_loss_t2, pair_stats_t2 = self._sampled_pairwise_partition_loss(
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
                {
                    "train/cm_loss_pair_main": pair_loss.detach(),
                    "train/cm_loss_row_anchor": row_loss.detach(),
                    "train/cm_pair_t": pair_loss_t.detach(),
                    "train/cm_pair_t2": pair_loss_t2.detach(),
                    "train/cm_pair_acc": 0.5 * (pair_stats_t["acc"] + pair_stats_t2["acc"]),
                    "train/cm_pair_pos_score": 0.5 * (pair_stats_t["pos_score"] + pair_stats_t2["pos_score"]),
                    "train/cm_pair_neg_score": 0.5 * (pair_stats_t["neg_score"] + pair_stats_t2["neg_score"]),
                    "train/cm_pair_num_pairs": 0.5 * (pair_stats_t["num_pairs"] + pair_stats_t2["num_pairs"]),
                    "train/cm_row_ce_t": row_ce_t.detach(),
                    "train/cm_row_ce_t2": row_ce_t2.detach(),
                    "train/cm_row_ce": row_loss.detach(),
                    "train/cm_row_acc_t": row_acc_t.detach(),
                    "train/cm_row_acc_t2": row_acc_t2.detach(),
                    "train/cm_row_acc": 0.5 * (row_acc_t.detach() + row_acc_t2.detach()),
                    "train/cm_cons_kl": cons_kl.detach(),
                    "train/cm_lam_pair": torch.tensor(lam_pair, device=device),
                    "train/cm_lam_row": torch.tensor(lam_row, device=device),
                    "train/cm_lam_cons": torch.tensor(lam_cons, device=device),
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # ------------------------------------------------------------
        # Optional auxiliary active-K prediction loss.
        # This is intentionally a side branch: it does not modify K_used,
        # active_edge, row diffusion, or row CE.
        # ------------------------------------------------------------
        if hasattr(model, "compute_k_pred_loss") and bool(getattr(model, "use_k_predictor", False)):
            k_loss, k_logs = model.compute_k_pred_loss(graph)
            k_w = float(getattr(model, "k_pred_loss_weight", 0.05))

            loss = loss + k_w * k_loss

            if hasattr(model, "log_dict") and k_logs:
                log_payload = dict(k_logs)
                log_payload["train/k_loss_weight"] = torch.tensor(k_w, device=device)
                model.log_dict(
                    log_payload,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss
