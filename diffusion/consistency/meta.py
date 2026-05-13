import torch
import torch.nn.functional as F


class MetaConsistency:
    """Minimal base class for Stage-A row-categorical consistency objectives."""

    def __init__(
        self,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.boundary_func = boundary_func
        self._Q_bar_torch = None
        self.args = None

    def _ensure_diffusion_mats(self, device, diffusion):
        if self._Q_bar_torch is None or self._Q_bar_torch.device != device:
            self._Q_bar_torch = torch.as_tensor(
                diffusion.Q_bar,
                device=device,
                dtype=torch.float32,
            )

    @staticmethod
    def _zero(ref: torch.Tensor) -> torch.Tensor:
        return ref.sum() * 0.0

class RowCategoricalConsistencyBase(MetaConsistency):
    """Shared row-categorical consistency utilities for Stage-A VRP assignment.

    Common semantics:
      - diffusion variable: customer-row slot label y_i;
      - GNN input: edge-form one-hot assignment state;
      - GNN output: edge logits interpreted row-wise by softmax;
      - sampler: row-categorical posterior / re-noise.
    """

    # ------------------------------------------------------------
    # Slot-count hook
    # ------------------------------------------------------------
    def _slot_count_per_graph(self, graph, veh_cnt: torch.Tensor, bsz: int, device):
        raise NotImplementedError

    # ------------------------------------------------------------
    # Row-categorical diffusion schedule
    # ------------------------------------------------------------
    def _row_alpha_bar_vec(self, t: torch.Tensor, diffusion, device) -> torch.Tensor:
        self._ensure_diffusion_mats(device, diffusion)

        T = int(diffusion.T)
        t = t.long().to(device).clamp(0, T)

        raw_t = self._Q_bar_torch[t, 1, 1]
        raw_T = self._Q_bar_torch[T, 1, 1]

        alpha = (raw_t - raw_T) / (1.0 - raw_T).clamp_min(1e-12)
        alpha = alpha.clamp(0.0, 1.0)
        alpha = torch.where(t <= 0, torch.ones_like(alpha), alpha)
        return alpha

    def _row_alpha_step_vec(
        self,
        t_prev: torch.Tensor,
        t_cur: torch.Tensor,
        diffusion,
        device,
    ) -> torch.Tensor:
        t_prev = t_prev.long().to(device)
        t_cur = t_cur.long().to(device)

        a_prev = self._row_alpha_bar_vec(t_prev, diffusion, device)
        a_cur = self._row_alpha_bar_vec(t_cur, diffusion, device)

        step = a_cur / a_prev.clamp_min(1e-12)
        step = step.clamp(0.0, 1.0)
        step = torch.where(t_cur <= t_prev, torch.ones_like(step), step)
        step = torch.where(a_prev <= 1e-12, torch.zeros_like(step), step)
        return step

    def q_sample_row(
        self,
        y0_row: torch.Tensor,
        t_node: torch.Tensor,
        k_node: torch.Tensor,
        diffusion,
    ) -> torch.Tensor:
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
        device = y_t_row.device

        p0_row = p0_row.to(device)
        N, Kmax = p0_row.shape

        if N == 0 or Kmax == 0:
            return y_t_row.long()

        K = k_node.long().to(device).clamp_min(1)
        cols = torch.arange(Kmax, device=device).view(1, Kmax)
        valid = cols < K.view(-1, 1)

        p0 = torch.nan_to_num(
            p0_row.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        p0 = p0.masked_fill(~valid, 0.0).clamp_min(0.0)
        p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        direct = (
            (target_t_node.long().to(device) <= 0)
            | (t_node.long().to(device) <= 0)
        )

        alpha_prev = self._row_alpha_bar_vec(target_t_node, diffusion, device)
        alpha_step = self._row_alpha_step_vec(
            target_t_node,
            t_node,
            diffusion,
            device,
        )

        invK = (1.0 / K.float()).view(-1, 1)

        prior_prev = (
            alpha_prev.view(-1, 1) * p0
            + (1.0 - alpha_prev).view(-1, 1) * invK
        )
        prior_prev = prior_prev.masked_fill(~valid, 0.0)

        obs = torch.minimum(torch.clamp_min(y_t_row.long().to(device), 0), K - 1)

        like = ((1.0 - alpha_step).view(-1, 1) * invK).expand(N, Kmax).clone()
        like = like.masked_fill(~valid, 0.0)
        like.scatter_add_(1, obs.view(-1, 1), alpha_step.view(-1, 1))

        post = prior_prev * like
        post = torch.nan_to_num(
            post,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
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

    def _batch_structure(self, graph, src: torch.Tensor, dst: torch.Tensor, bsz: int, device):
        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_graph = node_batch[dst]

        veh_cnt = torch.bincount(veh_batch, minlength=max(1, bsz)).long()

        ku = self._slot_count_per_graph(
            graph=graph,
            veh_cnt=veh_cnt,
            bsz=bsz,
            device=device,
        )

        starts = torch.cumsum(veh_cnt, 0) - veh_cnt
        veh_local = src.long().to(device) - starts[veh_batch[src.long().to(device)]]

        active_edge = veh_local < ku[edge_graph]
        k_node = ku[node_batch]

        return node_batch, veh_batch, edge_graph, veh_local, active_edge, ku, k_node

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
        del step_idx, total_steps

        device = xt_edge.device

        if not torch.is_tensor(t_graph):
            t_graph = torch.tensor(t_graph, device=device)
        t_graph = t_graph.long().to(device).view(-1)
        bsz = int(t_graph.numel())

        if bsz <= 0:
            zero = xt_edge.sum() * 0.0
            return xt_edge, xt_edge, {
                "logits": zero.detach(),
                "x0_hat": xt_edge.detach(),
            }

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]

        _, _, _, veh_local, active_calc, _, k_node = self._batch_structure(
            graph,
            src,
            dst,
            bsz,
            device,
        )
        active_edge = active_edge.to(device).bool() & active_calc.bool()

        xt_in = self._edge_logits_to_input(xt_edge)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits = model.forward(graph, xt_in.float(), t_graph).float()

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        _, p0_row, _ = self._row_prob_from_logits(
            graph,
            logits,
            active_edge,
            src,
            dst,
            veh_local,
            k_node,
            bsz,
        )

        p1 = self._edge_prob_from_row_prob(
            p0_row,
            dst,
            veh_local,
            active_edge,
        )

        deterministic = bool(getattr(self.args, "eval_deterministic", False))
        x0_hat_row = self._sample_row_from_prob(
            p0_row,
            k_node,
            deterministic=deterministic,
        )
        x0_hat_edge = self._edge_onehot_from_row(
            x0_hat_row,
            veh_local,
            dst,
            active_edge,
        )

        if int(t_next) <= 0:
            return x0_hat_edge, p1, {
                "logits": logits.detach(),
                "p0_row": p0_row.detach(),
                "x0_hat": x0_hat_edge.detach(),
                "x0_hat_row": x0_hat_row.detach(),
            }

        y_t_row = self._row_from_edge_state(
            graph,
            xt_edge,
            active_edge,
            src,
            dst,
            veh_local,
            k_node,
            bsz,
        )

        t_node = t_graph[node_batch]
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

        xt_next = self._edge_onehot_from_row(
            y_next,
            veh_local,
            dst,
            active_edge,
        )

        return xt_next, p1, {
            "logits": logits.detach(),
            "p0_row": p0_row.detach(),
            "x0_hat": x0_hat_edge.detach(),
            "x0_hat_row": x0_hat_row.detach(),
            "y_t_row": y_t_row.detach(),
            "y_next_row": y_next.detach(),
        }

    # ---------------------------------------------------------------------
    # Row-wise training objective
    # ---------------------------------------------------------------------
    def _row_ce_from_logits(
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
            return logits.sum() * 0.0

        y = graph.y.view(-1).long().to(device)
        y = torch.minimum(torch.clamp_min(y, 0), k_node - 1)

        S = S.masked_fill(~valid, -10000.0)
        return F.cross_entropy(S, y)

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

    @staticmethod
    def _edge_onehot_from_row(
            y_row: torch.Tensor,
            veh_local: torch.Tensor,
            dst: torch.Tensor,
            active_edge: torch.Tensor,
    ) -> torch.Tensor:
        return (
                (veh_local.long() == y_row.long()[dst]).float()
                * active_edge.float()
        )


