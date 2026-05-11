import torch
import torch.nn.functional as F


class MetaConsistency:
    """Common consistency/guidance utilities for Stage-A VRP assignment models.

    Subclasses define:
      - which edges are active
      - how x0 edge labels are built
      - problem-specific regularizers / auxiliary consistency targets
    """

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
        self._Q_torch = None
        self.args = None

    # ------------------------------------------------------------------
    # basic utilities
    # ------------------------------------------------------------------
    def get_scalings_for_boundary_condition(self, sigma):
        if self.boundary_func == "sigmoid":
            c_out = torch.sigmoid(12 * (sigma - self.sigma_max / 2) / self.sigma_max) * sigma / (sigma + 1e-8)
            c_skip = 1 - c_out
        elif self.boundary_func == "linear":
            c_out = sigma / self.sigma_max
            c_skip = 1 - c_out
        elif self.boundary_func == "truncate":
            c_out = sigma / (sigma + 1e-9)
            c_skip = 1 - c_out
        else:
            raise NotImplementedError
        return c_skip, c_out

    def get_weightings(self, weight_schedule, snrs, sigma_data):
        if weight_schedule == "snr":
            weightings = snrs
        elif weight_schedule == "snr+1":
            weightings = snrs + 1
        elif weight_schedule == "karras":
            weightings = snrs + 1.0 / sigma_data**2
        elif weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()
        return weightings

    def append_dims(self, x, target_dims):
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]

    def _ensure_diffusion_mats(self, device, diffusion):
        if self._Q_bar_torch is None or self._Q_bar_torch.device != device:
            self._Q_bar_torch = torch.as_tensor(diffusion.Q_bar, device=device, dtype=torch.float32)
        if self._Q_torch is None or self._Q_torch.device != device:
            self._Q_torch = torch.as_tensor(diffusion.Qs, device=device, dtype=torch.float32)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.sum() * 0.0

    def _safe_prob01(self, p: torch.Tensor, default: float = 0.5) -> torch.Tensor:
        p = torch.nan_to_num(p, nan=default, posinf=default, neginf=default)
        return p.clamp(0.0, 1.0)

    def _veh_local_index(self, graph, src: torch.Tensor, bsz: int, device):
        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        counts = torch.bincount(veh_batch, minlength=max(1, bsz))
        starts = torch.cumsum(counts, 0) - counts
        veh_local = src.long() - starts[veh_batch[src.long()]]
        return node_batch, veh_batch, starts, veh_local

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

    def _edge_prob_from_logits(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, eps: float = 1e-12):
        device = logits.device
        edge_index = graph.edge_index.long().to(device)
        dst = edge_index[1]
        logits_m = torch.nan_to_num(logits.float(), nan=-1e30, posinf=1e30, neginf=-1e30)
        logits_m = logits_m.masked_fill(~active_edge, -1e30)
        p = self._group_softmax(logits_m, dst, eps=eps)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return p * active_edge.float()

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

    def q_sample_edge(self, x0_edge: torch.Tensor, t_edge: torch.Tensor, diffusion, active_edge: torch.Tensor = None) -> torch.Tensor:
        device = x0_edge.device
        self._ensure_diffusion_mats(device, diffusion)

        T = int(diffusion.T)
        t_edge = t_edge.long().clamp_(0, T)
        x0 = x0_edge.long().clamp_(0, 1)
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

    def _posterior_sample_binary(self, xt: torch.Tensor, t: int, target_t: int, p0: torch.Tensor, diffusion):
        if target_t < 0:
            raise ValueError("target_t must be >= 0")
        if t <= 0:
            return xt

        device = xt.device
        self._ensure_diffusion_mats(device, diffusion)

        Qb = self._Q_bar_torch[target_t]
        if target_t == t - 1:
            Qt = self._Q_torch[t - 1]
        else:
            Qt = torch.eye(2, device=device)
            for s in range(target_t, t):
                Qt = Qt @ self._Q_torch[s]

        tmp = p0 @ Qb
        col0 = Qt[:, 0].view(1, 2)
        col1 = Qt[:, 1].view(1, 2)
        like = torch.where(xt.view(-1, 1) > 0.5, col1, col0)
        post = tmp * like
        post = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        u = torch.rand((post.size(0),), device=device)
        x_prev = (u < post[:, 1]).float()
        return x_prev

    # ------------------------------------------------------------------
    # weights / extensibility
    # ------------------------------------------------------------------
    def _guide_weight(self, name: str) -> float:
        aliases = {
            "sim": ["c_sim", "c1"],
            "cap": ["c_cap", "c2"],
            "compact": ["c_compact"],
            "open": ["c_open"],
            "unused": ["c_unused_slot", "c_unused"],
            "cost_proxy": ["c_cost_proxy", "c_cost", "guide_cost_proxy"],
            "fill": ["c_fill", "c_fill_rate", "guide_fill", "guide_fill_rate"],
        }
        for key in aliases.get(name, [f"c_{name}"]):
            if hasattr(self.args, key):
                return float(getattr(self.args, key))
        return 0.0

    def _cm_weight(self, name: str) -> float:
        aliases = {
            "cap": ["cm_lam_cap", "lam_cap"],
            "compact": ["cm_lam_compact", "lam_compact"],
            "open": ["cm_lam_open", "lam_open"],
            "unused": ["cm_lam_unused_slot", "lam_unused_slot", "cm_lam_unused", "lam_unused"],
            "open_target": ["cm_lam_open_target", "lam_open_target"],
            "tier": ["cm_lam_tier", "lam_tier"],
            "slot_target": ["cm_lam_slot_target", "lam_slot_target", "cm_lam_slot", "lam_slot"],
            "cost_proxy": ["cm_lam_cost_proxy", "lam_cost_proxy", "cm_lam_cost", "lam_cost"],
            "fill": ["cm_lam_fill", "lam_fill", "cm_lam_fill_rate", "lam_fill_rate"],

            # NEW
            "row_consistency": [
                "cm_lam_row_ct", "lam_row_ct",
                "cm_lam_ct", "lam_ct",
                "cm_lam_row_consistency", "lam_row_consistency",
            ],
        }
        for key in aliases.get(name, [f"cm_lam_{name}", f"lam_{name}"]):
            if hasattr(self.args, key):
                val = getattr(self.args, key)
                if val is None:
                    continue
                return float(val)
        return 0.0

    def _extra_guidance_terms(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        return {}

    def _extra_consistency_terms(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        return {}

    # ------------------------------------------------------------------
    # proxies / regularizers
    # ------------------------------------------------------------------
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
        zero = self._zero(logits)
        cap_loss = self._capacity_proxy_loss(graph, logits, active_edge) if lam_cap > 0 else zero
        compact_loss = self._compactness_proxy_loss(graph, logits, active_edge) if lam_compact > 0 else zero
        return cap_loss, compact_loss

    # ------------------------------------------------------------------
    # guidance
    # ------------------------------------------------------------------
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

    def _edge_logits_to_input(
            self,
            xt_prob_or_hard: torch.Tensor,
            apply_jitter: bool = False,
            global_step: int = 0,
    ) -> torch.Tensor:
        if xt_prob_or_hard.ndim == 2:
            xt = xt_prob_or_hard[:, 1].float()
        else:
            xt = xt_prob_or_hard.float()

        xt = xt * 2.0 - 1.0

        if apply_jitter:
            j0 = float(getattr(self.args, "xt_jitter", 0.0))
            warm = int(getattr(self.args, "xt_jitter_warmup", 0))

            if j0 > 0:
                if warm > 0:
                    frac = max(0.0, 1.0 - float(global_step) / float(warm))
                    j = j0 * frac
                else:
                    j = j0

                if j > 0:
                    mult = (1.0 + j * torch.randn_like(xt)).clamp(0.8, 1.2)
                    xt = xt * mult

        return xt.clamp(-1.0, 1.0)

    def denoise_edge(self, model, graph, x_t_edge: torch.Tensor, t_edge: torch.Tensor, x0_onehot: torch.Tensor):
        xt_in = self._edge_logits_to_input(
            x_t_edge,
            apply_jitter=True,
            global_step=int(getattr(model, "global_step", 0)),
        )
        with torch.amp.autocast("cuda", enabled=False):
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
    # def denoise_edge(self, model, graph, x_t_edge: torch.Tensor, t_edge: torch.Tensor, x0_onehot: torch.Tensor):
    #     """
    #     Fast-T2T-style denoising step for binary edge labels.
    #
    #     Keep the HF edge-label setting, but align the x_t preprocessing with
    #     the original Fast-T2T code:
    #         x_t <- (2*x_t - 1) * (1 + 0.05 * rand)
    #     """
    #     # Fast-T2T style x_t preprocessing
    #     xt_in = x_t_edge.float() * 2.0 - 1.0
    #     xt_in = xt_in * (1.0 + 0.05 * torch.rand_like(xt_in))
    #
    #     with torch.amp.autocast("cuda", enabled=False):
    #         logit1 = model.forward_edge(graph, xt_in.float(), t_edge).float()
    #
    #     logit1 = torch.nan_to_num(logit1, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    #
    #     # keep the binary edge-head parameterization:
    #     # class-0 logit fixed at 0, class-1 logit = logit1
    #     model_output = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
    #
    #     c_skip, c_out = [
    #         self.append_dims(x, model_output.ndim).float()
    #         for x in self.get_scalings_for_boundary_condition(t_edge)
    #     ]
    #
    #     denoise = c_out * model_output + c_skip * x0_onehot.float()
    #     denoise = torch.nan_to_num(denoise, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    #
    #     return model_output, denoise

    def guided_logits_step(self, graph, logits: torch.Tensor, active_edge: torch.Tensor, x0_ref_edge: torch.Tensor = None, guide_scale: float = None):
        device = logits.device
        guide_scale = float(self._guidance_scale_at_step(1, 1) if guide_scale is None else guide_scale)

        weights = {
            "sim": self._guide_weight("sim"),
            "cap": self._guide_weight("cap"),
            "compact": self._guide_weight("compact"),
        }

        extra_weights = {
            name: self._guide_weight(name)
            for name in self._extra_guidance_terms(graph, logits.detach(), active_edge).keys()
        }
        weights.update(extra_weights)

        if guide_scale <= 0 or sum(abs(v) for v in weights.values()) <= 0:
            p1 = self._edge_prob_from_logits(graph, logits, active_edge)
            zero = self._zero(logits)
            return logits, p1, {"sim": zero.detach(), "cap": zero.detach(), "compact": zero.detach(), "total": zero.detach()}

        with torch.enable_grad():
            logits_g = logits.detach().float().clone().requires_grad_(True)

            named_terms = {}
            named_terms["sim"] = self._zero(logits_g)
            if x0_ref_edge is not None and weights.get("sim", 0.0) > 0 and bool(active_edge.any()):
                named_terms["sim"] = F.binary_cross_entropy_with_logits(
                    logits_g[active_edge],
                    x0_ref_edge.float().to(device)[active_edge],
                )

            named_terms["cap"] = self._zero(logits_g)
            if weights.get("cap", 0.0) > 0:
                named_terms["cap"] = self._capacity_proxy_loss(graph, logits_g, active_edge)

            named_terms["compact"] = self._zero(logits_g)
            if weights.get("compact", 0.0) > 0:
                named_terms["compact"] = self._compactness_proxy_loss(graph, logits_g, active_edge)

            extra_terms = self._extra_guidance_terms(graph, logits_g, active_edge)
            named_terms.update(extra_terms)

            total_loss = self._zero(logits_g)
            for name, term in named_terms.items():
                w = float(weights.get(name, 0.0))
                if w != 0.0:
                    total_loss = total_loss + w * term

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

        out_terms = {k: v.detach() for k, v in named_terms.items()}
        out_terms["total"] = total_loss.detach()
        return guided_logits, p1_guided, out_terms

    def cm_project_resample_step(self, model, graph, xt_edge: torch.Tensor, t_graph: torch.Tensor, t_next: int, active_edge: torch.Tensor, step_idx: int, total_steps: int):
        device = xt_edge.device
        deterministic = bool(getattr(self.args, "eval_deterministic", False)) or bool(getattr(self.args, "guided_deterministic", False))
        edge_index = graph.edge_index.long().to(device)
        dst = edge_index[1]

        xt_in = self._edge_logits_to_input(
            xt_edge,
            apply_jitter=False,
            global_step=int(getattr(model, "global_step", 0)),
        )
        logits = model.forward(graph, xt_in.float(), t_graph).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        p1_unguided = self._edge_prob_from_logits(graph, logits, active_edge)

        guidance_enabled = bool(getattr(self.args, "guided", False))
        guide_scale = self._guidance_scale_at_step(step_idx, total_steps) if guidance_enabled else 0.0

        if guide_scale > 0:
            x0_ref = self._sample_group_onehot(p1_unguided, dst, active_edge, deterministic=deterministic)
            logits_used, p1, guide_losses = self.guided_logits_step(
                graph,
                logits,
                active_edge,
                x0_ref_edge=x0_ref,
                guide_scale=guide_scale,
            )
        else:
            logits_used = logits
            p1 = p1_unguided
            z = self._zero(logits)
            guide_losses = {"sim": z.detach(), "cap": z.detach(), "compact": z.detach(), "total": z.detach()}

        x0_hat = self._sample_group_onehot(p1, dst, active_edge, deterministic=deterministic)
        if int(t_next) <= 0:
            return x0_hat, p1, {
                "logits": logits_used.detach(),
                "x0_hat": x0_hat.detach(),
                "guide_scale": float(guide_scale),
                "guide_losses": guide_losses,
            }

        self._ensure_diffusion_mats(device, model.diffusion)
        Qb_tnext = self._Q_bar_torch[int(t_next)]

        x0_oh = torch.stack([1.0 - x0_hat, x0_hat], dim=-1).float()
        probs_tnext = x0_oh @ Qb_tnext
        probs_tnext = probs_tnext / probs_tnext.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs_tnext = torch.nan_to_num(probs_tnext, nan=0.5, posinf=0.5, neginf=0.5)

        p1_next = self._safe_prob01(probs_tnext[:, 1], default=0.5)
        xt_next = (p1_next >= 0.5).float() if deterministic else torch.bernoulli(p1_next).float()
        xt_next = xt_next * active_edge.float()

        return xt_next, p1, {
            "logits": logits_used.detach(),
            "x0_hat": x0_hat.detach(),
            "x_tnext_prob": probs_tnext.detach(),
            "guide_scale": float(guide_scale),
            "guide_losses": guide_losses,
        }

    # ------------------------------------------------------------------
    # consistency training
    # ------------------------------------------------------------------
    def _consistency_aux_terms(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        terms = {}
        if self._cm_weight("cap") > 0:
            terms["cap"] = self._capacity_proxy_loss(graph, logits, active_edge)
        if self._cm_weight("compact") > 0:
            terms["compact"] = self._compactness_proxy_loss(graph, logits, active_edge)
        terms.update(self._extra_consistency_terms(graph, logits, active_edge))
        return terms

    def consistency_losses(self, model, batch):
        if str(getattr(model.args, "task", "")).lower().strip() == "hfvrp":
            raise RuntimeError(
                "HFVRP must not use MetaConsistency.consistency_losses(), because it is "
                "edge-binary. Use HFVRPConsistency.consistency_losses() with row-categorical "
                "diffusion instead."
            )

        device = model.device
        graph = batch.to(device)

        node_batch = graph.node_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]
        bsz = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 0

        active_edge = self._get_active_edge(graph, edge_graph, src, dst, bsz, device)
        x0_edge = self._get_x0_edge(graph, src, dst, active_edge, edge_graph, bsz, device)
        x0_edge = x0_edge.long().clamp_(0, 1) * active_edge.long()
        x0_onehot = F.one_hot(x0_edge, num_classes=2).float()

        if bsz <= 0 or x0_edge.numel() == 0:
            return self._zero(x0_edge.float())

        t_max = int(model.diffusion.T)
        t_graph = torch.randint(1, t_max + 1, (bsz,), device=device)
        alpha = getattr(model.args, "alpha", 0.5)
        if alpha is None:
            alpha = 0.5
        t2_graph = (float(alpha) * t_graph.float()).long().clamp_(min=0)
        t_edge = t_graph[edge_graph]
        t2_edge = t2_graph[edge_graph]

        x_t = model.q_sample_edge(x0_edge.float(), t_edge, active_edge=active_edge)
        x_t2 = model.q_sample_edge(x0_edge.float(), t2_edge, active_edge=active_edge)

        raw1, denoise1 = self.denoise_edge(model, graph, x_t, t_edge, x0_onehot)
        raw2, denoise2 = self.denoise_edge(model, graph, x_t2, t2_edge, x0_onehot)

        if not bool(active_edge.any()):
            return self._zero(denoise1)

        denoise1_act = denoise1[active_edge].float()
        denoise2_act = denoise2[active_edge].float()
        target = x0_edge[active_edge]

        if (not torch.isfinite(denoise1_act).all()) or (not torch.isfinite(denoise2_act).all()):
            bad1 = (~torch.isfinite(denoise1_act)).any(dim=-1).sum().item()
            bad2 = (~torch.isfinite(denoise2_act)).any(dim=-1).sum().item()
            raise RuntimeError(
                f"non-finite denoise in consistency_losses: bad1={bad1}, bad2={bad2}, "
                f"x_t_range=({x_t.min().item():.4f},{x_t.max().item():.4f}), "
                f"x_t2_range=({x_t2.min().item():.4f},{x_t2.max().item():.4f})"
            )

        ce1 = F.cross_entropy(denoise1_act, target)
        ce2 = F.cross_entropy(denoise2_act, target)

        logits1 = raw1[:, 1].float()
        logits2 = raw2[:, 1].float()

        terms1 = self._consistency_aux_terms(graph, logits1, active_edge)
        terms2 = self._consistency_aux_terms(graph, logits2, active_edge)

        aux1 = self._zero(logits1)
        aux2 = self._zero(logits2)
        for name, val in terms1.items():
            aux1 = aux1 + self._cm_weight(name) * val
        for name, val in terms2.items():
            aux2 = aux2 + self._cm_weight(name) * val

        loss = ce1 + ce2 + aux1 + aux2
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"consistency_losses became non-finite: "
                f"ce1={ce1.item()}, ce2={ce2.item()}, aux1={aux1.item()}, aux2={aux2.item()}"
            )

        if hasattr(model, "log_dict"):
            is_hf_type_mode = bool(
                getattr(model.args, "task", "") == "hfvrp"
                and str(getattr(self.args, "hf_cm_x0_mode", "slot")).lower().strip() == "type"
            )

            if is_hf_type_mode:
                log_items = {
                    "train/cm_ce_type_t": ce1.detach(),
                    "train/cm_ce_type_t2": ce2.detach(),
                    # 兼容旧图表，先保留一份别名
                    "train/cm_ce_t": ce1.detach(),
                    "train/cm_ce_t2": ce2.detach(),
                }
            else:
                log_items = {
                    "train/cm_ce_t": ce1.detach(),
                    "train/cm_ce_t2": ce2.detach(),
                }

            for name in sorted(set(list(terms1.keys()) + list(terms2.keys()))):
                v1 = terms1.get(name, self._zero(loss)).detach()
                v2 = terms2.get(name, self._zero(loss)).detach()
                log_items[f"train/cm_{name}"] = 0.5 * (v1 + v2)

            if is_hf_type_mode:
                log_items["train/cm_x0_mode_type"] = torch.tensor(1.0, device=loss.device)

            model.log_dict(log_items, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------
    def _get_active_edge(self, graph, edge_graph: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, bsz: int, device):
        raise NotImplementedError

    def _get_x0_edge(self, graph, src: torch.Tensor, dst: torch.Tensor, active_edge: torch.Tensor, edge_graph: torch.Tensor, bsz: int, device):
        raise NotImplementedError

    def _capacity_proxy_loss(self, graph, logits: torch.Tensor, active_edge: torch.Tensor):
        raise NotImplementedError