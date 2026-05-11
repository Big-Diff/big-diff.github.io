"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
import torch.nn.functional as F


class CategoricalDiffusion(object):
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    # diffusion/utils/diffusion_schedulers.py
    def _ensure_onehot(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Make sure x0 becomes one-hot with last dim=2.
        Accepts:
          - label tensor in {0,1} with shape (B, ...) or (...)
          - already-onehot tensor with shape (B, ..., 2) or (..., 2)
          - float labels: treated as prob/logit-like; threshold at 0.5
        Returns:
          one-hot float tensor with shape (B, ..., 2) (or (..., 2) if no batch)
        """
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0)

        # already one-hot/prob form
        if x0.dim() >= 1 and x0.size(-1) == 2:
            return x0.float()

        # convert to 0/1 labels
        if x0.dtype.is_floating_point:
            x0_lbl = (x0 > 0.5).long()
        else:
            x0_lbl = x0.long()

        x0_lbl = x0_lbl.clamp_(0, 1)
        x0_oh = F.one_hot(x0_lbl, num_classes=2).float()
        return x0_oh

    def sample(self, x0, t):
        x0_onehot = self._ensure_onehot(x0)  # (B, ..., 2)
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)  # (B,2,2)

        # 对每个位置的 one-hot 向量做类别转移：(...,2) x (2,2) -> (...,2)
        xt_prob = torch.einsum("b...c,bck->b...k", x0_onehot, Q_bar)

        # 采样得到 {0,1} 标签
        return torch.bernoulli(xt_prob[..., 1].clamp(0, 1))

    def consistency_sample(self, x0, t, t2):
        x0_onehot = self._ensure_onehot(x0)  # (B, ..., 2)
        Q_bar_t = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)  # (B,2,2)
        Q_bar_t2 = torch.from_numpy(self.Q_bar[t2]).float().to(x0_onehot.device)  # (B,2,2)

        xt_prob = torch.einsum("b...c,bck->b...k", x0_onehot, Q_bar_t)
        xt2_prob = torch.einsum("b...c,bck->b...k", x0_onehot, Q_bar_t2)

        # 从 t2 映射回 t：xt_prob @ inv(Q_bar_t2) @ Q_bar_t
        M = torch.linalg.inv(Q_bar_t2) @ Q_bar_t  # (B,2,2)
        xt_prob_from_t2 = torch.einsum("b...c,bck->b...k", xt2_prob, M)

        return (
            torch.bernoulli(xt_prob[..., 1].clamp(0, 1)),
            torch.bernoulli(xt_prob_from_t2[..., 1].clamp(0, 1)),
        )


class InferenceSchedule(object):
    def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
