from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotSelfAttentionBlock(nn.Module):
    """Graph-local vehicle-slot self-attention for homogeneous CVRP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(max(1, n_heads))

        if self.hidden_dim % self.n_heads != 0:
            self.n_heads = 1

        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)

        mid_dim = int(ffn_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, self.hidden_dim),
        )

        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, h_v: torch.Tensor, B: int) -> torch.Tensor:
        if h_v.numel() == 0:
            return h_v

        B = int(B)
        total_slots = int(h_v.size(0))

        if B <= 0:
            return h_v

        if total_slots % B != 0:
            raise ValueError(
                f"SlotSelfAttentionBlock expects fixed K per graph, "
                f"but got total_slots={total_slots}, B={B}."
            )

        K = total_slots // B
        if K <= 1:
            return h_v

        H = self.hidden_dim
        hv = h_v.reshape(B, K, H)

        qkv = self.qkv_proj(hv)
        qkv = qkv.view(B, K, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, K, H)
        out = self.out_proj(out)

        hv = self.norm_attn(hv + self.resid_dropout(out))
        hv = self.norm_ffn(hv + self.resid_dropout(self.ffn(hv)))

        return hv.reshape(total_slots, H)


class TypeAwareSlotSelfAttentionBlock(nn.Module):
    """Type-aware graph-local vehicle-slot self-attention for HFVRP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 2,
        pair_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(max(1, n_heads))

        if self.hidden_dim % self.n_heads != 0:
            self.n_heads = 1

        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        pair_dim = 5
        mid = int(pair_hidden_dim) if pair_hidden_dim is not None else max(32, self.hidden_dim // 4)

        self.pair_bias_mlp = nn.Sequential(
            nn.Linear(pair_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, self.n_heads),
        )

        nn.init.zeros_(self.pair_bias_mlp[-1].weight)
        nn.init.zeros_(self.pair_bias_mlp[-1].bias)

        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)

        ffn_dim = int(ffn_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ffn_dim, self.hidden_dim),
        )

        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    @staticmethod
    def _reshape_slot_attr(
        x: torch.Tensor,
        B: int,
        K: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return x.to(dtype=dtype).view(B, K)

    def forward(
        self,
        h_v: torch.Tensor,
        fleet_ctx: dict[str, torch.Tensor],
        veh_batch: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        del veh_batch

        if h_v.numel() == 0:
            return h_v

        B = int(B)
        total_slots = int(h_v.size(0))

        if B <= 0:
            return h_v

        if total_slots % B != 0:
            raise ValueError(
                f"TypeAwareSlotSelfAttentionBlock expects fixed K per graph, "
                f"but got total_slots={total_slots}, B={B}."
            )

        K = total_slots // B
        if K <= 1:
            return h_v

        H = self.hidden_dim
        dtype = h_v.dtype
        hv = h_v.view(B, K, H)

        cap = self._reshape_slot_attr(fleet_ctx["cap_v"], B, K, dtype)
        cap_rel = cap / cap.max(dim=1, keepdim=True).values.clamp_min(1e-6)

        fixed_rel = self._reshape_slot_attr(fleet_ctx["fixed_rel"], B, K, dtype)
        unit_rel = self._reshape_slot_attr(fleet_ctx["unit_rel"], B, K, dtype)
        tier_rel = self._reshape_slot_attr(fleet_ctx["tier_rel"], B, K, dtype)

        if "tier_raw" in fleet_ctx:
            tier_raw = self._reshape_slot_attr(fleet_ctx["tier_raw"], B, K, dtype)
        else:
            tier_raw = tier_rel

        slot_attr = torch.stack([cap_rel, fixed_rel, unit_rel, tier_rel], dim=-1)

        ai = slot_attr.unsqueeze(2)
        aj = slot_attr.unsqueeze(1)
        abs_diff = (ai - aj).abs()

        same_tier = (
            tier_raw.unsqueeze(2)
            .eq(tier_raw.unsqueeze(1))
            .to(dtype=dtype)
            .unsqueeze(-1)
        )

        pair_feat = torch.cat([same_tier, abs_diff], dim=-1)

        pair_bias = self.pair_bias_mlp(pair_feat)
        pair_bias = pair_bias.permute(0, 3, 1, 2).contiguous()
        pair_bias = pair_bias.to(dtype=dtype)

        qkv = self.qkv_proj(hv)
        qkv = qkv.view(B, K, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=pair_bias,
            dropout_p=attn_dropout,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, K, H)
        out = self.out_proj(out)

        hv = self.norm_attn(hv + self.resid_dropout(out))
        hv = self.norm_ffn(hv + self.resid_dropout(self.ffn(hv)))

        return hv.view(total_slots, H)


__all__ = [
    "SlotSelfAttentionBlock",
    "TypeAwareSlotSelfAttentionBlock",
]