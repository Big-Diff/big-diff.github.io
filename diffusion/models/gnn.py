import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import MessagePassing

class CVRPVehNodeData(Data):
    def __init__(
        self,
        veh_features=None,
        node_features=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        node_batch=None,
        veh_batch=None,
        capacity=None,
        depot_xy=None,
        actions=None,
        gt_cost=None,
        graph_feat=None,
        node_knn_edge_index=None,
        node_knn_edge_attr=None,
    ):
        super().__init__()

        if veh_features is not None:
            self.veh_features = veh_features
        if node_features is not None:
            self.node_features = node_features
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y

        if node_batch is not None:
            self.node_batch = node_batch
        if veh_batch is not None:
            self.veh_batch = veh_batch

        if capacity is not None:
            self.capacity = capacity
        if depot_xy is not None:
            self.depot_xy = depot_xy

        if actions is not None:
            self.actions = actions
        if gt_cost is not None:
            self.gt_cost = gt_cost

        if graph_feat is not None:
            self.graph_feat = graph_feat
        if node_knn_edge_index is not None:
            self.node_knn_edge_index = node_knn_edge_index
        if node_knn_edge_attr is not None:
            self.node_knn_edge_attr = node_knn_edge_attr

        self.src_count = int(veh_features.size(0)) if veh_features is not None else 0
        self.dst_count = int(node_features.size(0)) if node_features is not None else 0

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == "edge_index":
            src = self.src_count if self.src_count > 0 else int(self.veh_features.size(0))
            dst = self.dst_count if self.dst_count > 0 else int(self.node_features.size(0))
            return torch.tensor([[src], [dst]])

        if key == "node_knn_edge_index":
            n = self.dst_count if self.dst_count > 0 else int(self.node_features.size(0))
            return torch.tensor([[n], [n]])

        if key in ["node_batch", "veh_batch"]:
            return int(value.max().item()) + 1 if torch.is_tensor(value) and value.numel() > 0 else 1

        return super().__inc__(key, value, store, *args, **kwargs)


class BipartiteGraphConvolution(MessagePassing):
    """
    Bipartite message passing with multi-head attention (QKV) + edge feature injection.

    Edge direction:
        edge_index[0] = left/src indices (e.g., vehicle)
        edge_index[1] = right/dst indices (e.g., customer/node)

    We compute attention per dst node:
        alpha_{j->i} = softmax_{j in N(i)}( <q_i, k_j + k_e> / sqrt(d) )
        msg_{j->i}   = alpha_{j->i} * (v_j + v_e)

    Then aggregate (sum) over incoming edges for each dst node.
    """

    def __init__(
        self,
        embd_size: int,
        edge_dim: int = 4,
        *,
        n_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
        deg_norm_alpha: float = 0.0,  # attention already normalizes; keep 0.0 by default
    ):
        super().__init__(aggr="add", node_dim=0)
        self.embd_size = int(embd_size)
        self.edge_dim = int(edge_dim)

        self.n_heads = int(max(1, n_heads))
        if head_dim is None:
            if self.embd_size % self.n_heads != 0:
                # fallback: 1 head so shapes always legal
                self.n_heads = 1
                self.head_dim = self.embd_size
            else:
                self.head_dim = self.embd_size // self.n_heads
        else:
            self.head_dim = int(head_dim)
            # ensure output dim is consistent
            if self.n_heads * self.head_dim != self.embd_size:
                # project to embd_size anyway (still works), but keep stable by forcing match
                # simplest: override head_dim to match embd_size
                if self.embd_size % self.n_heads != 0:
                    self.n_heads = 1
                    self.head_dim = self.embd_size
                else:
                    self.head_dim = self.embd_size // self.n_heads

        self.dropout = float(dropout)
        self.deg_norm_alpha = float(deg_norm_alpha)

        # Q (dst), K/V (src)
        self.Wq = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.embd_size, self.n_heads * self.head_dim, bias=False)

        # edge -> per-head bias/value
        self.Wke = nn.Linear(self.edge_dim, self.n_heads * self.head_dim, bias=False)
        self.Wve = nn.Linear(self.edge_dim, self.n_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.embd_size, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.post_conv_module = nn.Sequential(nn.BatchNorm1d(self.embd_size))

        # keep your original "residual mix with right_features" style
        self.output_module = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features, edge_logit_bias=None):
        out = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            left=left_features,
            right=right_features,
            edge_features=edge_features,
            edge_logit_bias=edge_logit_bias,
        )  # (Nr, n_heads, head_dim) aggregated

        out = out.reshape(right_features.size(0), self.n_heads * self.head_dim)
        out = self.out_proj(out)

        # optional degree scaling (usually keep 0.0 for attention)
        if self.deg_norm_alpha > 0:
            dst = edge_indices[1]
            deg = torch.bincount(dst, minlength=right_features.size(0)).float().clamp_min(1.0)
            out = out / (deg.unsqueeze(-1) ** self.deg_norm_alpha)

        out = self.post_conv_module(out)
        return self.output_module(torch.cat([out, right_features], dim=-1))

    def message(self, right_i, left_j, edge_features, edge_logit_bias, index, ptr, size_i):
        q = self.Wq(right_i).view(-1, self.n_heads, self.head_dim)  # (E,H,d)
        k = self.Wk(left_j).view(-1, self.n_heads, self.head_dim)  # (E,H,d)
        v = self.Wv(left_j).view(-1, self.n_heads, self.head_dim)  # (E,H,d)
        ke = self.Wke(edge_features).view(-1, self.n_heads, self.head_dim)
        ve = self.Wve(edge_features).view(-1, self.n_heads, self.head_dim)

        logits = (q * (k + ke)).sum(dim=-1) / (float(self.head_dim) ** 0.5)  # (E,H)

        # ---- NEW: add per-edge feasibility bias before softmax ----
        if edge_logit_bias is not None:
            # edge_logit_bias: (E,) or (E,1) or (E,H)
            if edge_logit_bias.dim() == 1:
                logits = logits + edge_logit_bias.unsqueeze(-1)  # (E,1) -> broadcast to (E,H)
            elif edge_logit_bias.dim() == 2:
                if edge_logit_bias.size(1) == 1:
                    logits = logits + edge_logit_bias  # (E,1) -> broadcast
                else:
                    logits = logits + edge_logit_bias  # (E,H)
            else:
                raise ValueError("edge_logit_bias must have shape (E,), (E,1), or (E,H)")

        alpha = pyg_softmax(logits, index=index, ptr=ptr, num_nodes=size_i)  # (E,H)
        alpha = self.attn_dropout(alpha)

        return (v + ve) * alpha.unsqueeze(-1)

class _SinTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float().view(-1, 1)
        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(0, half, device=device, dtype=torch.float32)
            / max(1, half - 1)
        ).view(1, -1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb



class LiteGatedGCNLayer(nn.Module):
    """Light gated node-node propagation for local customer geometry."""

    def __init__(self, hidden_dim: int, edge_dim: int = 7, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.dropout = float(dropout)

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(edge_dim, hidden_dim, bias=True)

        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h_in = h
        Uh = self.U(h)

        # DDP-safe: even with empty edges we still execute parameterized ops.
        if edge_index.numel() == 0:
            out = F.dropout(F.silu(Uh), p=self.dropout, training=self.training)
            if out.dtype != h_in.dtype:
                out = out.to(h_in.dtype)
            h_out = self.norm_h(h_in + out)
            e_out = edge_attr.new_zeros((0, self.hidden_dim))
            return h_out, e_out

        src, dst = edge_index[0], edge_index[1]
        Vh = self.V(h[src])
        Ah = self.A(h[dst])
        Bh = self.B(h[src])
        Ce = self.C(edge_attr)

        e = Ah + Bh + Ce
        gates = torch.sigmoid(e)
        msg = gates * Vh

        # AMP/fp16-safe: aggregation buffer must match msg dtype.
        agg = msg.new_zeros((h.size(0), self.hidden_dim))
        agg.index_add_(0, dst, msg)

        out = F.dropout(F.silu(Uh + agg), p=self.dropout, training=self.training)
        if out.dtype != h_in.dtype:
            out = out.to(h_in.dtype)
        if e.dtype != h_in.dtype:
            e = e.to(h_in.dtype)

        h_out = self.norm_h(h_in + out)
        e_out = self.norm_e(F.silu(e))
        return h_out, e_out

class SparseKNNNodeAttentionLayer(nn.Module):
    """Sparse edge-aware attention over customer KNN graph.

    This layer performs n2n message passing on the already-built KNN graph.

    Important convention:
        edge_index[0] = current / receiver node
        edge_index[1] = KNN neighbor / source node

    Therefore each customer attends to its own KNN neighbors without any
    per-graph Python loop.

    Input:
        h:          [sumN, hidden_dim]
        edge_index: [2, E_knn]
        edge_attr:  [E_knn, edge_dim]

    Output:
        h_out:      [sumN, hidden_dim]
        e_out:      [E_knn, hidden_dim]  # returned for API compatibility
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 7,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 2,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.n_heads = int(max(1, n_heads))

        if self.hidden_dim % self.n_heads != 0:
            self.n_heads = 1

        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.Wke = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.Wve = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.edge_bias = nn.Linear(self.edge_dim, self.n_heads, bias=False)

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)
        self.norm_e = nn.LayerNorm(self.hidden_dim)

        mid_dim = int(ffn_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, self.hidden_dim),
        )

        self.attn_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        h_in = h

        if edge_index.numel() == 0:
            # Keep a valid residual path for rare empty-graph cases.
            zero = self.out_proj(self.Wv(h) * 0.0)
            h = self.norm_attn(h + zero)
            h = self.norm_ffn(h + self.resid_dropout(self.ffn(h)))
            e_out = edge_attr.new_zeros((0, self.hidden_dim))
            return h, e_out

        # KNN convention from _build_knn_edges_by_batch:
        #   cur: current node that receives information
        #   nbr: its KNN neighbor that provides key/value
        cur = edge_index[0].long()
        nbr = edge_index[1].long()

        E = int(cur.numel())
        Hh = self.n_heads
        Dh = self.head_dim

        edge_attr = edge_attr.to(dtype=h.dtype)

        q = self.Wq(h[cur]).view(E, Hh, Dh)
        k = self.Wk(h[nbr]).view(E, Hh, Dh)
        v = self.Wv(h[nbr]).view(E, Hh, Dh)

        ke = self.Wke(edge_attr).view(E, Hh, Dh)
        ve = self.Wve(edge_attr).view(E, Hh, Dh)
        eb = self.edge_bias(edge_attr)  # [E, heads]

        logits = (q * (k + ke)).sum(dim=-1) / (float(Dh) ** 0.5)
        logits = logits + eb

        # Softmax over each current node's KNN neighborhood.
        alpha = pyg_softmax(logits, index=cur, num_nodes=h.size(0))
        alpha = self.attn_dropout(alpha)

        msg = (v + ve) * alpha.unsqueeze(-1)
        msg = msg.reshape(E, self.hidden_dim)

        agg = msg.new_zeros((h.size(0), self.hidden_dim))
        agg.index_add_(0, cur, msg)

        out = self.out_proj(agg)

        h = self.norm_attn(h_in + self.resid_dropout(out))
        h = self.norm_ffn(h + self.resid_dropout(self.ffn(h)))

        # Returned only for API compatibility with LiteGatedGCNLayer.
        e_out = self.norm_e(F.silu((ke + ve).reshape(E, self.hidden_dim)))

        return h, e_out

class SlotSelfAttentionBlock(nn.Module):
    """Graph-local vehicle-slot self-attention.

    This block performs v2v message passing among candidate vehicle slots inside
    each CVRP instance.

    Input:
        h_v: [B * K, H]

    Internal:
        h_v -> [B, K, H]
        self-attention over K slots for each graph

    No per-graph Python loop is used.
    """

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
            # Keep shapes legal and deterministic.
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

        # [B, K, H]
        hv = h_v.reshape(B, K, H)

        # [B, K, 3H] -> [B, K, 3, heads, head_dim]
        qkv = self.qkv_proj(hv)
        qkv = qkv.view(B, K, 3, self.n_heads, self.head_dim)

        # q, k, v: [B, heads, K, head_dim]
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0

        # Uses PyTorch SDPA backend; on supported GPUs this can dispatch to
        # optimized kernels automatically.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=False,
        )

        # [B, heads, K, head_dim] -> [B, K, H]
        out = out.transpose(1, 2).contiguous().view(B, K, H)
        out = self.out_proj(out)

        hv = self.norm_attn(hv + self.resid_dropout(out))
        hv = self.norm_ffn(hv + self.resid_dropout(self.ffn(hv)))

        return hv.reshape(total_slots, H)


class EdgeBipartiteDenoiserV4(nn.Module):
    """
    V4:
      - attention only on vehicle<->node assignment
      - lightweight gated n2n propagation on sparse local graph
      - optional v2v slot self-attention for route-level slot coordination
      - static prior dynamic context from xt, plus sparse refresh during layers
      - thin edge state: static edge embedding + dynamic hidden/bias injection
      - DDP-safe module creation: only create modules for actually used layers
    """

    def __init__(
        self,
        node_in_dim: int = 10,
        veh_in_dim: int = 7,
        edge_in_dim: int = 8,
        graph_in_dim: int = 6,
        hidden_dim: int = 192,
        n_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        biattn_heads: int = 4,
        biattn_dropout: float = 0.0,
        biattn_head_dim: int = None,
        use_n2n: bool = True,
        use_v2v: bool = False,
        use_global: bool = True,
        use_adaln: bool = False,
        n2n_knn_k: int = 8,
        dyn_refresh_every: int = 2,
        intra_every: int = 2,

        n2n_mode: str = "gated",
        n2n_attn_heads: int = 4,
        n2n_attn_dropout: float = 0.05,
        n2n_attn_ffn_mult: int = 2,

        v2v_every: int = 2,
        v2v_heads: int = 4,
        v2v_dropout: float = 0.05,
        v2v_ffn_mult: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.use_n2n = bool(use_n2n)
        self.use_global = bool(use_global)
        self.use_adaln = bool(use_adaln)
        self.use_v2v_requested = bool(use_v2v)
        self.use_v2v = bool(use_v2v)
        self.n2n_knn_k = int(n2n_knn_k)
        self.edge_dyn_dim = 5
        self.dyn_refresh_every = max(1, int(dyn_refresh_every))
        self.intra_every = max(1, int(intra_every))

        self.n2n_mode = str(n2n_mode).lower()
        if self.n2n_mode in {"attention", "knn_attention", "knn_attn"}:
            self.n2n_mode = "attn"
        if self.n2n_mode not in {"gated", "attn"}:
            raise ValueError(f"Unsupported n2n_mode={n2n_mode!r}. Use 'gated' or 'attn'.")

        self.n2n_attn_heads = int(n2n_attn_heads)
        self.n2n_attn_dropout = float(n2n_attn_dropout)
        self.n2n_attn_ffn_mult = int(n2n_attn_ffn_mult)
        self.v2v_every = max(1, int(v2v_every))
        self.v2v_heads = int(v2v_heads)
        self.v2v_dropout = float(v2v_dropout)
        self.v2v_ffn_mult = int(v2v_ffn_mult)
        refresh_layer_ids = {0, self.n_layers - 1}
        refresh_layer_ids.update(range(0, self.n_layers, self.dyn_refresh_every))
        self.refresh_layer_ids = sorted(int(x) for x in refresh_layer_ids if 0 <= x < self.n_layers)
        self.refresh_layer_to_idx = {l: i for i, l in enumerate(self.refresh_layer_ids)}

        if self.use_n2n:
            intra_layer_ids = set(range(0, self.n_layers, self.intra_every))
            self.intra_layer_ids = sorted(int(x) for x in intra_layer_ids if 0 <= x < self.n_layers)
        else:
            self.intra_layer_ids = []
        self.intra_layer_to_idx = {l: i for i, l in enumerate(self.intra_layer_ids)}

        if self.use_v2v:
            v2v_layer_ids = set(range(0, self.n_layers, self.v2v_every))
            self.v2v_layer_ids = sorted(int(x) for x in v2v_layer_ids if 0 <= x < self.n_layers)
        else:
            self.v2v_layer_ids = []

        self.v2v_layer_to_idx = {l: i for i, l in enumerate(self.v2v_layer_ids)}

        self.time_emb = _SinTimeEmbed(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.veh_proj = nn.Sequential(
            nn.Linear(veh_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(graph_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        H = int(max(1, biattn_heads))
        hd = biattn_head_dim
        if hd is None:
            if hidden_dim % H != 0:
                H, hd = 1, hidden_dim
            else:
                hd = hidden_dim // H
        else:
            hd = int(hd)
            if H * hd != hidden_dim:
                if hidden_dim % H != 0:
                    H, hd = 1, hidden_dim
                else:
                    hd = hidden_dim // H

        self.v2n = nn.ModuleList(
            [
                BipartiteGraphConvolution(
                    hidden_dim,
                    edge_dim=hidden_dim,
                    n_heads=H,
                    head_dim=hd,
                    dropout=float(biattn_dropout),
                    deg_norm_alpha=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.n2v = nn.ModuleList(
            [
                BipartiteGraphConvolution(
                    hidden_dim,
                    edge_dim=hidden_dim,
                    n_heads=H,
                    head_dim=hd,
                    dropout=float(biattn_dropout),
                    deg_norm_alpha=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.use_n2n:
            if self.n2n_mode == "attn":
                self.n2n = nn.ModuleList(
                    [
                        SparseKNNNodeAttentionLayer(
                            hidden_dim=hidden_dim,
                            edge_dim=7,
                            n_heads=self.n2n_attn_heads,
                            dropout=self.n2n_attn_dropout,
                            ffn_mult=self.n2n_attn_ffn_mult,
                        )
                        for _ in self.intra_layer_ids
                    ]
                )
            else:
                self.n2n = nn.ModuleList(
                    [
                        LiteGatedGCNLayer(
                            hidden_dim=hidden_dim,
                            edge_dim=7,
                            dropout=float(dropout),
                        )
                        for _ in self.intra_layer_ids
                    ]
                )
        else:
            self.n2n = None

        self.v2v = nn.ModuleList(
            [
                SlotSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    n_heads=self.v2v_heads,
                    dropout=self.v2v_dropout,
                    ffn_mult=self.v2v_ffn_mult,
                )
                for _ in self.v2v_layer_ids
            ]
        ) if self.use_v2v else None

        self.norm_n_intra = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_n_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v_global = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_msg_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.edge_out_norm = nn.LayerNorm(hidden_dim)

        self.adaLN = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 8)) for _ in range(self.n_layers)]
        ) if self.use_adaln else None

        self.pre_score = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in self.refresh_layer_ids
            ]
        )

        self.edge_dyn_proj = nn.Sequential(
            nn.Linear(self.edge_dyn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(self.edge_dyn_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.edge_delta_bias = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3 + self.edge_dyn_dim + 1, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.vehicle_global = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.global_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _batch_mean(x: torch.Tensor, batch: torch.Tensor, B: int):
        out = x.new_zeros((B, x.size(-1)))
        cnt = x.new_zeros((B, 1))
        out.index_add_(0, batch, x)
        cnt.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
        return out / cnt.clamp_min(1.0)

    @staticmethod
    def _build_knn_edges_by_batch(
            node_xy: torch.Tensor,
            batch_ids: torch.Tensor,
            k: int,
            B: int | None = None,
    ):
        """Build node-node KNN edges for fixed-size batched CVRP instances.

        Assumption:
            Nodes are stored in PyG batch order:

                graph0 nodes, graph1 nodes, ..., graph(B-1) nodes

            and every graph in the batch has the same number of nodes N.

        Edge direction is kept identical to the old implementation:

            src = current node
            dst = one of its K nearest neighbours
        """
        device = node_xy.device
        k = int(k)

        if k <= 0 or node_xy.numel() == 0:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        total_nodes = int(node_xy.size(0))

        if B is None:
            if batch_ids.numel() == 0:
                B = 0
            else:
                # One scalar read only when B is not provided.
                # In forward(), pass B explicitly to avoid recomputing it here.
                B = int(batch_ids[-1].detach().item()) + 1

        B = int(B)

        if B <= 0:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        if total_nodes % B != 0:
            raise ValueError(
                f"Fixed-batch KNN requires total_nodes % B == 0, "
                f"got total_nodes={total_nodes}, B={B}."
            )

        N = total_nodes // B

        if N <= 1:
            return torch.empty((2, 0), device=device, dtype=torch.long)

        kk = min(k, N - 1)

        # [B, N, 2]
        xy = node_xy.reshape(B, N, node_xy.size(-1))

        # [B, N, N]
        dist = torch.cdist(xy, xy, p=2)

        diag = torch.arange(N, device=device)
        dist[:, diag, diag] = float("inf")

        # nn_idx[b, i, :] gives nearest-neighbour local indices of node i.
        # Shape: [B, N, kk]
        nn_idx = torch.topk(dist, kk, largest=False, dim=-1).indices

        base = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * N

        # Preserve old edge direction:
        #   src = current node i
        #   dst = neighbour nn_idx[b, i, :]
        src_local = torch.arange(N, device=device, dtype=torch.long).view(1, N, 1).expand(B, N, kk)
        dst_local = nn_idx

        src = (base + src_local).reshape(-1)
        dst = (base + dst_local).reshape(-1)

        return torch.stack([src, dst], dim=0)

    @staticmethod
    def _row_normalize_by_dst(
            x: torch.Tensor,
            dst: torch.Tensor,
            num_nodes: int,
            eps: float = 1e-12,
    ):
        """Normalize edge values per customer row.

        Args:
            x:
                Edge values, shape [E].
            dst:
                Destination customer index of each edge, shape [E].
            num_nodes:
                Number of customer nodes.
            eps:
                Numerical stability constant.

        Returns:
            Row-normalized edge values, shape [E].
        """
        x = torch.nan_to_num(
            x.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        denom = x.new_zeros((num_nodes,))
        denom.index_add_(0, dst, x)

        p = x / denom[dst].clamp_min(eps)

        # Fallback for all-zero rows.
        # Compute it unconditionally to avoid bool(tensor.any()) GPU sync.
        ones = torch.ones_like(x)
        cnt = x.new_zeros((num_nodes,))
        cnt.index_add_(0, dst, ones)
        uniform = ones / cnt[dst].clamp_min(1.0)

        bad = denom[dst] <= eps
        p = torch.where(bad, uniform, p)

        return p

    @staticmethod
    def _n2n_edge_attr(node_xy: torch.Tensor, demand: torch.Tensor, nn_edge_index: torch.Tensor):
        if nn_edge_index.numel() == 0:
            return torch.empty((0, 7), device=node_xy.device, dtype=node_xy.dtype)
        src, dst = nn_edge_index[0], nn_edge_index[1]
        dxy = node_xy[src] - node_xy[dst]
        dist = torch.sqrt((dxy ** 2).sum(dim=-1) + 1e-12)
        inv = 1.0 / (dist + 1e-6)
        di = demand[src].to(node_xy.dtype)
        dj = demand[dst].to(node_xy.dtype)
        return torch.cat(
            [
                dxy,
                dist.unsqueeze(-1),
                inv.unsqueeze(-1),
                di.unsqueeze(-1),
                dj.unsqueeze(-1),
                (di - dj).abs().unsqueeze(-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _scatter_add_1d(index: torch.Tensor, val: torch.Tensor, out_size: int):
        out = val.new_zeros((out_size,))
        out.index_add_(0, index, val)
        return out

    @staticmethod
    def _scatter_add_2d(index: torch.Tensor, val: torch.Tensor, out_size: int):
        out = val.new_zeros((out_size, val.size(-1)))
        out.index_add_(0, index, val)
        return out

    def _build_vehicle_state(
        self,
        src,
        dst,
        node_batch,
        veh_batch,
        node_xy,
        node_unit,
        demand,
        capacity_b,
        p,
    ):
        Ktot = int(veh_batch.numel())
        cap_v = capacity_b[veh_batch].clamp_min(1e-6)

        load = self._scatter_add_1d(src, p * demand[dst], Ktot)
        count = self._scatter_add_1d(src, p, Ktot)

        centroid_num = self._scatter_add_2d(src, p.unsqueeze(-1) * node_xy[dst], Ktot)
        centroid = centroid_num / count.unsqueeze(-1).clamp_min(1e-6)

        dir_num = self._scatter_add_2d(src, p.unsqueeze(-1) * node_unit[dst], Ktot)
        mean_dir = dir_num / count.unsqueeze(-1).clamp_min(1e-6)
        mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        dist_to_centroid = (node_xy[dst] - centroid[src]).norm(dim=-1)
        radius_num = self._scatter_add_1d(src, p * dist_to_centroid, Ktot)
        radius = radius_num / count.clamp_min(1e-6)

        node_cnt_b = torch.bincount(node_batch, minlength=int(capacity_b.numel())).float().to(node_xy.device).clamp_min(1.0)
        client_count_ratio = count / node_cnt_b[veh_batch]

        load_ratio = load / cap_v
        remain_ratio = torch.clamp(1.0 - load_ratio, min=0.0)

        return {
            "load": load,
            "count": count,
            "centroid": centroid,
            "mean_dir": mean_dir,
            "radius": radius,
            "load_ratio": load_ratio,
            "remain_ratio": remain_ratio,
            "client_count_ratio": client_count_ratio,
            "cap_v": cap_v,
        }

    def _build_edge_dyn(self, src, dst, node_xy, node_unit, demand, state):
        dist_centroid = (node_xy[dst] - state["centroid"][src]).norm(dim=-1)
        cosang = (node_unit[dst] * state["mean_dir"][src]).sum(dim=-1).clamp(-1.0, 1.0)
        angle_diff = 1.0 - cosang
        load_after = (state["load"][src] + demand[dst]) / state["cap_v"][src].clamp_min(1e-6)
        overload = torch.relu(load_after - 1.0)
        radius_inc = torch.relu(dist_centroid - state["radius"][src])
        return torch.stack([dist_centroid, angle_diff, load_after, overload, radius_inc], dim=-1)

    def _refresh_dynamic_context(
        self,
        refresh_idx: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_batch: torch.Tensor,
        veh_batch: torch.Tensor,
        node_xy: torch.Tensor,
        node_unit: torch.Tensor,
        demand: torch.Tensor,
        capacity_b: torch.Tensor,
        h_v_in: torch.Tensor,
        h_n_in: torch.Tensor,
        h_e_in: torch.Tensor,
    ):
        score_in = torch.cat([h_v_in[src], h_n_in[dst], h_e_in], dim=-1)
        score_l = self.pre_score[refresh_idx](score_in).squeeze(-1)
        p = pyg_softmax(score_l, index=dst)

        veh_state = self._build_vehicle_state(
            src=src,
            dst=dst,
            node_batch=node_batch,
            veh_batch=veh_batch,
            node_xy=node_xy,
            node_unit=node_unit,
            demand=demand,
            capacity_b=capacity_b,
            p=p,
        )
        edge_dyn_raw = self._build_edge_dyn(src, dst, node_xy, node_unit, demand, veh_state)
        edge_dyn_h = self.edge_dyn_proj(edge_dyn_raw)
        edge_bias = self.edge_bias_mlp(edge_dyn_raw)
        return veh_state, edge_dyn_raw, edge_dyn_h, edge_bias

    def forward(self, graph, xt_edge: torch.Tensor, t: torch.Tensor):
        device = graph.node_features.device

        # The diffusion edge state is assumed to be already in [0, 1].
        # Do not infer the range with xt_edge.min().item(), because that creates
        # a GPU -> CPU synchronization point in every forward pass.
        xt01 = xt_edge.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        t = t.long().to(device).view(-1)

        node_batch = graph.node_batch.long().to(device)
        veh_batch = graph.veh_batch.long().to(device)
        edge_index = graph.edge_index.long().to(device)
        src, dst = edge_index[0], edge_index[1]
        edge_graph = node_batch[dst]

        B = int(t.numel()) if t.numel() > 0 else 1

        t_emb = self.time_proj(self.time_emb(t))

        h_n = self.node_proj(graph.node_features.float())
        h_v = self.veh_proj(graph.veh_features.float())
        h_e_static = self.edge_proj(torch.cat([graph.edge_attr.float(), xt01.view(-1, 1)], dim=-1))

        if hasattr(graph, "graph_feat") and graph.graph_feat is not None:
            graph_feat = graph.graph_feat.float().to(device)
        else:
            capacity_b = graph.capacity.float().view(-1).to(device)
            if capacity_b.numel() == 1 and B > 1:
                capacity_b = capacity_b.repeat(B)

            total_dem = (
                    self._scatter_add_1d(
                        node_batch,
                        graph.demand_linehaul.float().to(device),
                        B,
                    )
                    / capacity_b.clamp_min(1e-6)
            )

            node_cnt = torch.bincount(node_batch, minlength=B).float().to(device).clamp_min(1.0)

            veh_cnt = torch.bincount(veh_batch, minlength=B).float().to(device).clamp_min(1.0)
            if hasattr(graph, "K_max") and graph.K_max is not None:
                K_graph = graph.K_max.view(-1).float().to(device)
                if K_graph.numel() == 1 and B > 1:
                    K_graph = K_graph.repeat(B)
                K_graph = torch.minimum(K_graph[:B].clamp_min(1.0), veh_cnt)
            else:
                K_graph = veh_cnt

            cap_lb = torch.ceil(total_dem).clamp_min(1.0)
            cap_lb_ratio = cap_lb / K_graph.clamp_min(1.0)
            slot_density = K_graph / torch.sqrt(node_cnt.clamp_min(1.0))

            graph_feat = torch.stack(
                [
                    graph.depot_xy[:, 0, 0],
                    graph.depot_xy[:, 0, 1],
                    total_dem,
                    total_dem / node_cnt,
                    cap_lb_ratio,
                    slot_density,
                ],
                dim=-1,
            )

        h_g = self.global_proj(graph_feat)

        node_xy = graph.node_features[:, 0:2].float().to(device)
        node_r = graph.node_features[:, 2].float().to(device).clamp_min(1e-6)
        node_unit = node_xy / node_r.unsqueeze(-1)

        demand = graph.demand_linehaul.float().to(device)
        capacity_b = graph.capacity.float().view(-1).to(device)
        if capacity_b.numel() == 1 and B > 1:
            capacity_b = capacity_b.repeat(B)

        if hasattr(graph, "node_knn_edge_index") and graph.node_knn_edge_index is not None:
            nn_edge_index = graph.node_knn_edge_index.long().to(device)
            nn_edge_attr = graph.node_knn_edge_attr.float().to(device)
        else:
            nn_edge_index = self._build_knn_edges_by_batch(
                node_xy=node_xy,
                batch_ids=node_batch,
                k=self.n2n_knn_k,
                B=B,
            )
            nn_edge_attr = self._n2n_edge_attr(node_xy, demand, nn_edge_index)

        rev_edge_index = torch.stack([dst, src], dim=0)

        # Static prior from the diffusion state itself: cheap, always available.
        p0 = self._row_normalize_by_dst(
            xt01.view(-1),
            dst,
            num_nodes=graph.node_features.size(0),
        )
        veh_state_static = self._build_vehicle_state(
            src=src,
            dst=dst,
            node_batch=node_batch,
            veh_batch=veh_batch,
            node_xy=node_xy,
            node_unit=node_unit,
            demand=demand,
            capacity_b=capacity_b,
            p=p0,
        )
        edge_dyn_static_raw = self._build_edge_dyn(src, dst, node_xy, node_unit, demand, veh_state_static)
        edge_dyn_static_h = self.edge_dyn_proj(edge_dyn_static_raw)
        edge_bias_static = self.edge_bias_mlp(edge_dyn_static_raw)

        cached_veh_state = veh_state_static
        cached_edge_dyn_raw = edge_dyn_static_raw
        cached_edge_dyn_h = edge_dyn_static_h
        cached_edge_bias = edge_bias_static

        for l in range(self.n_layers):
            cond = t_emb + h_g

            if self.use_adaln:
                ns, nb, vs, vb, es, eb, gn, gv = self.adaLN[l](cond).chunk(8, dim=-1)
                h_n_in = h_n * (1.0 + ns[node_batch]) + nb[node_batch]
                h_v_in = h_v * (1.0 + vs[veh_batch]) + vb[veh_batch]
                h_e_in = h_e_static * (1.0 + es[edge_graph]) + eb[edge_graph]
                gate_n = torch.sigmoid(gn)
                gate_v = torch.sigmoid(gv)
            else:
                h_n_in, h_v_in, h_e_in = h_n, h_v, h_e_static
                gate_n = torch.ones_like(h_g)
                gate_v = torch.ones_like(h_g)

            intra_idx = self.intra_layer_to_idx.get(l, None)
            if intra_idx is not None:
                n2n_h, _ = self.n2n[intra_idx](h_n_in, nn_edge_index, nn_edge_attr)
                h_n = self.norm_n_intra[l](h_n + F.dropout(n2n_h - h_n_in, p=self.dropout, training=self.training))
                h_n_in = h_n

            refresh_idx = self.refresh_layer_to_idx.get(l, None)
            if refresh_idx is not None:
                cached_veh_state, cached_edge_dyn_raw, cached_edge_dyn_h, cached_edge_bias = self._refresh_dynamic_context(
                    refresh_idx,
                    src,
                    dst,
                    node_batch,
                    veh_batch,
                    node_xy,
                    node_unit,
                    demand,
                    capacity_b,
                    h_v_in,
                    h_n_in,
                    h_e_in,
                )

            edge_msg = self.edge_msg_norm[l](h_e_in + cached_edge_dyn_h)
            edge_bias_delta = self.edge_delta_bias[l](
                torch.cat(
                    [
                        h_v_in[src],
                        h_n_in[dst],
                        h_g[edge_graph],
                        cached_edge_dyn_raw,
                        xt01.view(-1, 1),
                    ],
                    dim=-1,
                )
            )
            total_edge_bias = edge_bias_static + cached_edge_bias + edge_bias_delta

            v2n_out = self.v2n[l](h_v_in, edge_index, edge_msg, h_n_in, edge_logit_bias=total_edge_bias)
            n2v_out = self.n2v[l](h_n_in, rev_edge_index, edge_msg, h_v_in, edge_logit_bias=total_edge_bias)

            h_n = self.norm_n_cross[l](
                h_n + gate_n[node_batch] * F.dropout(v2n_out, p=self.dropout, training=self.training)
            )
            h_v = self.norm_v_cross[l](
                h_v + gate_v[veh_batch] * F.dropout(n2v_out, p=self.dropout, training=self.training)
            )

            # Optional v2v slot self-attention.
            # This is graph-local attention over vehicle slots:
            #   h_v: [B*K, H] -> [B, K, H] -> attention over K -> [B*K, H]
            # No per-graph loop is used.
            v2v_idx = self.v2v_layer_to_idx.get(l, None)
            if v2v_idx is not None:
                h_v = self.v2v[v2v_idx](h_v, B)

            veh_mean = self._batch_mean(h_v, veh_batch, B)
            vg_out = self.vehicle_global[l](torch.cat([h_v, h_g[veh_batch], veh_mean[veh_batch]], dim=-1))
            vg_delta, vg_gate = vg_out.chunk(2, dim=-1)
            h_v = self.norm_v_global[l](h_v + torch.sigmoid(vg_gate) * F.dropout(vg_delta, p=self.dropout, training=self.training))

            if self.use_global:
                h_g = h_g + self.global_update[l](
                    torch.cat(
                        [
                            h_g,
                            self._batch_mean(h_n, node_batch, B),
                            self._batch_mean(h_v, veh_batch, B),
                        ],
                        dim=-1,
                    )
                )

        h_e_final = self.edge_out_norm(h_e_static + cached_edge_dyn_h)
        edge_feat = torch.cat(
            [
                h_v[src],
                h_n[dst],
                h_e_final,
                h_v[src] * h_n[dst],
                h_v[src] * h_e_final,
                h_n[dst] * h_e_final,
                h_g[edge_graph],
            ],
            dim=-1,
        )
        logit = self.edge_head(edge_feat).squeeze(-1)
        return logit


# Backward-compatible aliases.
EdgeBipartiteDenoiserV3 = EdgeBipartiteDenoiserV4
EdgeBipartiteDenoiser = EdgeBipartiteDenoiserV4
