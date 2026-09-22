"""Model definitions for the Fundamental_SPP reproduction (workstream 3).

All models map (batch, L=252, F) -> (batch,) scores. The GNN additionally takes
``edge_index``. All models target ~0.2M parameters. Pure PyTorch, CPU only.

Panel / data conventions live in train.py (see its module docstring).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 252
D_MODEL = 64
D_STATE = 16

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _assoc_scan(a: torch.Tensor, c: torch.Tensor, micro_batch: int = 32) -> torch.Tensor:
    """Parallel associative (Hillis-Steele) scan for the linear recurrence
    ``h[t] = a[t] * h[t-1] + c[t]`` with ``h[-1] = 0``.

    ``a``, ``c``: (B, L, D). Returns ``h``: (B, L, D). The batch dimension is
    processed in micro-batches to bound peak memory (the flattened state is
    (B, L, d_inner * d_state) which is large for B=256).
    """
    B, L, D = a.shape
    outs = []
    for b0 in range(0, B, micro_batch):
        aa = a[b0:b0 + micro_batch]
        cc = c[b0:b0 + micro_batch]
        step = 1
        while step < L:
            a_l, c_l = aa[:, :-step], cc[:, :-step]
            a_r, c_r = aa[:, step:], cc[:, step:]
            # (p, q) <- (p, q) ⊗ (p_prev, q_prev), identity on first `step` rows
            aa = torch.cat([aa[:, :step], a_r * a_l], dim=1)
            cc = torch.cat([cc[:, :step], a_r * c_l + c_r], dim=1)
            step *= 2
        outs.append(cc)
    return torch.cat(outs, dim=0)


def _sequential_scan(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reference sequential scan (used only in validation tests)."""
    B, L, D = a.shape
    h = torch.zeros(B, D, dtype=c.dtype)
    outs = []
    for t in range(L):
        h = a[:, t] * h + c[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


@torch.jit.script
def _scan_forward(dA: torch.Tensor, delta: torch.Tensor, Bm: torch.Tensor,
                  Cm: torch.Tensor, xh: torch.Tensor,
                  D: torch.Tensor) -> torch.Tensor:
    """Scripted selective-scan forward: h_t = dA_t*h_{t-1} + delta_t*B_t*x_t."""
    B, L, di = xh.shape
    n = Bm.shape[2]
    h = torch.zeros((B, di, n), dtype=xh.dtype, device=xh.device)
    y = torch.empty((B, L, di), dtype=xh.dtype, device=xh.device)
    for t in range(L):
        h = (dA[:, t].unsqueeze(-1) * h
             + delta[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)
             * xh[:, t].unsqueeze(-1))
        y[:, t] = (Cm[:, t].unsqueeze(1) * h).sum(-1) + D * xh[:, t]
    return y


@torch.jit.script
def _scan_backward(dA: torch.Tensor, delta: torch.Tensor, Bm: torch.Tensor,
                   Cm: torch.Tensor, xh: torch.Tensor, D: torch.Tensor,
                   gy: torch.Tensor):
    """Manual adjoint backward for the selective scan.

    Recomputes hidden states (no grad), then runs the reverse recurrence
    dh[t] = C[t]^T gy[t] + dA[t+1] dh[t+1]. Returns
    (gdA, gdelta, gBm, gCm, gxh, gD).
    """
    B, L, di = xh.shape
    n = Bm.shape[2]
    h_all = torch.empty((B, L, di, n), dtype=xh.dtype, device=xh.device)
    h = torch.zeros((B, di, n), dtype=xh.dtype, device=xh.device)
    for t in range(L):
        h = (dA[:, t].unsqueeze(-1) * h
             + delta[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)
             * xh[:, t].unsqueeze(-1))
        h_all[:, t] = h
    gdA = torch.empty_like(dA)
    gdelta = torch.empty_like(delta)
    gBm = torch.zeros_like(Bm)
    gCm = torch.zeros_like(Cm)
    gxh = torch.empty_like(xh)
    gD = torch.zeros_like(D)
    dh = torch.zeros((B, di, n), dtype=xh.dtype, device=xh.device)
    zero_h = torch.zeros((B, di, n), dtype=xh.dtype, device=xh.device)
    for t in range(L - 1, -1, -1):
        gy_t = gy[:, t]
        h_t = h_all[:, t]
        Cm_t = Cm[:, t]
        Bm_t = Bm[:, t]
        xh_t = xh[:, t]
        dl_t = delta[:, t]
        # dh currently = dA[t+1] * dh[t+1] (0 at t = L-1)
        dh_t = Cm_t.unsqueeze(1) * gy_t.unsqueeze(-1) + dh
        gCm[:, t] = (gy_t.unsqueeze(-1) * h_t).sum(1)
        gD = gD + (gy_t * xh_t).sum(0)
        gxh[:, t] = (gy_t * D
                     + (dh_t * dl_t.unsqueeze(-1)
                        * Bm_t.unsqueeze(1)).sum(-1))
        gdelta[:, t] = (dh_t * Bm_t.unsqueeze(1)
                        * xh_t.unsqueeze(-1)).sum(-1)
        gBm[:, t] = (dh_t * dl_t.unsqueeze(-1)
                     * xh_t.unsqueeze(-1)).sum(1)
        if t > 0:
            h_prev = h_all[:, t - 1]
        else:
            h_prev = zero_h
        gdA[:, t] = (dh_t * h_prev).sum(-1)
        dh = dA[:, t].unsqueeze(-1) * dh_t
    return gdA, gdelta, gBm, gCm, gxh, gD


class _SelectiveScanFn(torch.autograd.Function):
    """Memory-light selective scan: scripted forward, manual adjoint backward.

    Avoids autograd tracing the 252-step loop (which stores ~GBs of
    intermediates); backward recomputes states once and runs one reverse pass.
    The backward is chunked over the batch (128) to bound the peak
    (B, L, d_inner, d_state) state tensor.
    """

    @staticmethod
    def forward(ctx, dA, delta, Bm, Cm, xh, D):
        ctx.save_for_backward(dA, delta, Bm, Cm, xh, D)
        return _scan_forward(dA, delta, Bm, Cm, xh, D)

    @staticmethod
    def backward(ctx, gy):
        dA, delta, Bm, Cm, xh, D = ctx.saved_tensors
        B = xh.shape[0]
        chunk, outs = 128, []
        # accumulate grads per chunk to bound peak memory
        gdA = torch.zeros_like(dA)
        gdelta = torch.zeros_like(delta)
        gBm = torch.zeros_like(Bm)
        gCm = torch.zeros_like(Cm)
        gxh = torch.zeros_like(xh)
        gD = torch.zeros_like(D)
        for b0 in range(0, B, chunk):
            sl = slice(b0, b0 + chunk)
            r = _scan_backward(dA[sl].contiguous(), delta[sl].contiguous(),
                               Bm[sl].contiguous(), Cm[sl].contiguous(),
                               xh[sl].contiguous(), D, gy[sl].contiguous())
            gdA[sl], gdelta[sl], gBm[sl], gCm[sl], gxh[sl] = r[:5]
            gD += r[5]
        return gdA, gdelta, gBm, gCm, gxh, gD


def _ssm_apply(dA: torch.Tensor, delta: torch.Tensor, Bm: torch.Tensor,
               Cm: torch.Tensor, xh: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Selective scan + output projection for a full batch (autograd-safe)."""
    return _SelectiveScanFn.apply(dA, delta, Bm, Cm, xh, D)


# ---------------------------------------------------------------------------
# MISS: selective state-space model (pure PyTorch, no mamba-ssm)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM block.

    Per timestep t (x_t in R^d_inner):
        Δ_t = softplus(W_Δ x_t)            (input-dependent step size)
        B_t = W_B x_t,  C_t = W_C x_t      (input-dependent matrices)
        A   = -exp(A_log)                  (learned, diagonal, stable)
        dA_t = exp(Δ_t * A)                (ZOH discretization)
        dB_t = Δ_t * B_t                   (first-order approx of ZOH)
        h_t  = dA_t * h_{t-1} + dB_t * x_t
        y_t  = C_t · h_t + D * x_t
    Output is gated (SiLU) and projected back to d_model.
    """

    def __init__(self, d_model: int = D_MODEL, d_inner: int = 192,
                 d_state: int = D_STATE, conv_kernel: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                              groups=d_inner, padding=conv_kernel - 1)
        self.act = nn.SiLU()
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)
        self.A_log = nn.Parameter(torch.randn(d_inner))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)
        xh, gate = xz.chunk(2, dim=-1)                      # (B, L, d_inner) each
        # causal depthwise conv: output[t] sees inputs [t-k+1 .. t]
        xc = self.conv(xh.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xh = self.act(xc)
        delta = F.softplus(self.dt_proj(xh))                # (B, L, d_inner)
        Bm = self.B_proj(xh)                                # (B, L, d_state)
        Cm = self.C_proj(xh)                                # (B, L, d_state)
        A = -torch.exp(self.A_log)                          # (d_inner,)
        dA = torch.exp(delta * A)                           # (B, L, d_inner)
        y = _ssm_apply(dA, delta, Bm, Cm, xh, self.D)        # (B, L, d_inner)
        y = y * self.act(gate)
        return self.out_proj(y)                             # (B, L, d_model)


class MISSBlock(nn.Module):
    """One MISS block: selective SSM -> residual Linear proj -> LayerNorm."""

    def __init__(self, d_model: int = D_MODEL, d_inner: int = 192,
                 d_state: int = D_STATE):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_inner, d_state)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.proj(self.ssm(x)))


class MISS(nn.Module):
    """MISS: Linear(F->64) -> 2 x MISSBlock -> Linear(64->1) on last step."""

    def __init__(self, in_dim: int, d_model: int = D_MODEL, d_inner: int = 192,
                 d_state: int = D_STATE, n_blocks: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [MISSBlock(d_model, d_inner, d_state) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B,)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """2-layer LSTM (hidden 128) -> Linear head on last step."""

    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# StockMixer (Fan & Shen style)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class StockMixerBlock(nn.Module):
    """One block with three mixers (each: residual MLP + LayerNorm).

    Input (B_stocks, L, F):
      - indicator mixer: MLP across the feature dim F (per stock, per time)
      - time mixer:      MLP across the time dim L (per stock, per feature)
      - stock mixer:     MLP across the stock dim, in fixed groups of G stocks
                         (batch padded to a multiple of G).
    """

    def __init__(self, n_features: int, seq_len: int = SEQ_LEN,
                 group: int = 32, hidden_mult: int = 4, time_hidden: int = 96):
        super().__init__()
        self.group = group
        self.ind_mixer = _MLP(n_features, n_features * hidden_mult)
        self.ind_norm = nn.LayerNorm(n_features)
        self.time_mixer = _MLP(seq_len, time_hidden)
        self.time_norm = nn.LayerNorm(seq_len)
        self.stock_mixer = _MLP(group, group * 2)
        self.stock_norm = nn.LayerNorm(group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        B, L, n_feat = x.shape
        x = self.ind_norm(x + self.ind_mixer(x))                    # over F
        xt = x.transpose(1, 2)                                      # (B, F, L)
        xt = self.time_norm(xt + self.time_mixer(xt))              # over L
        x = xt.transpose(1, 2)                                      # (B, L, F)
        G = self.group
        pad = (-B) % G
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        Bp = x.shape[0]
        xs = x.view(Bp // G, G, L, n_feat).permute(0, 2, 3, 1)    # (nG, L, F, G)
        xs = self.stock_norm(xs + self.stock_mixer(xs))           # over G
        x = xs.permute(0, 3, 1, 2).reshape(Bp, L, n_feat)
        if pad:
            x = x[:B]
        return x


class StockMixer(nn.Module):
    """Stacked mixer blocks -> temporal mean pool -> MLP head -> score."""

    def __init__(self, in_dim: int, seq_len: int = SEQ_LEN, n_blocks: int = 4,
                 group: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StockMixerBlock(in_dim, seq_len, group) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_stocks, L, F) -> (B_stocks,)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)                               # (B, F)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# GNN: per-stock temporal encoder + message passing
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Mean-aggregation message passing: h_i' = relu(W_s h_i + W_n mean_{j->i} h_j)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, dim); edge_index: (2, E)
        N = x.shape[0]
        if edge_index.numel() == 0:
            return F.relu(self.lin_self(x))
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = torch.zeros(N, 1, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp(min=1.0)
        return F.relu(self.lin_self(x) + self.lin_neigh(agg))


class StockGNN(nn.Module):
    """Per-stock 2-layer GRU encoder over L=252 -> 2 message-passing layers -> head."""

    def __init__(self, in_dim: int, hidden: int = 112):
        super().__init__()
        self.encoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True)
        self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.conv1 = GraphConv(hidden)
        self.conv2 = GraphConv(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 56), nn.ReLU(),
                                  nn.Linear(56, 1))

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor = None) -> torch.Tensor:
        # x: (N_stocks, L, F) -> (N_stocks,)
        _, h = self.encoder(x)                                     # (2, N, hidden)
        z = self.post(h[-1])
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        z = self.conv1(z, edge_index)
        z = self.conv2(z, edge_index)
        return self.head(z).squeeze(-1)


def build_graph(sectors, rets, top_k: int = 5) -> torch.Tensor:
    """Build an undirected stock graph.

    Edges from (a) same GICS sector membership (fully connected within sector)
    and (b) rolling-63d return-correlation top-k peers per stock.

    Args:
        sectors: (N,) array-like of sector ids/labels.
        rets: (N, W) array-like of daily returns over the trailing window
            (NaNs allowed; treated as 0 after z-scoring).
        top_k: number of correlation peers per stock.

    Returns:
        edge_index: (2, E) LongTensor, deduplicated, with both directions.
    """
    sectors = np.asarray(sectors)
    rets = np.nan_to_num(np.asarray(rets, dtype=np.float64), nan=0.0)
    N = len(sectors)
    # (a) sector membership edges: all ordered pairs within a sector
    same = sectors[:, None] == sectors[None, :]
    np.fill_diagonal(same, False)
    src_s, dst_s = np.nonzero(same)
    src = src_s.tolist()
    dst = dst_s.tolist()
    if N > 1 and rets.shape[1] > 1:
        mu = rets.mean(axis=1, keepdims=True)
        sd = rets.std(axis=1, keepdims=True) + 1e-8
        z = (rets - mu) / sd
        corr = (z @ z.T) / rets.shape[1]
        np.fill_diagonal(corr, -np.inf)
        k = min(top_k, N - 1)
        top = np.argpartition(corr, -k, axis=1)[:, -k:]
        for i in range(N):
            for j in top[i]:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    if edge_index.shape[1]:
        edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def build_model(name: str, in_dim: int) -> nn.Module:
    name = name.lower()
    if name == 'miss':
        return MISS(in_dim)
    if name == 'lstm':
        return LSTMPredictor(in_dim)
    if name in ('stockmixer', 'stock_mixer', 'mixer'):
        return StockMixer(in_dim)
    if name == 'gnn':
        return StockGNN(in_dim)
    raise ValueError(f'unknown model {name!r}; choose from miss/lstm/stockmixer/gnn')


if __name__ == '__main__':
    for n_feat in (24, 15):
        print(f'--- in_dim F={n_feat} ---')
        for name in ('miss', 'lstm', 'stockmixer', 'gnn'):
            m = build_model(name, n_feat)
            n = count_parameters(m)
            flag = 'OK' if 50_000 <= n <= 500_000 else 'OUT OF RANGE'
            print(f'{name:12s} params={n:,}  (~{n/1e6:.3f}M)  [{flag}]')
            # smoke-test forward
            x = torch.randn(8, SEQ_LEN, n_feat)
            if name == 'gnn':
                ei = build_graph(np.random.randint(0, 3, 8), np.random.randn(8, 63))
                y = m(x, ei)
            else:
                y = m(x)
            assert y.shape == (8,), (name, y.shape)
        print('forward smoke test passed')
