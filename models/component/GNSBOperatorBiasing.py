import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_GNSB_SHARED = {}

def _ensure_rw_kernel(A, eps=1e-6):
    """
    把任意非负邻接矩阵 A 变成随机游走核（row-stochastic），并强制每行有自环。

    重要：始终返回 (B,K,K)，即使 B=1 也不 squeeze。
    这样后续 graph distance / batch statistics 不会因为 batch=1 变成 2D 而出错。
    """
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if A.ndim != 3:
        raise RuntimeError(f"_ensure_rw_kernel expects (K,K) or (B,K,K), got shape={tuple(A.shape)}")

    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, _ = A.shape
    I = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    A = A.clamp_min(0) + eps * I
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
    return A
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,1) in [0,1]
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device))
    angles = t * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant", value=0.0)
    return emb

class LatentBaryPosBias(nn.Module):
    """
    LBPB++：动态质心 + 小幅 per-head 仿射偏移（首选方案）
    输入:
      attn_latent_token: (B,h,K,N) —— 常用 P 作为 h=1 的注意力
      H, W: 空间尺寸
    输出:
      bias: (B,h,N,K) —— 回投影前，对列(像素)的 K 分配做加性偏置
    """
    def __init__(self, num_heads: int = 1, num_scales: int = 2, max_offset: float = 0.03):
        super().__init__()
        self.h = int(num_heads)
        self.num_scales = int(num_scales)
        self.max_offset = float(max_offset)

        self.alpha   = nn.Parameter(torch.ones(self.h, self.num_scales))         # 几何核权重
        self.log_sig = nn.Parameter(torch.full((self.h, self.num_scales), math.log(0.15)))  # σ≈0.15
        self.beta    = nn.Parameter(torch.zeros(self.h, self.num_scales))        # 频域权重（初期 0）
        self.freq_x  = nn.Parameter(torch.full((self.h, self.num_scales), 2.0))  # x 频率
        self.freq_y  = nn.Parameter(torch.full((self.h, self.num_scales), 2.0))  # y 频率

        self.center_W = nn.Parameter(torch.zeros(self.h, 2, 2))
        self.center_b = nn.Parameter(torch.zeros(self.h, 2))
        with torch.no_grad():
            for hh in range(self.h):
                self.center_W[hh].copy_(torch.eye(2) * 0.05)
            self.center_b.zero_()

    @staticmethod
    def _make_coords(H, W, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        return torch.stack([yy, xx], dim=-1).reshape(H * W, 2)  # (N,2), (y,x)

    def forward(self, attn_latent_token: torch.Tensor, H: int, W: int):
        B, h, K, N = attn_latent_token.shape
        assert h == self.h, f"h mismatch: got {h}, expected {self.h}"
        device, dtype = attn_latent_token.device, attn_latent_token.dtype

        Pxy = self._make_coords(H, W, device, dtype)

        w = attn_latent_token.clamp_min(1e-9)
        w_sum = w.sum(dim=-1, keepdim=True)                             # (B,h,K,1)
        centers = torch.einsum('bhkn,nc->bhkc', w, Pxy) / w_sum         # (B,h,K,2)

        c4 = centers.view(B, h, K, 1, 2)                                 # (B,h,K,1,2)
        W = self.center_W.view(1, h, 1, 2, 2)                            # (1,h,1,2,2)
        b = self.center_b.view(1, h, 1, 1, 2)                            # (1,h,1,1,2)
        delta = torch.matmul(c4, W).squeeze(3) + b.squeeze(3)            # (B,h,K,2)
        delta = torch.tanh(delta) * self.max_offset
        centers = (centers + delta).clamp_(0.0, 1.0)                     # 限制到 [0,1]

        PN = Pxy.view(1, 1, N, 1, 2)
        CK = centers.view(B, h, 1, K, 2)
        dxy = PN - CK
        dx, dy = dxy[..., 0], dxy[..., 1]
        dist = (dx.square() + dy.square()).sqrt().clamp_min(1e-6)

        bias_total = 0.0
        for s in range(self.num_scales):
            sigma = self.log_sig[:, s].exp().view(1, self.h, 1, 1) + 1e-6
            alpha = self.alpha[:, s].view(1, self.h, 1, 1)
            beta  = self.beta[:, s].view(1, self.h, 1, 1)
            fx    = self.freq_x[:, s].view(1, self.h, 1, 1)
            fy    = self.freq_y[:, s].view(1, self.h, 1, 1)

            k_geo  = torch.exp(-dist / sigma)                            # (B,h,N,K)
            k_four = torch.cos(2*math.pi*fx*dx) + torch.cos(2*math.pi*fy*dy)

            bias_total = bias_total + alpha * (k_geo + beta * k_four)

        return bias_total.to(dtype)  # (B,h,N,K)

    
class ParamFreePosBias:
    """
    将 RPE / RoPE / ALiBi 以 “logit 加性偏置” 的形式，统一作用在 P 上。
    所有模式均为“无可学习参数”，仅使用固定公式与网格/质心坐标。
    用法：
      bias = ParamFreePosBias.compute(P=(B,K,N), H, W, mode='alibi'|'rpe'|'rope')
      返回 bias: (B,N,K)
    """
    @staticmethod
    def _coords(H, W, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        return torch.stack([yy, xx], dim=-1).reshape(H*W, 2)  # (N,2)

    @staticmethod
    def _centers_from_P(P, H, W):
        B, K, N = P.shape
        coords = ParamFreePosBias._coords(H, W, P.device, P.dtype)  # (N,2)
        w = P.clamp_min(1e-9)
        c = torch.einsum('bkn,nc->bkc', w, coords) / w.sum(-1, keepdim=True)
        return c  # (B,K,2)

    @staticmethod
    def _pairwise_dxdy(centers, H, W):
        B, K, _ = centers.shape
        coords = ParamFreePosBias._coords(H, W, centers.device, centers.dtype)  # (N,2)
        PN = coords.view(1, 1, -1, 2)           # (1,1,N,2)
        CK = centers.view(B, K, 1, 2)           # (B,K,1,2)
        dxy = PN - CK                            # (B,K,N,2) = (y,x) difference
        dy = dxy[..., 0]                         # (B,K,N)
        dx = dxy[..., 1]                         # (B,K,N)
        return dx, dy

    @staticmethod
    def compute(P, H, W, mode: str = "alibi"):
        """
        返回 bias: (B,N,K)
        mode:
          'alibi' :  - L1 距离（线性衰减）
          'rpe'   :  - 高斯核（固定 σ），纯相对位移函数
          'rope'  :  - 固定频率的 2D Fourier 特征点积（RoPE 等价的无参相位对齐）
        """
        assert mode in ("alibi", "rpe", "rope")
        B, K, N = P.shape
        centers = ParamFreePosBias._centers_from_P(P, H, W)   # (B,K,2)
        dx, dy = ParamFreePosBias._pairwise_dxdy(centers, H, W)  # (B,K,N)

        if mode == "alibi":
            bias_bkn = -(dx.abs() + dy.abs())                 # (B,K,N)

        elif mode == "rpe":
            sigma = 0.15
            dist2 = dx.square() + dy.square()                 # (B,K,N)
            bias_bkn = torch.exp(-dist2 / (2.0 * sigma * sigma))

        else:  # 'rope'
            F = 16  # 频带数（固定）
            freqs = torch.logspace(0.0, math.log10(1000.0), steps=F, device=P.device, dtype=P.dtype)
            coords = ParamFreePosBias._coords(H, W, P.device, P.dtype)   # (N,2)
            y, x = coords[:, 0:1], coords[:, 1:2]                       # (N,1)
            cy, cx = centers[..., 0], centers[..., 1]                    # (B,K)
            ph_y = torch.cat([torch.sin(2*math.pi*y*freqs), torch.cos(2*math.pi*y*freqs)], dim=-1)  # (N,2F)
            ph_x = torch.cat([torch.sin(2*math.pi*x*freqs), torch.cos(2*math.pi*x*freqs)], dim=-1)  # (N,2F)
            phi_p = torch.cat([ph_y, ph_x], dim=-1)   # (N,4F)
            ch_y = torch.cat([torch.sin(2*math.pi*cy.unsqueeze(-1)*freqs), torch.cos(2*math.pi*cy.unsqueeze(-1)*freqs)], dim=-1)  # (B,K,2F)
            ch_x = torch.cat([torch.sin(2*math.pi*cx.unsqueeze(-1)*freqs), torch.cos(2*math.pi*cx.unsqueeze(-1)*freqs)], dim=-1)  # (B,K,2F)
            phi_c = torch.cat([ch_y, ch_x], dim=-1)   # (B,K,4F)
            bias_bkn = torch.einsum('bkd,nd->bkn', phi_c, phi_p) / math.sqrt(float(phi_p.shape[-1]) + 1e-6)

        return bias_bkn.transpose(1, 2).contiguous()



def attn_diversity_regularizers(P: torch.Tensor, queries: torch.Tensor,
                                reg_q: float = 5e-3, reg_p: float = 5e-3) -> torch.Tensor:
    """
    多样性正则：
      - Q 正交：鼓励 Q Q^T ~ I，防止不同 query 学到同一方向
      - P 去相关：减少不同行（query）的覆盖重叠（off-diagonal of P_row P_row^T）
    P: (B,K,N)；queries: (K,d)
    返回: 标量 loss
    """
    Q = F.normalize(queries, dim=-1)                     # (K,d)
    I = torch.eye(Q.size(0), device=Q.device, dtype=Q.dtype)
    loss_q_orth = ((Q @ Q.t() - I) ** 2).mean()

    P_row = P / P.sum(-1, keepdim=True).clamp_min(1e-6) # (B,K,N) 行归一便于比较
    PPt = torch.einsum('bkn,bmn->bkm', P_row, P_row) / P_row.size(-1)  # (B,K,K)
    I_K = torch.eye(P.size(1), device=P.device, dtype=P.dtype)[None]
    off = PPt - I_K
    loss_p_deoverlap = (off ** 2).mean()

    return reg_q * loss_q_orth + reg_p * loss_p_deoverlap


class AttnAssignment(nn.Module):
    """
    Perceiver-IO 风格的防坍缩 cross-attn：
      - Latents (Kxd) 做 Query，输入 tokens(+Fourier 位置) 做 Key/Value
      - Softmax over N（每个 latent 在输入上分布），得到 P:(B,K,N)
      - 多头注意力、logits dropout、learnable temperature 抑制锐化坍缩
      - latent 内共享权重 Transformer block 迭代提炼
      - 熵正则：鼓励 cross-attn 覆盖度（可关/可调）
    """
    def __init__(self, in_dim: int, token_dim: int, K: int,
                 num_heads: int = 4, iters: int = 2,
                 attn_drop: float = 0.1, entropy_weight: float = 1e-4,
                 use_fourier_pos: bool = True, fourier_bands: int = 16):
        super().__init__()
        self.K = int(K)
        self.d = int(token_dim)
        self.h = int(num_heads)
        self.iters = int(iters)
        self.entropy_weight = float(entropy_weight)
        self.use_fourier_pos = bool(use_fourier_pos)
        self.fourier_bands = int(fourier_bands)
        self.pos_dim = 4 * self.fourier_bands

        self.proj_in = nn.Linear(in_dim, self.d, bias=False)
        self.ln_x = nn.LayerNorm(self.d)
        self.ln_lat = nn.LayerNorm(self.d)
        self.pos_proj = nn.Linear(self.pos_dim, self.d, bias=False) if self.use_fourier_pos else None

        self.queries = nn.Parameter(torch.randn(K, self.d) / math.sqrt(max(self.d, 1)))

        hd = self.d // self.h
        assert hd * self.h == self.d, "token_dim 必须能被 num_heads 整除"
        self.q_proj = nn.Linear(self.d, self.d, bias=False)
        self.k_proj = nn.Linear(self.d, self.d, bias=False)
        self.v_proj = nn.Linear(self.d, self.d, bias=False)
        self.o_proj = nn.Linear(self.d, self.d, bias=False)

        self.log_tau = nn.Parameter(torch.log(torch.tensor(1.5)))

        self.logits_drop = nn.Dropout(attn_drop)

        self.pre_ln1 = nn.LayerNorm(self.d)
        self.ff_qkv = nn.Sequential(
            nn.Linear(self.d, 4*self.d, bias=False), nn.GELU(),
            nn.Linear(4*self.d, self.d, bias=False)
        )
        self.pre_ln2 = nn.LayerNorm(self.d)
        self.ff_mlp = nn.Sequential(
            nn.Linear(self.d, 4*self.d, bias=False), nn.GELU(),
            nn.Linear(4*self.d, self.d, bias=False)
        )

        self.slot_log_sigma = nn.Parameter(torch.full((1, 1, self.d), -6.0))

        self._extra_losses = {}

    def extra_losses(self):
        return dict(self._extra_losses)

    def _fourier_pos(self, H, W, device, dtype):
        if not self.use_fourier_pos:
            return None
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        y, x = yy.reshape(-1,1), xx.reshape(-1,1)  # (N,1)
        freqs = torch.logspace(0.0, math.log10(1000.0), steps=self.fourier_bands, device=device, dtype=dtype)
        ph_y = torch.cat([torch.sin(2*math.pi*y*freqs), torch.cos(2*math.pi*y*freqs)], dim=-1)
        ph_x = torch.cat([torch.sin(2*math.pi*x*freqs), torch.cos(2*math.pi*x*freqs)], dim=-1)
        return torch.cat([ph_y, ph_x], dim=-1)  # (N, 4F)
    
    def _xattn(self, lat, tokens, mask=None):
        """
        lat:(B,K,d) as Q, tokens:(B,N,d) as K/V
        return: lat_upd:(B,K,d), attn_prob:(B,h,K,N)
        """
        B, K, d = lat.shape
        N = tokens.shape[1]
        hd = d // self.h

        wdtype = self.q_proj.weight.dtype
        lat_n = self.ln_lat(lat).to(wdtype)      # (B,K,d)
        tok_n = self.ln_x(tokens).to(wdtype)     # (B,N,d)

        Q  = self.q_proj(lat_n).view(B, K, self.h, hd).transpose(1, 2)   # (B,h,K,hd)
        Kk = self.k_proj(tok_n).view(B, N, self.h, hd).transpose(1, 2)   # (B,h,N,hd)
        V  = self.v_proj(tok_n).view(B, N, self.h, hd).transpose(1, 2)   # (B,h,N,hd)

        with torch.autocast(device_type='cuda', enabled=False):
            Qf, Kf, Vf = Q.float(), Kk.float(), V.float()
            scale = 1.0 / (math.sqrt(hd + 1e-6) * torch.exp(self.log_tau).clamp_min(0.25).float())
            logits = torch.einsum('bhkd,bhnd->bhkn', Qf, Kf) * scale
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
            logits = logits - logits.amax(dim=-1, keepdim=True)
            logits = logits.clamp(min=-40.0, max=20.0)
            logits = self.logits_drop(logits)
            A = torch.softmax(logits, dim=-1)                     # (B,h,K,N)

            lat_upd = torch.einsum('bhkn,bhnd->bhkd', A, Vf)      # (B,h,K,hd)
            lat_upd = lat_upd.transpose(1, 2).contiguous().view(B, K, d)

        lat_upd = lat_upd.to(self.o_proj.weight.dtype)
        lat_upd = self.o_proj(lat_upd).to(lat.dtype)
        A = A.to(lat.dtype)

        return lat_upd, A


    def forward(self, x):
        """
        x: (B,C,H,W)
        return:
          T: (B,K,d), P: (B,K,N), z: (B,N,d)
        """
        B, C, H, W = x.shape
        N = H * W
        z = x.flatten(2).transpose(1, 2)                 # (B,N,C)
        z = self.proj_in(z)                              # (B,N,d)
        pos = self._fourier_pos(H, W, z.device, z.dtype) # (N,4F) or None
        if pos is not None:
            pos_lin = self.pos_proj(pos.to(self.pos_proj.weight.dtype)).to(z.dtype)
            z = z + pos_lin[None, :, :]                  # (B,N,d)
        z = self.ln_x(z)
        z = torch.nan_to_num(z, 0.0, 1e4, -1e4)

        mu  = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B,K,d)
        if self.training:
            sig = self.slot_log_sigma.exp().expand(B, self.K, -1)
            lat = mu + torch.randn_like(mu) * sig
        else:
            lat = mu

        P_last = None
        for _ in range(self.iters):
            lat_upd, A = self._xattn(lat, z)     # A:(B,h,K,N)
            lat = lat + self.ff_qkv(self.pre_ln1(lat + lat_upd))  # 残差

            lat = lat + self.ff_mlp(self.pre_ln2(lat))

            P_last = A  # 记录最后一轮的注意力

        P = P_last.mean(dim=1)  # (B,K,N)
        T = self.ln_lat(lat)    # (B,K,d)

        self._extra_losses.clear()
        div_loss = attn_diversity_regularizers(P, self.queries, reg_q=5e-3, reg_p=5e-3)
        with torch.autocast(device_type='cuda', enabled=False):
            Pf = P.float().clamp_min(1e-9)
            Hk = -(Pf * Pf.log()).sum(dim=-1).mean()  # (B,K)->均值
            entropy_loss = (-self.entropy_weight) * Hk  # 负熵（鼓励高熵覆盖）
        self._extra_losses["loss_attn_div"] = div_loss + entropy_loss.to(T.dtype)

        T = torch.nan_to_num(T, 0.0, 1e4, -1e4)
        P = torch.nan_to_num(P, 0.0, 0.0, 0.0)

        return T.to(z.dtype), P.to(z.dtype), z


class LatentGraph(nn.Module):
    """
    构造 latent 图 A^{lat} ∈ R^{B×K×K}:
      - 基于 tokens 之间的余弦相似
      - top-k 稀疏 + 对称
      - 行归一化
    """
    def __init__(self, topk=8):
        super().__init__()
        self.topk = topk

    @torch.no_grad()
    def forward(self, T_src, T_tgt=None):
        """
        T_src: (B,K,d) ; T_tgt: (B,K,d) or None (默认用 T_src)
        """
        if T_tgt is None:
            T_tgt = T_src
        B, K, d = T_src.shape
        Ts = F.normalize(T_src, dim=-1, eps=1e-6)
        Tt = F.normalize(T_tgt, dim=-1, eps=1e-6)
        S = torch.einsum('bkd,bmd->bkm', Ts, Tt)  # (B,K,K) in [-1,1]

        eye_mask = torch.eye(K, device=T_src.device, dtype=torch.bool)[None]
        S = S.masked_fill(eye_mask, torch.finfo(S.dtype).min)

        k_eff = min(self.topk, max(K - 1, 1))
        topv, topi = torch.topk(S, k=k_eff, dim=-1)
        A = torch.zeros_like(S)
        A.scatter_(-1, topi, torch.sigmoid(topv))  # 平滑边权
        A = 0.5 * (A + A.transpose(-2, -1))
        deg = A.sum(-1, keepdim=True).clamp_min(1e-6)
        
        A = A / deg
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        return A


class LocalSubgraphRefiner(nn.Module):
    """
    局部子图精炼：对每个中心节点 i，取其 top-m 邻居 N_i（含自己），
    用邻接权重做 softmax 聚合，经过 MLP + 残差，重复 L 层。
    只更新“中心节点”的向量，避免K×K全图再次计算，稳定且高效。
    """
    def __init__(self, d, m_neighbors=8, num_layers=2, hidden=None, dropout=0.0):
        super().__init__()
        self.m = int(m_neighbors)
        self.L = int(num_layers)
        self.d = int(d)
        self.h = int(hidden) if hidden is not None else int(d)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d),
                nn.Linear(self.d, self.h, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.h, self.d, bias=False)
            ) for _ in range(self.L)
        ])

    @torch.no_grad()
    def _topk_idx(self, A):
        """
        A: (B,K,K) 行归一（或非负）；返回 idx: (B,K,m)
        确保自环在首位（中心节点自己），其余为top-k-1邻居。
        """
        B, K, _ = A.shape
        m = min(self.m, K)
        eye = torch.eye(K, device=A.device, dtype=torch.bool)[None]
        scores = A.masked_fill(eye, torch.finfo(A.dtype).min)
        _, nn_idx = torch.topk(scores, k=max(1, m-1), dim=-1)  # (B,K,m-1)
        self_idx = torch.arange(K, device=A.device)[None, :, None].expand(B, -1, 1)  # (B,K,1)
        idx = torch.cat([self_idx, nn_idx], dim=-1)  # (B,K,m)
        return idx

    def forward(self, T, A):
        """
        T: (B,K,d), A: (B,K,K) —— 用 A 挑选每个中心的 top-m 邻域并聚合
        return: T_refined: (B,K,d)
        """
        B, K, d = T.shape
        idx = self._topk_idx(A)                         # (B,K,m)
        w = torch.gather(A, dim=-1, index=idx).clamp_min(0)      # (B,K,m)
        w = torch.softmax(w, dim=-1).unsqueeze(-1)               # (B,K,m,1)

        T_exp = T.unsqueeze(1).expand(B, K, K, d)                # (B,K,K,d)
        m_eff = idx.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(B, K, m_eff, d)       # (B,K,m_eff,d)
        neigh = torch.gather(T_exp, dim=2, index=idx_exp)        # (B,K,m_eff,d)

        agg = (w * neigh).sum(dim=2)                             # (B,K,d)
        x = T
        for mlp in self.mlps:
            x = x + mlp(agg)
        return x


class GAT(nn.Module):
    def __init__(self, d, hidden=128, num_layers=2, time_dim=64, kl_rw_weight=1e-3):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.time_dim = time_dim

        self.mlp_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU())
        self.mlp_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))

        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden)
            ) for _ in range(num_layers)
        ])

        self.attn_q = nn.Linear(hidden, hidden, bias=False)
        self.attn_k = nn.Linear(hidden, hidden, bias=False)
        self.attn_v = nn.Linear(hidden, hidden, bias=False)

        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden * 2)  # -> gamma, beta
        )

        self.kl_rw_weight = float(kl_rw_weight)
        self._loss_buf = {}
        self._ref_A = None

    def extra_losses(self):
        return dict(self._loss_buf)

    def forward(self, T, A, te):
        if not torch.is_tensor(T) or T.ndim != 3:
            raise RuntimeError(f"GAT expects T as (B,K,d). Got {type(T)} with shape {getattr(T,'shape',None)}")
        B, K, d_in = T.shape

        if A.ndim == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)
        elif A.ndim == 3 and A.shape[0] == 1 and B > 1:
            A = A.expand(B, -1, -1)
        elif A.ndim != 3:
            raise RuntimeError(f"Adjacency must be (K,K) or (B,K,K), got ndim={A.ndim}")

        T  = T.to(dtype=self.mlp_in[1].weight.dtype, device=self.mlp_in[1].weight.device)
        A  = A.to(dtype=T.dtype, device=T.device)
        te = te.to(dtype=T.dtype, device=T.device)

        x = self.mlp_in(T)  # (B,K,H)
        H = x.shape[-1]
        tb = self.time_proj(te)                 # (B, 2H)
        gamma, beta = tb.chunk(2, dim=-1)       # (B,H), (B,H)
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        kl_total = x.new_tensor(0.0)

        for layer in self.gnn_layers:
            x_agg = torch.einsum('bkm,bmd->bkd', A, x)  # (B,K,H)

            q = self.attn_q(x).float()
            k = self.attn_k(x).float()
            v = self.attn_v(x).float()
            scores = torch.einsum('bkd,bmd->bkm', q, k) / math.sqrt(H + 1e-6)

            mask_bool = (A > 0)
            scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)
            scores = scores - scores.amax(dim=-1, keepdim=True)
            scores = scores.clamp(min=-40.0, max=20.0)

            attn = torch.softmax(scores, dim=-1)              # (B,K,K)

            bad = torch.isnan(attn).any(dim=-1) | (attn.sum(-1) <= 0)
            if bad.any():
                ref_rw = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
                attn = torch.where(bad[..., None], ref_rw, attn)

            if self.kl_rw_weight > 0.0:
                A_ref = getattr(self, "_ref_A", None)
                if A_ref is None:
                    A_ref = A
                if A_ref.ndim == 2:
                    A_ref = A_ref.unsqueeze(0).expand(B, -1, -1)
                elif A_ref.ndim == 3 and A_ref.shape[0] == 1 and B > 1:
                    A_ref = A_ref.expand(B, -1, -1)
                A_ref = A_ref.to(dtype=attn.dtype, device=attn.device)

                ref_rw  = A_ref / A_ref.sum(-1, keepdim=True).clamp_min(1e-6)
                attn_eps = attn.clamp_min(1e-9)
                ref_eps  = ref_rw.clamp_min(1e-9)
                kl = (attn_eps * (attn_eps.log() - ref_eps.log())).sum(dim=-1).mean()
                kl_total = kl_total + kl

            attn_out = torch.einsum('bkm,bmd->bkd', attn, v).to(x.dtype)
            x = x + layer(x_agg + attn_out)

        v = self.mlp_out(x)                  # (B,K,d)
        self._loss_buf.clear()
        if self.kl_rw_weight > 0.0:
            self._loss_buf["loss_sb_kl"] = kl_total * self.kl_rw_weight
        else:
            self._loss_buf["loss_sb_kl"] = v.new_tensor(0.0)

        if v.shape[:2] != T.shape[:2]:
            raise RuntimeError(f"[GAT] Shape drift: v={tuple(v.shape)} vs T={tuple(T.shape)}")

        v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
        return v


class LatentSBOnGraph(nn.Module):
    """
    在 latent 图上做桥接流：从 T0 -> T1
    - CFM 损失（可选）：在随机时间 tau 上，对 v_theta(T_tau, tau, A) 回归真向量场 v*=(T1-T0)
    """
    def __init__(self, d, gno_hidden=128, gno_layers=2,
                 T_steps=4, time_dim=64,
                 cfm_enable=False, cfm_loss_weight=0.0, bridge_mode="hybrid",
                 kl_rw_weight: float = 1e-3,
                 cfm_steps: int = None,
                 latent_cfm_noise_std: float = 0.0,
                 latent_cfm_drop_prob: float = 0.0):
        super().__init__()
        assert bridge_mode in ("hybrid", "cfm")
        self.T_steps = int(cfm_steps) if cfm_steps is not None else int(T_steps)
        if self.T_steps < 1:
            raise ValueError(
                f"cfm_steps/T_steps must be >= 1 for graph evolution, got {self.T_steps}. "
                f"For w/o CFM, use cfm_enable=False while keeping cfm_steps >= 1."
            )
        self.time_dim = int(time_dim)
        self.cfm_enable = bool(cfm_enable)
        self.cfm_loss_weight = float(cfm_loss_weight)
        self.bridge_mode = bridge_mode
        self.latent_cfm_noise_std = float(latent_cfm_noise_std)
        self.latent_cfm_drop_prob = float(latent_cfm_drop_prob)

        self.gat = GAT(d, hidden=gno_hidden, num_layers=gno_layers,
                       time_dim=time_dim, kl_rw_weight=kl_rw_weight)
        self._loss_buf = {}

    def extra_losses(self):
        return dict(self._loss_buf)

    def set_cfm_steps(self, steps: int):
        """Runtime helper for CFM/bridge step ablation without rebuilding the module."""
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"cfm_steps must be >= 1, got {steps}")
        self.T_steps = steps

    def _time_embed(self, tau, B, device):
        if not torch.is_tensor(tau):
            tau = torch.full((B, 1), float(tau), device=device)
        te = sinusoidal_time_embedding(tau, self.time_dim)  # (B, time_dim)

        if hasattr(self, "scale_embed"):
            sid = int(getattr(self, "current_scale", 0))
            sid_t = torch.full((B,), sid, device=device, dtype=torch.long)  # (B,)
            se = self.scale_embed(sid_t)                                    # (B, time_dim)
            se = se.to(dtype=te.dtype)
            te = te + se

        return te


    def cfm_loss(self, T0, T1, A):
        if not self.cfm_enable or self.cfm_loss_weight <= 0.0:
            return T0.new_tensor(0.0)
        B, K, d = T0.shape
        tau = torch.rand(B, 1, 1, device=T0.device).clamp_(1e-3, 1.0 - 1e-3)  # (B,1,1)
        tau_te = tau[:, :, 0]                                                  # (B,1)
        te = self._time_embed(tau_te, B, T0.device)
        T0_cfm = T0
        if self.training:
            if self.latent_cfm_noise_std > 0.0:
                T0_cfm = T0_cfm + torch.randn_like(T0_cfm) * self.latent_cfm_noise_std
            if self.latent_cfm_drop_prob > 0.0:
                keep = (torch.rand(B, K, 1, device=T0.device, dtype=T0.dtype) > self.latent_cfm_drop_prob).to(T0.dtype)
                T0_cfm = T0_cfm * keep

        T_tau = T0_cfm + (T1 - T0_cfm) * tau
        v_star = (T1 - T0_cfm)
        v_pred = self.gat(T_tau, A, te)
        return F.mse_loss(v_pred, v_star) * self.cfm_loss_weight

    def forward(self, T0, T1, A):
        B, K, d = T0.shape
        Tt = T0
        self._loss_buf.clear()
        loss_cfm = self.cfm_loss(T0, T1, A)

        kl_weighted_acc = T0.new_tensor(0.0)
        kl_count = 0

        def _accumulate_gat_kl():
            nonlocal kl_weighted_acc, kl_count
            losses = self.gat.extra_losses()
            if "loss_sb_kl" in losses:
                kl_weighted_acc = kl_weighted_acc + losses["loss_sb_kl"].to(T0.dtype)
                kl_count += 1

        if self.cfm_enable and self.cfm_loss_weight > 0.0:
            _accumulate_gat_kl()

        for t in range(self.T_steps):
            tau = (t + 1) / self.T_steps
            te = self._time_embed(tau, B, T0.device)
            target = (1.0 - tau) * T0 + tau * T1

            v_pred = self.gat(Tt, A, te)
            _accumulate_gat_kl()

            drift = v_pred if self.bridge_mode == "cfm" else (target - Tt) + v_pred
            Tt = torch.nan_to_num(Tt + drift * (1.0 / self.T_steps), nan=0.0, posinf=1e4, neginf=-1e4)

        if kl_count > 0:
            self._loss_buf["loss_sb_kl"] = kl_weighted_acc / float(kl_count)
        else:
            self._loss_buf["loss_sb_kl"] = T0.new_tensor(0.0)
        return Tt, loss_cfm


class GeoReferenceKernel(nn.Module):
    """
    Sample-conditioned Geo-Reference Kernel with soft structured reference.

    This version prevents two opposite failure modes observed in training:
      1) dense-uniform collapse: R ~= Uniform, matching ~= Uniform;
      2) over-sharpening: matching becomes nearly one-hot and R becomes too deterministic.

    Main design:
      - cross-modal matching uses token similarity plus detached assignment-overlap as a weak spatial prior;
      - matching is top-k supported but mixed with a uniform distribution on the selected support;
      - reference kernel R is top-k supported and also mixed with support-uniform mass;
      - assignment-overlap is detached by default so the detection/assignment branch is not forced into hard spatial matching.
    """
    def __init__(self, d: int, hidden: int = None,
                 temperature: float = 1.0,
                 match_temperature: float = 0.80,
                 self_loop: float = 1e-3,
                 ref_topk: int = 64,
                 sparse_reference: bool = True,
                 match_logit_scale_init: float = 0.5,
                 ref_logit_scale_init: float = 0.5,
                 match_token_weight: float = 1.0,
                 match_spatial_weight: float = 0.25,
                 ref_token_weight: float = 1.0,
                 ref_spatial_weight: float = 0.25,
                 match_topk: int = 24,
                 match_support_mix: float = 0.30,
                 ref_support_mix: float = 0.30,
                 detach_assignment_prior: bool = True):
        super().__init__()
        self.d = int(d)
        self.hidden = int(hidden) if hidden is not None else int(d)
        self.temperature = float(temperature)
        self.match_temperature = float(match_temperature)
        self.self_loop = float(self_loop)
        self.ref_topk = int(ref_topk)
        self.match_topk = int(match_topk)
        self.sparse_reference = bool(sparse_reference)
        self.match_support_mix = float(min(max(match_support_mix, 0.0), 0.95))
        self.ref_support_mix = float(min(max(ref_support_mix, 0.0), 0.95))
        self.detach_assignment_prior = bool(detach_assignment_prior)
        self.match_token_weight = float(match_token_weight)
        self.match_spatial_weight = float(match_spatial_weight)
        self.ref_token_weight = float(ref_token_weight)
        self.ref_spatial_weight = float(ref_spatial_weight)

        self.fuse = nn.Sequential(
            nn.LayerNorm(4 * self.d),
            nn.Linear(4 * self.d, self.hidden, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden, self.d, bias=False),
            nn.LayerNorm(self.d),
        )
        self.q_proj = nn.Linear(self.d, self.d, bias=False)
        self.k_proj = nn.Linear(self.d, self.d, bias=False)

        self.match_logit_scale = nn.Parameter(torch.log(torch.tensor(float(match_logit_scale_init))))
        self.ref_logit_scale = nn.Parameter(torch.log(torch.tensor(float(ref_logit_scale_init))))

    @staticmethod
    def _row_standardize(S: torch.Tensor, eps: float = 1e-5):
        S = torch.nan_to_num(S.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = S.mean(dim=-1, keepdim=True)
        std = S.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (S - mean) / std

    @staticmethod
    def _assignment_cosine(P_a: torch.Tensor, P_b: torch.Tensor):
        """Cosine overlap between latent assignment maps, used only as a weak spatial prior."""
        P_a = torch.nan_to_num(P_a.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P_b = torch.nan_to_num(P_b.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P_a = P_a / P_a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        P_b = P_b / P_b.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        P_a = F.normalize(P_a, p=2, dim=-1, eps=1e-6)
        P_b = F.normalize(P_b, p=2, dim=-1, eps=1e-6)
        return torch.einsum('bkn,bmn->bkm', P_a, P_b).clamp_min(0.0)

    @staticmethod
    def _support_softmax(logits: torch.Tensor, topk: int, support_mix: float,
                         remove_self: bool = False):
        """
        Row-wise top-k softmax with an entropy floor on the selected support.
        support_mix=0 -> normal top-k softmax;
        support_mix>0 -> mix with uniform distribution over the same top-k support.
        """
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
        B, Kq, Kt = logits.shape
        if Kt == 1:
            return torch.ones_like(logits)

        work = logits
        if remove_self and Kq == Kt:
            eye = torch.eye(Kt, device=logits.device, dtype=torch.bool).unsqueeze(0)
            work = work.masked_fill(eye, torch.finfo(work.dtype).min)

        if topk is not None and int(topk) > 0 and int(topk) < Kt:
            k_eff = min(int(topk), Kt - 1 if remove_self and Kq == Kt else Kt)
            _, topi = torch.topk(work, k=k_eff, dim=-1)
            mask = torch.zeros_like(work, dtype=torch.bool)
            mask.scatter_(-1, topi, True)
            work = work.masked_fill(~mask, torch.finfo(work.dtype).min)
        else:
            mask = torch.isfinite(work)

        work = work - work.amax(dim=-1, keepdim=True)
        prob = torch.softmax(work, dim=-1)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)

        support = (prob > 0).to(prob.dtype)
        uniform_support = support / support.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mix = float(min(max(support_mix, 0.0), 0.95))
        prob = (1.0 - mix) * prob + mix * uniform_support
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return prob

    def _soft_match(self, T_query_basis: torch.Tensor, T_other: torch.Tensor,
                    P_query_basis: torch.Tensor = None, P_other: torch.Tensor = None):
        """
        Soft one-to-many cross-modal retrieval. It should not become one-hot.
        The spatial assignment prior is detached by default to avoid forcing the tokenizer to solve
        a hard correspondence problem.
        """
        Q = F.normalize(T_query_basis, dim=-1, eps=1e-6)
        K = F.normalize(T_other, dim=-1, eps=1e-6)
        feat_sim = torch.einsum('bkd,bmd->bkm', Q, K)
        feat_logits = self._row_standardize(feat_sim)

        if P_query_basis is not None and P_other is not None:
            Pq = P_query_basis.detach() if self.detach_assignment_prior else P_query_basis
            Po = P_other.detach() if self.detach_assignment_prior else P_other
            spatial_sim = self._assignment_cosine(Pq, Po).to(feat_logits.device)
            spatial_logits = self._row_standardize(torch.log1p(4.0 * spatial_sim))
        else:
            spatial_sim = torch.zeros_like(feat_sim.float())
            spatial_logits = torch.zeros_like(feat_logits)

        logits = self.match_token_weight * feat_logits + self.match_spatial_weight * spatial_logits
        scale = self.match_logit_scale.exp().clamp(0.10, 10.0)
        logits = logits * scale / max(self.match_temperature, 1e-6)
        M = self._support_softmax(
            logits,
            topk=self.match_topk,
            support_mix=self.match_support_mix,
            remove_self=False,
        ).to(T_query_basis.dtype)
        T_other_match = torch.einsum('bkm,bmd->bkd', M, T_other)
        return T_other_match

    def _make_reference(self, T_basis: torch.Tensor, T_other_match: torch.Tensor,
                        P_basis: torch.Tensor = None):
        x = torch.cat([
            T_basis,
            T_other_match,
            T_basis - T_other_match,
            T_basis * T_other_match,
        ], dim=-1)
        H = self.fuse(x)

        Q = F.normalize(self.q_proj(H), dim=-1, eps=1e-6)
        K = F.normalize(self.k_proj(H), dim=-1, eps=1e-6)
        learned_sim = torch.einsum('bkd,bmd->bkm', Q, K)
        learned_logits = self._row_standardize(learned_sim)

        if P_basis is not None:
            Pb = P_basis.detach() if self.detach_assignment_prior else P_basis
            spatial_self = self._assignment_cosine(Pb, Pb).to(learned_logits.device)
            spatial_logits = self._row_standardize(torch.log1p(4.0 * spatial_self))
        else:
            spatial_self = torch.zeros_like(learned_sim.float())
            spatial_logits = torch.zeros_like(learned_logits)

        logits = self.ref_token_weight * learned_logits + self.ref_spatial_weight * spatial_logits
        scale = self.ref_logit_scale.exp().clamp(0.10, 10.0)
        logits = logits * scale / max(self.temperature, 1e-6)
        R = self._support_softmax(
            logits,
            topk=self.ref_topk if self.sparse_reference else 0,
            support_mix=self.ref_support_mix,
            remove_self=True,
        )

        B, Knum, _ = R.shape
        I = torch.eye(Knum, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, -1, -1)
        R = R + self.self_loop * I
        R = R / R.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        return R

    def forward(self, T_rgb: torch.Tensor, T_nir: torch.Tensor,
                P_rgb: torch.Tensor = None, P_nir: torch.Tensor = None):
        if T_rgb.shape != T_nir.shape:
            raise RuntimeError(f"GeoReferenceKernel expects paired latent shapes to match, got {T_rgb.shape} and {T_nir.shape}")

        T_nir_on_rgb = self._soft_match(T_rgb, T_nir, P_rgb, P_nir)
        T_rgb_on_nir = self._soft_match(T_nir, T_rgb, P_nir, P_rgb)
        R_rgb = self._make_reference(T_rgb, T_nir_on_rgb, P_rgb)
        R_nir = self._make_reference(T_nir, T_rgb_on_nir, P_nir)
        return R_rgb, R_nir



class LatentReconstruction(nn.Module):
    r"""
    使用 P（B,K,N）将 latent tokens (B,K,d) 回投影到空间特征：
      \\hat{Z}_n = sum_k P_{k,n} T_k
    允许传入 bias_nk (B,N,K)，在回投影前对每个像素n的K分配做加性偏置重加权：
      P_hat(:,n) = softmax( log(P(:,n)+eps) + alpha * bias(:,n) )
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj_out = nn.Linear(d_in, d_out, bias=False)

    def forward(self, T, P, H, W, bias=None, bias_alpha: float = 0.0):
        B, K, d = T.shape
        if bias is not None and bias_alpha != 0.0:
            bias_kn = bias.transpose(1, 2).contiguous()   # (B,K,N)
            logP = torch.log(P.clamp_min(1e-9))
            P = torch.softmax(logP + bias_alpha * bias_kn, dim=1)  # 列方向(K)归一
        z_recon = torch.einsum('bkn,bkd->bnd', P.contiguous(), T)  # (B,N,d)
        z_recon = self.proj_out(z_recon)                           # (B,N,d_out)
        x = z_recon.transpose(1, 2).reshape(B, -1, H, W)           # (B,C,H,W)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x



class GNSBOperator(nn.Module):
    """
    流程：
      - 两路 LatentAssignment 得 (T_rgb, P_rgb), (T_nir, P_nir)
      - 各自构造模态内 latent graph A_rgb/A_nir
      - 由当前 RGB/NIR latent pair 动态生成 Geo-Reference Kernel R_rgb/R_nir
      - 使用有界可学习 λ 混合 A_modal 与 R_ref
      - GAT 中的 QK transition policy 通过 KL(pi || R_ref) 向参考核靠近
      - 用对应 P 回投影，再经门控输出
    """
    def __init__(self, dim,
                 token_dim=None, K=256,
                 graph_topk=8,
                 gno_hidden=128, gno_layers=2,
                 sb_T_steps=4, time_dim=64,
                 cfm_enable=False, cfm_loss_weight=0.0, bridge_mode="hybrid",
                 recon_dim=None,
                 share_assignment: bool = False,
                 pos_bias_scales: int = 2, pos_bias_alpha: float = 1.0,
                 ref_mix_lambda_init: float = 0.70,
                 ref_mix_lambda_min: float = 0.35,
                 ref_mix_lambda_max: float = 0.90,
                 ref_mix_learnable: bool = True,
                 ref_mix_separate_modalities: bool = True,
                 ref_match_temperature: float = 0.80,
                 ref_match_logit_scale_init: float = 0.5,
                 ref_logit_scale_init: float = 0.5,
                 ref_topk: int = 64,
                 ref_sparse: bool = True,
                 ref_target_alpha: float = 0.15,
                 ref_match_token_weight: float = 1.5,
                 ref_match_spatial_weight: float = 0.10,
                 ref_token_weight: float = 1.5,
                 ref_spatial_weight: float = 0.10,
                 ref_match_topk: int = 24,
                 ref_match_support_mix: float = 0.30,
                 ref_support_mix: float = 0.30,
                 ref_detach_assignment_prior: bool = True,
                 ref_kl_mix_with_modal: float = 0.15,
                 bridge_output_alpha: float = 0.35,
                 enable_gbe: bool = True,
                 use_reference_kernel: bool = True,
                 use_rw_kl: bool = True,
                 kl_rw_weight: float = 1e-3,
                 cfm_steps: int = None,
                 latent_cfm_noise_std: float = 0.0,
                 latent_cfm_drop_prob: float = 0.0,
                 subgraph_refine: bool = True,
                 subgraph_neighbors: int = 8,
                 subgraph_layers: int = 2,
                 subgraph_hidden: int = None,
                 subgraph_dropout: float = 0.0,
                 pos_bias_type: str = "lbpb",  # 'lbpb' | 'alibi' | 'rpe' | 'rope' | 'none'
                 ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.token_dim = token_dim if token_dim is not None else dim
        
        

        self.pos_bias_alpha = float(pos_bias_alpha)
        
        self.pos_bias_type = pos_bias_type.lower()
        if self.pos_bias_type == "lbpb":
            self.pos_bias_mod = LatentBaryPosBias(num_heads=1, num_scales=pos_bias_scales)
        else:
            self.pos_bias_mod = None

        self.share_assignment = bool(share_assignment)
        self.assign_rgb = AttnAssignment(in_dim=dim, token_dim=self.token_dim, K=K)
        self.assign_nir = self.assign_rgb if self.share_assignment else AttnAssignment(in_dim=dim, token_dim=self.token_dim, K=K)

        self.graph_maker = LatentGraph(topk=graph_topk)

        self.geo_ref = GeoReferenceKernel(
            d=self.token_dim,
            hidden=self.token_dim,
            temperature=1.0,
            match_temperature=ref_match_temperature,
            self_loop=1e-3,
            ref_topk=ref_topk,
            sparse_reference=ref_sparse,
            match_logit_scale_init=ref_match_logit_scale_init,
            ref_logit_scale_init=ref_logit_scale_init,
            match_token_weight=ref_match_token_weight,
            match_spatial_weight=ref_match_spatial_weight,
            ref_token_weight=ref_token_weight,
            ref_spatial_weight=ref_spatial_weight,
            match_topk=ref_match_topk,
            match_support_mix=ref_match_support_mix,
            ref_support_mix=ref_support_mix,
            detach_assignment_prior=ref_detach_assignment_prior,
        )
        self.ref_target_alpha = float(ref_target_alpha)
        self.ref_kl_mix_with_modal = float(min(max(ref_kl_mix_with_modal, 0.0), 0.95))
        self.bridge_output_alpha = float(min(max(bridge_output_alpha, 0.0), 1.0))
        self.enable_gbe = bool(enable_gbe)
        self.use_reference_kernel = bool(use_reference_kernel)
        self.use_rw_kl = bool(use_rw_kl)
        self.kl_rw_weight = float(kl_rw_weight) if self.use_rw_kl else 0.0
        self.sb_T_steps_config = int(sb_T_steps)
        self.cfm_steps = int(cfm_steps) if cfm_steps is not None else self.sb_T_steps_config
        if self.cfm_steps < 1:
            raise ValueError(
                f"cfm_steps must be >= 1, got {self.cfm_steps}. "
                f"For w/o CFM, set cfm_enable=False and keep cfm_steps >= 1."
            )
        self.latent_cfm_noise_std = float(latent_cfm_noise_std)
        self.latent_cfm_drop_prob = float(latent_cfm_drop_prob)

        self.sb_A2C = LatentSBOnGraph(d=self.token_dim, gno_hidden=gno_hidden,
                                      gno_layers=gno_layers,
                                      T_steps=self.cfm_steps,
                                      time_dim=time_dim, cfm_enable=cfm_enable,
                                      cfm_loss_weight=cfm_loss_weight, bridge_mode=bridge_mode,
                                      kl_rw_weight=self.kl_rw_weight,
                                      cfm_steps=self.cfm_steps,
                                      latent_cfm_noise_std=latent_cfm_noise_std,
                                      latent_cfm_drop_prob=latent_cfm_drop_prob)
        self.sb_B2C = LatentSBOnGraph(d=self.token_dim, gno_hidden=gno_hidden,
                                      gno_layers=gno_layers,
                                      T_steps=self.cfm_steps,
                                      time_dim=time_dim, cfm_enable=cfm_enable,
                                      cfm_loss_weight=cfm_loss_weight, bridge_mode=bridge_mode,
                                      kl_rw_weight=self.kl_rw_weight,
                                      cfm_steps=self.cfm_steps,
                                      latent_cfm_noise_std=latent_cfm_noise_std,
                                      latent_cfm_drop_prob=latent_cfm_drop_prob)

        d_out = recon_dim if recon_dim is not None else dim
        self.d_out = int(d_out)
        self.recon_rgb = LatentReconstruction(self.token_dim, d_out)
        self.recon_nir = LatentReconstruction(self.token_dim, d_out)

        self.gate_rgb = nn.Sequential(
            nn.Conv2d(d_out * 2, d_out, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(d_out, d_out, kernel_size=1, bias=False)
        )
        self.gate_nir = nn.Sequential(
            nn.Conv2d(d_out * 2, d_out, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(d_out, d_out, kernel_size=1, bias=False)
        )

        self.ref_mix_lambda_min = float(ref_mix_lambda_min)
        self.ref_mix_lambda_max = float(ref_mix_lambda_max)
        if not (0.0 <= self.ref_mix_lambda_min < self.ref_mix_lambda_max <= 1.0):
            raise ValueError(
                f"Invalid ref_mix lambda bounds: "
                f"min={self.ref_mix_lambda_min}, max={self.ref_mix_lambda_max}"
            )
        self.ref_mix_separate_modalities = bool(ref_mix_separate_modalities)
        self.ref_mix_learnable = bool(ref_mix_learnable)

        def _init_bounded_logit(init_value: float) -> torch.Tensor:
            init_value = float(init_value)
            init_value = min(
                max(init_value, self.ref_mix_lambda_min + 1e-6),
                self.ref_mix_lambda_max - 1e-6,
            )
            p = (init_value - self.ref_mix_lambda_min) / (
                self.ref_mix_lambda_max - self.ref_mix_lambda_min
            )
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            return torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32)

        if self.ref_mix_learnable:
            if self.ref_mix_separate_modalities:
                self.ref_mix_lambda_logit_rgb = nn.Parameter(_init_bounded_logit(ref_mix_lambda_init))
                self.ref_mix_lambda_logit_nir = nn.Parameter(_init_bounded_logit(ref_mix_lambda_init))
                self.ref_mix_lambda_logit = None
            else:
                self.ref_mix_lambda_logit = nn.Parameter(_init_bounded_logit(ref_mix_lambda_init))
                self.ref_mix_lambda_logit_rgb = None
                self.ref_mix_lambda_logit_nir = None
        else:
            const_value = float(ref_mix_lambda_init)
            const_value = min(max(const_value, self.ref_mix_lambda_min), self.ref_mix_lambda_max)
            self.register_buffer('ref_mix_lambda_const_rgb', torch.tensor(const_value, dtype=torch.float32))
            self.register_buffer('ref_mix_lambda_const_nir', torch.tensor(const_value, dtype=torch.float32))
            self.ref_mix_lambda_logit = None
            self.ref_mix_lambda_logit_rgb = None
            self.ref_mix_lambda_logit_nir = None

            
        self.subgraph_refine = bool(subgraph_refine)
        if self.subgraph_refine:
            self.local_refiner = LocalSubgraphRefiner(
                d=self.token_dim,
                m_neighbors=subgraph_neighbors,
                num_layers=subgraph_layers,
                hidden=subgraph_hidden if subgraph_hidden is not None else self.token_dim,
                dropout=subgraph_dropout
            )

        self._extra_losses = {}
        

    def extra_losses(self):
        return dict(self._extra_losses)

    def _bounded_lambda_from_logit(self, logit: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained logit to [ref_mix_lambda_min, ref_mix_lambda_max]."""
        return self.ref_mix_lambda_min + (
            self.ref_mix_lambda_max - self.ref_mix_lambda_min
        ) * torch.sigmoid(logit)

    def _get_ref_mix_lambdas(self, dtype, device):
        """
        Return λ_rgb, λ_nir in bounded range.
        λ controls how much we trust the modality-specific top-k graph:
            A_used = λ A_modal + (1-λ) R_ref.
        """
        if self.ref_mix_learnable:
            if self.ref_mix_separate_modalities:
                lam_rgb = self._bounded_lambda_from_logit(self.ref_mix_lambda_logit_rgb)
                lam_nir = self._bounded_lambda_from_logit(self.ref_mix_lambda_logit_nir)
            else:
                lam = self._bounded_lambda_from_logit(self.ref_mix_lambda_logit)
                lam_rgb, lam_nir = lam, lam
        else:
            lam_rgb = self.ref_mix_lambda_const_rgb
            lam_nir = self.ref_mix_lambda_const_nir
        return lam_rgb.to(dtype=dtype, device=device), lam_nir.to(dtype=dtype, device=device)

    def _mix_with_reference(self, A_modal, R_ref, lam):
        """
        Reference-guided soft topology.

        A_modal 是当前模态由 hard top-k 得到的候选图；R_ref 是动态生成的
        dense random-walk reference kernel。混合后重新 row-normalize，保证
        message passing 尺度稳定。
        """
        if R_ref.ndim == 2:
            R_ref = R_ref.unsqueeze(0).expand(A_modal.shape[0], -1, -1)
        elif R_ref.ndim == 3 and R_ref.shape[0] == 1 and A_modal.shape[0] > 1:
            R_ref = R_ref.expand(A_modal.shape[0], -1, -1)
        R_ref = R_ref.to(dtype=A_modal.dtype, device=A_modal.device)
        lam = lam.to(dtype=A_modal.dtype, device=A_modal.device)
        A_used = lam * A_modal + (1.0 - lam) * R_ref
        return _ensure_rw_kernel(A_used)

    def set_cfm_steps(self, steps: int):
        """Update CFM/bridge evolution steps for step-ablation runs."""
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"cfm_steps must be >= 1, got {steps}")
        self.cfm_steps = steps
        self.sb_A2C.set_cfm_steps(steps)
        self.sb_B2C.set_cfm_steps(steps)

    def forward(self, rgb, nir):
        B, C, H, W = rgb.shape

        if not self.enable_gbe:
            self._extra_losses.clear()
            if C == self.d_out:
                rgb_out = self.gate_rgb(torch.cat([rgb, nir], dim=1))
                nir_out = self.gate_nir(torch.cat([nir, rgb], dim=1))
            else:
                rgb_out, nir_out = rgb, nir
            rgb_out = torch.nan_to_num(rgb_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
            nir_out = torch.nan_to_num(nir_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
            return rgb_out, nir_out

        T_rgb, P_rgb, _ = self.assign_rgb(rgb)   # (B,K,d), (B,K,N)
        T_nir, P_nir, _ = self.assign_nir(nir)

        A_rgb = self.graph_maker(T_rgb, T_rgb)   # (B,K,K)
        A_nir = self.graph_maker(T_nir, T_nir)   # (B,K,K)
        
        if self.subgraph_refine:
            T_rgb = self.local_refiner(T_rgb, A_rgb)  # -> T_rgb'
            T_nir = self.local_refiner(T_nir, A_nir)  # -> T_nir'
            A_rgb = self.graph_maker(T_rgb, T_rgb)
            A_nir = self.graph_maker(T_nir, T_nir)

        if self.use_reference_kernel:
            R_rgb, R_nir = self.geo_ref(T_rgb, T_nir, P_rgb, P_nir)
        else:
            R_rgb = _ensure_rw_kernel(A_rgb)
            R_nir = _ensure_rw_kernel(A_nir)

        lam_rgb, lam_nir = self._get_ref_mix_lambdas(dtype=A_rgb.dtype, device=A_rgb.device)

        A_rgb_used = self._mix_with_reference(A_rgb, R_rgb, lam_rgb)
        A_nir_used = self._mix_with_reference(A_nir, R_nir, lam_nir)

        T_rgb_smooth = torch.einsum('bkm,bmd->bkd', R_rgb.to(T_rgb.dtype), T_rgb)
        T_nir_smooth = torch.einsum('bkm,bmd->bkd', R_nir.to(T_nir.dtype), T_nir)
        alpha_ref = float(self.ref_target_alpha)
        T_rgb_ref = T_rgb + alpha_ref * (T_rgb_smooth - T_rgb)
        T_nir_ref = T_nir + alpha_ref * (T_nir_smooth - T_nir)

        kl_mix = float(self.ref_kl_mix_with_modal)
        R_rgb_kl = _ensure_rw_kernel((1.0 - kl_mix) * R_rgb + kl_mix * A_rgb)
        R_nir_kl = _ensure_rw_kernel((1.0 - kl_mix) * R_nir + kl_mix * A_nir)

        self.sb_A2C.gat._ref_A = R_rgb_kl if self.use_rw_kl else None
        T_rgb2c_raw, loss_r = self.sb_A2C(T_rgb, T_rgb_ref, A_rgb_used)
        self.sb_A2C.gat._ref_A = None

        self.sb_B2C.gat._ref_A = R_nir_kl if self.use_rw_kl else None
        T_nir2c_raw, loss_n = self.sb_B2C(T_nir, T_nir_ref, A_nir_used)
        self.sb_B2C.gat._ref_A = None

        beta_bridge = float(self.bridge_output_alpha)
        T_rgb2c = T_rgb + beta_bridge * (T_rgb2c_raw - T_rgb)
        T_nir2c = T_nir + beta_bridge * (T_nir2c_raw - T_nir)

        if self.pos_bias_type != "none":
            if self.pos_bias_type == "lbpb":
                attn_rgb = P_rgb.unsqueeze(1)  # (B,1,K,N)
                attn_nir = P_nir.unsqueeze(1)
                bias_rgb = self.pos_bias_mod(attn_rgb, H, W).squeeze(1)  # (B,N,K)
                bias_nir = self.pos_bias_mod(attn_nir, H, W).squeeze(1)
            else:
                bias_rgb = ParamFreePosBias.compute(P_rgb, H, W, mode=self.pos_bias_type)  # (B,N,K)
                bias_nir = ParamFreePosBias.compute(P_nir, H, W, mode=self.pos_bias_type)
        else:
            bias_rgb, bias_nir = None, None

        self._extra_losses.clear()
        self._extra_losses["loss_cfm_rgb2c"] = loss_r
        self._extra_losses["loss_cfm_nir2c"] = loss_n

        for prefix, bridge in (("rgb2c", self.sb_A2C), ("nir2c", self.sb_B2C)):
            losses = bridge.extra_losses()
            if "loss_sb_kl" in losses:
                self._extra_losses[f"{prefix}_loss_sb_kl"] = losses["loss_sb_kl"]

        assign_rgb_losses = self.assign_rgb.extra_losses()
        if "loss_attn_div" in assign_rgb_losses:
            self._extra_losses["assign_rgb_loss_attn_div"] = assign_rgb_losses["loss_attn_div"]
        if not self.share_assignment:
            assign_nir_losses = self.assign_nir.extra_losses()
            if "loss_attn_div" in assign_nir_losses:
                self._extra_losses["assign_nir_loss_attn_div"] = assign_nir_losses["loss_attn_div"]

        x_rgb = self.recon_rgb(T_rgb2c, P_rgb, H, W, bias=bias_rgb, bias_alpha=self.pos_bias_alpha)  # (B,C,H,W)
        x_nir = self.recon_nir(T_nir2c, P_nir, H, W, bias=bias_nir, bias_alpha=self.pos_bias_alpha)  # (B,C,H,W)

        xrgb = torch.cat([rgb, x_rgb], dim=1)        # (B,2C,H,W)
        xnir = torch.cat([nir, x_nir], dim=1)        # (B,2C,H,W)
        rgb_out = self.gate_rgb(xrgb)
        nir_out = self.gate_nir(xnir)

        rgb_out = torch.nan_to_num(rgb_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        nir_out = torch.nan_to_num(nir_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        return rgb_out, nir_out



class GNSBOperatorFusion(nn.Module):
    """
    外部调用接口：输入两路 (B,C,H,W)，输出两路融合 (B,C,H,W)

    可选参数：
      - share_group: str，同一组名的实例共享同一“核心”权重
      - share_owner: bool，是否由该实例负责注册核心（建议只让 P5 置 True，其它尺度 False）
      - core_dim: int，把各尺度输入适配到该内部通道数做融合（必须各尺度一致）
      - num_scales: int，共享的尺度数（默认3）
      - scale_id: int，本实例对应的尺度编号（如 P3:0, P4:1, P5:2）
    不传以上参数时，各尺度使用独立权重。
    """
    def __init__(self, dim, **kwargs):
        super().__init__()

        self._share_group: str = kwargs.pop('share_group', None)
        self._share_owner: bool = bool(kwargs.pop('share_owner', False))
        self._core_dim: int = kwargs.pop('core_dim', None)
        self._num_scales: int = int(kwargs.pop('num_scales', 3))
        self._scale_id: int = int(kwargs.pop('scale_id', 0))

        self._time_dim: int = int(kwargs.get('time_dim', 64))

        if self._share_group is None:
            self.fuse = GNSBOperator(dim=dim, **kwargs)
            self.scale_embed = nn.Embedding(self._num_scales, self._time_dim)
            self.current_scale = self._scale_id
            return

        in_dim = int(dim)
        core_dim = int(self._core_dim or in_dim)

        self.rgb_in = nn.Conv2d(in_dim, core_dim, 1, bias=False) if core_dim != in_dim else nn.Identity()
        self.nir_in = nn.Conv2d(in_dim, core_dim, 1, bias=False) if core_dim != in_dim else nn.Identity()
        self.rgb_out = nn.Conv2d(core_dim, in_dim, 1, bias=False) if core_dim != in_dim else nn.Identity()
        self.nir_out = nn.Conv2d(core_dim, in_dim, 1, bias=False) if core_dim != in_dim else nn.Identity()

        group = self._share_group
        if group not in _GNSB_SHARED:
            assert self._share_owner, f"[{group}] 尚未创建共享核心：请把 P5 那个实例的 share_owner 设为 True"
            core_kwargs = dict(kwargs)
            core_kwargs['recon_dim'] = core_dim
            self.fuse = GNSBOperator(dim=core_dim, **core_kwargs)
            self.scale_embed = nn.Embedding(self._num_scales, self._time_dim)
            _GNSB_SHARED[group] = {'core': self.fuse, 'scale_embed': self.scale_embed, 'owner': self}
        else:
            self._shared_group = group
        self.current_scale = self._scale_id

    def set_scale(self, sid: int):
        self.current_scale = int(sid)

    def set_cfm_steps(self, steps: int):
        """Forward runtime CFM-step updates to the underlying GNSB core."""
        core, _ = self._get_shared_core_and_embed()
        if hasattr(core, "set_cfm_steps"):
            core.set_cfm_steps(steps)

    def extra_losses(self):
        core, _ = self._get_shared_core_and_embed()
        return core.extra_losses()

    def _get_shared_core_and_embed(self):
        if self._share_group is None:
            return self.fuse, self.scale_embed
        if hasattr(self, 'fuse') and isinstance(self.fuse, nn.Module):
            return self.fuse, _GNSB_SHARED[self._share_group]['scale_embed']
        pack = _GNSB_SHARED[self._share_group]
        return pack['core'], pack['scale_embed']

    def forward(self, x):
        rgb, nir = x

        if self._share_group is None:
            if hasattr(self, "scale_embed"):
                self.fuse.sb_A2C.scale_embed = self.scale_embed
                self.fuse.sb_B2C.scale_embed = self.scale_embed
                self.fuse.sb_A2C.current_scale = getattr(self, "current_scale", 0)
                self.fuse.sb_B2C.current_scale = getattr(self, "current_scale", 0)
            rgb_out, nir_out = self.fuse(rgb, nir)
            rgb_out = torch.nan_to_num(rgb_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
            nir_out = torch.nan_to_num(nir_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
            return rgb_out, nir_out

        core, se = self._get_shared_core_and_embed()

        core.sb_A2C.scale_embed = se
        core.sb_B2C.scale_embed = se
        core.sb_A2C.current_scale = getattr(self, "current_scale", 0)
        core.sb_B2C.current_scale = getattr(self, "current_scale", 0)

        rgb_p = self.rgb_in(rgb)
        nir_p = self.nir_in(nir)

        rgb_p_out, nir_p_out = core(rgb_p, nir_p)

        rgb_out = self.rgb_out(rgb_p_out)
        nir_out = self.nir_out(nir_p_out)

        rgb_out = torch.nan_to_num(rgb_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        nir_out = torch.nan_to_num(nir_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        return rgb_out, nir_out
