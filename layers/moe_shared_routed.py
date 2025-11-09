# layers/moe_shared_routed.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 基础专家：建议用 GEGLU ----------
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class ExpertFFN(nn.Module):
    def __init__(self, d_model, r=4, dropout=0.0, use_geglu=True):
        super().__init__()
        hid = r * d_model
        if use_geglu:
            self.fc1 = nn.Linear(d_model, 2*hid)  # GEGLU
            self.act = GEGLU()
        else:
            self.fc1 = nn.Linear(d_model, hid)
            self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

# ---------- Top-2 路由器 ----------
class Top2Router(nn.Module):
    def __init__(self, d_model, n_routed, noisy_std=1.0, tau=1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, n_routed)
        self.noisy_std = float(noisy_std)
        self.tau = float(tau)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, train: bool = True):
        logits = self.proj(x)                      # [B,T,E]
        if train and self.noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_std
        logits = logits / max(self.tau, 1e-6)

        # Top-2 索引/值
        top_val, top_idx = torch.topk(logits, k=2, dim=-1)   # [B,T,2]
        top_w = F.softmax(top_val, dim=-1)                   # [B,T,2]

        # full 概率（做均衡损失的 p_i）
        probs_full = F.softmax(logits, dim=-1)               # [B,T,E]
        return top_idx, top_w, probs_full, logits

# ---------- 主模块：共享 + 路由 ----------
class SharedRoutedMoE(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_shared: int = 1,
                 n_routed: int = 12,
                 r_shared: int = 2,
                 r_routed: int = 3,
                 expert_dropout: float = 0.1,
                 router_noisy_std: float = 1.0,
                 router_tau: float = 1.5,
                 capacity_factor: float | None = 1.25,  # None 表示 dropless
                 use_dropless: bool = False):
        super().__init__()
        self.n_shared, self.n_routed = n_shared, n_routed
        self.capacity_factor = capacity_factor
        self.use_dropless = use_dropless

        # 共享专家（所有 token 必经）
        self.shared = nn.ModuleList([
            ExpertFFN(d_model, r=r_shared, dropout=expert_dropout, use_geglu=True)
            for _ in range(n_shared)
        ])

        # 路由专家（按 Top-2 选择）
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, r=r_routed, dropout=expert_dropout, use_geglu=True)
            for _ in range(n_routed)
        ])
        self.router = Top2Router(d_model, n_routed,
                                 noisy_std=router_noisy_std, tau=router_tau)

    @torch.no_grad()
    def _capacity(self, B, T):
        if self.capacity_factor is None:
            return None
        # Top-2：每 token 最多占用 2 个槽
        tokens = B * T * 2
        return int(math.ceil(self.capacity_factor * tokens / self.n_routed))

    def forward(self, x, train: bool = True):
        """
        x: [B,T,d]
        返回: y, aux(dict)
        aux 包含:
          f_i, p_i, router_logits, router_entropy, (可选) dropped_frac
        """
        B, T, D = x.shape
        aux = {}

        # 共享支路
        y_shared = 0.
        for s in self.shared:
            y_shared = y_shared + s(x)
        if self.n_shared > 0:
            y_shared = y_shared / self.n_shared

        # 路由
        top_idx, top_w, probs_full, logits = self.router(x, train=train)

        # 统计 f_i / p_i（用于均衡损失）
        with torch.no_grad():
            # f_i: 被 top-2 命中的比例（one-hot 统计）
            hit = F.one_hot(top_idx, num_classes=self.n_routed).sum(dim=-2)  # [B,T,E]
            f_i = hit.float().sum(dim=(0,1)) / (B*T)
            p_i = probs_full.mean(dim=(0,1))
            aux['f_i'] = f_i
            aux['p_i'] = p_i
            aux['router_entropy'] = (-probs_full * (probs_full + 1e-9).log()).sum(-1).mean()

        # 聚合/派发（dropless 或 capacity）
        y_routed = torch.zeros_like(x)
        x_flat = x.reshape(-1, D)
        idx_flat = top_idx.reshape(-1, 2)
        w_flat = top_w.reshape(-1, 2)

        if self.use_dropless or self.capacity_factor is None:
            # dropless：不丢 token
            for e in range(self.n_routed):
                m = (idx_flat == e)  # [N,2]
                if not m.any():
                    continue
                tok_idx = torch.nonzero(m.any(dim=-1), as_tuple=False).squeeze(-1)
                x_e = x_flat.index_select(0, tok_idx)
                y_e = self.experts[e](x_e)
                w_e = torch.where(m[tok_idx, 0], w_flat[tok_idx, 0], torch.zeros_like(w_flat[tok_idx, 0]))
                w_e += torch.where(m[tok_idx, 1], w_flat[tok_idx, 1], torch.zeros_like(w_flat[tok_idx, 1]))
                y_routed.view(-1, D).index_add_(0, tok_idx, y_e * w_e.unsqueeze(-1))
        else:
            # capacity：超量丢弃
            cap = self._capacity(B, T)
            dropped = 0
            for e in range(self.n_routed):
                m = (idx_flat == e).any(dim=-1)
                if not m.any():
                    continue
                tok_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
                if tok_idx.numel() > cap:
                    keep = tok_idx[:cap]
                    dropped += (tok_idx.numel() - cap)
                else:
                    keep = tok_idx
                x_e = x_flat.index_select(0, keep)
                y_e = self.experts[e](x_e)
                m2 = (idx_flat[keep] == e)
                w_e = torch.where(m2[:, 0], w_flat[keep, 0], torch.zeros_like(w_flat[keep, 0]))
                w_e += torch.where(m2[:, 1], w_flat[keep, 1], torch.zeros_like(w_flat[keep, 1]))
                y_routed.view(-1, D).index_add_(0, keep, y_e * w_e.unsqueeze(-1))
            aux['dropped_frac'] = dropped / float(B*T*2)

        y = y_shared + y_routed
        aux['router_logits'] = logits  # 供 z-loss 使用
        return y, aux

# ---------- 两个损失：Switch 风格均衡 + z-loss ----------
def moe_auxiliary_loss(f_i, p_i, alpha=0.05):
    n = f_i.numel()
    return alpha * n * torch.sum(f_i * p_i)

def router_z_loss(router_logits, beta=1e-3):
    z = torch.logsumexp(router_logits, dim=-1)  # [B,T]
    return beta * torch.mean(z ** 2)
