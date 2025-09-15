# layers/moe_ffn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))


class MoEFeedForward(nn.Module):
    """
    改良版 Switch-style MoE FFN：
    - 支持 top-k(1或2) 路由与容量限制(capacity factor)
    - learnable_gate_temp：路由 softmax 温度可学习
    - gate_dropout：训练期随机屏蔽部分专家（鼓励探索，缓解失衡），保证至少保留 top_k 个专家
    - 负载均衡正则：Switch-style balance + KL(uniform||load)
    - z-loss / entropy 正则
    - top_k=2 时的“回退装箱”：top1 溢出 token 尝试分配到 top2
    - 强力数值防护：softmax 前后、正则与输出均做 NaN/Inf 清理
    - _last_aux：{'balance_switch','balance_kl','importance','assigned_frac','entropy','zloss','total'}
    * 与旧版保持兼容：不传新增参数时行为与原实现一致
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int = 8,
            dropout: float = 0.1,
            activation: str = "relu",
            # 原有路由/正则参数
            top_k: int = 1,
            capacity_factor: float = 1.25,
            gate_temp: float = 1.0,
            gate_noise_std: float = 0.0,
            lb_alpha: float = 0.0,
            imp_alpha: float = 0.0,
            zloss_beta: float = 0.0,
            entropy_reg: float = 0.0,
            # 新增（均有默认，保持兼容）
            learnable_gate_temp: bool = False,
            gate_dropout: float = 0.0,   # [0,1)
            kl_alpha: float = 0.0,       # KL(uniform||load) 权重
    ):
        super().__init__()
        assert top_k in (1, 2)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.base_gate_temp = float(gate_temp)
        self.learnable_gate_temp = learnable_gate_temp
        self.gate_noise_std = gate_noise_std
        self.lb_alpha = lb_alpha
        self.imp_alpha = imp_alpha
        self.zloss_beta = zloss_beta
        self.entropy_reg = entropy_reg
        self.gate_dropout = gate_dropout
        self.kl_alpha = kl_alpha

        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff, dropout, activation) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts)
        # 路由器零初始化：均匀起步更稳
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # 可学习温度（softplus>0），不启用则为常数
        if self.learnable_gate_temp:
            init = torch.tensor(self.base_gate_temp, dtype=torch.float32)
            self._log_temp = nn.Parameter(torch.log(torch.exp(init) - 1.0))
        else:
            self.register_parameter("_log_temp", None)
        # 温度下限，避免过尖锐
        self._min_temp = 0.2

        self._last_aux = {
            "balance_switch": torch.tensor(0.0),
            "balance_kl": torch.tensor(0.0),
            "importance": torch.tensor(0.0),
            "assigned_frac": torch.tensor(0.0),
            "zloss": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        }

    @torch.no_grad()
    def init_from_conv1x1(self, conv1, conv2, noise_std: float = 0.0):
        """
        从 Dense FFN(conv1/conv2: 1x1 Conv 等价 Linear) 初始化 expert0；其余专家加小噪声微扰。
        """
        w1 = conv1.weight.squeeze(-1)  # [d_ff, d_model]
        b1 = conv1.bias
        w2 = conv2.weight.squeeze(-1)  # [d_model, d_ff]
        b2 = conv2.bias

        def copy_into(expert: ExpertFFN, add_noise: bool):
            with torch.no_grad():
                expert.w1.weight.copy_(w1)
                expert.w1.bias.copy_(b1)
                expert.w2.weight.copy_(w2)
                expert.w2.bias.copy_(b2)
                if add_noise and noise_std > 0:
                    for p in expert.parameters():
                        p.add_(noise_std * torch.randn_like(p))

        copy_into(self.experts[0], add_noise=False)
        for e in self.experts[1:]:
            copy_into(e, add_noise=True)

    def _effective_temp(self, device):
        if self.learnable_gate_temp:
            temp = F.softplus(self._log_temp) + 1e-6
            return torch.clamp(temp, min=self._min_temp)
        return torch.tensor(max(self._min_temp, self.base_gate_temp), device=device)

    def _routing(self, h: torch.Tensor, training: bool):
        B, L, D = h.shape
        T = B * L
        x = h.reshape(T, D)

        logits = self.gate(x)  # [T, E]
        logits = torch.nan_to_num(logits, neginf=-1e4, posinf=1e4)

        # 训练期扰动
        if training and self.gate_noise_std > 0:
            logits = logits + self.gate_noise_std * torch.randn_like(logits)

        # expert dropout（保证至少保留 top_k 个专家）
        if training and self.gate_dropout > 0.0:
            drop = torch.rand(self.num_experts, device=logits.device) < self.gate_dropout
            max_drop = self.num_experts - self.top_k
            if drop.sum() > max_drop:
                # 恢复多余的 drop，随机选择恢复
                to_recover = (drop.sum() - max_drop).item()
                idx = torch.nonzero(drop, as_tuple=False).squeeze(-1)
                rec = idx[torch.randperm(idx.numel(), device=idx.device)[:to_recover]]
                drop[rec] = False
            logits = logits.masked_fill(drop.unsqueeze(0), float('-inf'))

        temp = self._effective_temp(logits.device)
        if temp.item() != 1.0:
            logits = logits / temp

        probs = F.softmax(logits, dim=-1)            # [T, E]
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        # 行归一化兜底：若全 0，则置为均匀分布
        row_sum = probs.sum(dim=-1, keepdim=True)
        bad = row_sum.squeeze(-1) <= 1e-12
        if bad.any():
            probs[bad] = 1.0 / self.num_experts
        else:
            probs = probs / row_sum.clamp_min(1e-12)

        topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # [T, K]

        # 统计信息
        importance = probs.sum(dim=0) / T            # [E]
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, topk_idx, 1.0)
        load = one_hot.sum(dim=0) / T                # [E]

        zloss = (logits.logsumexp(dim=-1) ** 2).mean()
        zloss = torch.nan_to_num(zloss, nan=0.0, posinf=1e4, neginf=0.0)
        ent = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
        ent = torch.nan_to_num(ent, nan=0.0)

        # Switch-style balance（越大越好）
        balance_switch = (importance * load).sum() * self.num_experts
        # KL(uniform || load)（越小越好）
        uniform = torch.full_like(load, 1.0 / self.num_experts)
        balance_kl = (uniform * (uniform / load.clamp_min(1e-9)).log()).sum()
        balance_kl = torch.nan_to_num(balance_kl, nan=0.0, posinf=0.0, neginf=0.0)

        # importance（平方和）按 E 缩放，方便不同 E 可比
        imp = (importance * importance).sum() * self.num_experts

        self._last_aux.update({
            "balance_switch": balance_switch.detach(),
            "balance_kl": balance_kl.detach(),
            "importance": imp.detach(),
            "zloss": zloss.detach(),
            "entropy": ent.detach(),
        })

        return topk_idx, topk_prob, probs

    def forward(self, h: torch.Tensor):
        B, L, D = h.shape
        T = B * L

        topk_idx, topk_prob, _ = self._routing(h, self.training)

        # 每个 expert 的容量
        capacity = math.ceil(self.capacity_factor * (T * self.top_k / self.num_experts))
        capacity = max(1, capacity)

        h_flat = h.reshape(T, D)
        out = h.new_zeros(T, D)
        assigned = 0

        if self.top_k == 1:
            idx = topk_idx[:, 0]     # [T]
            prb = topk_prob[:, 0]    # [T]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    sel = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if sel.numel() > capacity:
                        sel = sel[:capacity]
                    inp = h_flat[sel]
                    w = prb[sel].unsqueeze(-1)
                    y = self.experts[e](inp) * w
                    out.index_add_(0, sel, y)
                    assigned += sel.numel()
        else:
            # Top-2 回退装箱
            top1_idx, top2_idx = topk_idx[:, 0], topk_idx[:, 1]
            top1_prb, top2_prb = topk_prob[:, 0], topk_prob[:, 1]

            buckets = {e: torch.nonzero(top1_idx == e, as_tuple=False).squeeze(-1)
                       for e in range(self.num_experts)}
            used_top2 = {e: [] for e in range(self.num_experts)}

            # 处理 top1 溢出
            remain = []
            for e in range(self.num_experts):
                sel = buckets[e]
                if sel.numel() > capacity:
                    remain.append(sel[capacity:])
                    buckets[e] = sel[:capacity]
            if len(remain) > 0:
                remain = torch.cat(remain, dim=0)
                sizes_top1 = {e: buckets[e].numel() for e in range(self.num_experts)}
                for idx_token in remain.tolist():
                    e2 = int(top2_idx[idx_token].item())
                    cur = sizes_top1.get(e2, 0) + len(used_top2[e2])
                    if cur < capacity:
                        used_top2[e2].append(idx_token)

            # 写回输出
            for e in range(self.num_experts):
                sel1 = buckets[e]
                if sel1.numel() > 0:
                    inp1 = h_flat[sel1]
                    w1 = top1_prb[sel1].unsqueeze(-1)
                    y1 = self.experts[e](inp1) * w1
                    out.index_add_(0, sel1, y1)
                    assigned += sel1.numel()

                if len(used_top2[e]) > 0:
                    sel2 = torch.tensor(used_top2[e], device=h_flat.device, dtype=torch.long)
                    inp2 = h_flat[sel2]
                    w2 = top2_prb[sel2].unsqueeze(-1)
                    y2 = self.experts[e](inp2) * w2
                    out.index_add_(0, sel2, y2)
                    assigned += sel2.numel()

        y = out.reshape(B, L, D)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        # 组装 aux 正则
        total = 0.0
        if self.training:
            if self.lb_alpha > 0:
                total += self.lb_alpha * self._last_aux["balance_switch"]
            if self.imp_alpha > 0:
                total += self.imp_alpha * self._last_aux["importance"]
            if self.kl_alpha > 0:
                total += self.kl_alpha * self._last_aux["balance_kl"]
            if self.zloss_beta > 0:
                total += self.zloss_beta * self._last_aux["zloss"]
            if self.entropy_reg > 0:
                total += self.entropy_reg * self._last_aux["entropy"]

        self._last_aux["total"] = torch.tensor(float(total), device=y.device)
        self._last_aux["assigned_frac"] = torch.tensor(assigned / float(T), device=y.device)

        return y
