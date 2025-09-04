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
    Switch-style MoE FFN with top-k routing, capacity factor, load-balance/importance loss,
    z-loss on router logits, temperature & noise in gating.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int = 8,
            dropout: float = 0.1,
            activation: str = "relu",
            # router & regularization
            top_k: int = 1,
            capacity_factor: float = 1.25,
            gate_temp: float = 1.0,
            gate_noise_std: float = 0.0,
            lb_alpha: float = 0.0,
            imp_alpha: float = 0.0,
            zloss_beta: float = 0.0,
            entropy_reg: float = 0.0,
    ):
        super().__init__()
        assert top_k in (1, 2)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_temp = gate_temp
        self.gate_noise_std = gate_noise_std
        self.lb_alpha = lb_alpha
        self.imp_alpha = imp_alpha
        self.zloss_beta = zloss_beta
        self.entropy_reg = entropy_reg

        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff, dropout, activation) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts)

        self._last_aux = {
            "balance": torch.tensor(0.0),
            "importance": torch.tensor(0.0),
            "zloss": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        }

    @torch.no_grad()
    def init_from_conv1x1(self, conv1, conv2, noise_std: float = 0.0):
        """
        从 Dense FFN(conv1,conv2: 1x1 Conv 等价 Linear) 初始化 expert0；其余专家加小噪声微扰。
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

    def _routing(self, h: torch.Tensor, training: bool):
        B, L, D = h.shape
        T = B * L
        x = h.reshape(T, D)

        logits = self.gate(x)
        if training and self.gate_noise_std > 0:
            logits = logits + self.gate_noise_std * torch.randn_like(logits)
        if self.gate_temp != 1.0:
            logits = logits / self.gate_temp

        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        importance = probs.sum(dim=0) / T
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, topk_idx, 1.0)
        load = one_hot.sum(dim=0) / T

        zloss = (logits.logsumexp(dim=-1) ** 2).mean()
        ent = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()

        aux_balance = (importance * load).sum() * self.num_experts
        aux_importance = (importance * importance).sum() * self.num_experts

        self._last_aux = {
            "balance": aux_balance.detach(),
            "importance": aux_importance.detach(),
            "zloss": zloss.detach(),
            "entropy": ent.detach(),
            "total": torch.tensor(0.0),
        }
        return topk_idx, topk_prob, logits

    def forward(self, h: torch.Tensor):
        B, L, D = h.shape
        T = B * L
        topk_idx, topk_prob, logits = self._routing(h, self.training)

        capacity = math.ceil(self.capacity_factor * (T * self.top_k / self.num_experts))

        dispatch = [[] for _ in range(self.num_experts)]
        prob_for_token = [[] for _ in range(self.num_experts)]

        for choice in range(self.top_k):
            idx = topk_idx[:, choice]
            prb = topk_prob[:, choice]
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    sel = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if sel.numel() > capacity:
                        sel = sel[:capacity]
                    dispatch[e].append(sel)
                    prob_for_token[e].append(prb[sel])

        out = h.new_zeros(B * L, D)
        h_flat = h.reshape(T, D)
        for e, expert in enumerate(self.experts):
            if len(dispatch[e]) == 0:
                continue
            sel = torch.cat(dispatch[e], dim=0)
            w = torch.cat(prob_for_token[e], dim=0).unsqueeze(-1)
            inp = h_flat[sel]
            y = expert(inp) * w
            out.index_add_(0, sel, y)

        y = out.reshape(B, L, D)

        aux_total = 0.0
        if self.training:
            if self.lb_alpha > 0:
                aux_total = aux_total + self.lb_alpha * self._last_aux["balance"]
            if self.imp_alpha > 0:
                aux_total = aux_total + self.imp_alpha * self._last_aux["importance"]
            if self.zloss_beta > 0:
                aux_total = aux_total + self.zloss_beta * self._last_aux["zloss"]
            if self.entropy_reg > 0:
                aux_total = aux_total + self.entropy_reg * self._last_aux["entropy"]
        self._last_aux["total"] = torch.tensor(float(aux_total))

        return y
