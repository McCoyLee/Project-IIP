# layers/moe_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unknown activation: {name}")


class _ExpertFFN(nn.Module):
    """
    单个专家：两层前馈（等价于原先的 1x1 Conv FFN，但用 Linear 实现，便于逐 token 路由）
    输入/输出形状: [..., D] -> [..., D]
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_tokens, D]
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class MoEFeedForward(nn.Module):
    """
    轻量级 MoE 前馈（Top-1 Router，Switch-Transformer 风格，**无**容量限制）
    - Router: Linear(d_model -> num_experts), softmax 后 argmax 选专家
    - Experts: num_experts 个 ExpertFFN
    - 形状：输入 [B, L, D]，输出 [B, L, D]
    - 兼容 TimerLayer.enable_moe() 的 init_from_conv1x1(...)
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        assert num_experts >= 1
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts

        # router 初始化成近似均匀（避免一开始就偏置某个专家）
        self.router = nn.Linear(d_model, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.experts = nn.ModuleList([
            _ExpertFFN(d_model, d_ff, dropout, activation) for _ in range(num_experts)
        ])

    @torch.no_grad()
    def init_from_conv1x1(
        self,
        conv1: nn.Conv1d,
        conv2: nn.Conv1d,
        noise_std: float = 0.0,
    ):
        """
        将原 Dense FFN 的 1x1 Conv 权重复制到所有专家中：
          conv1: [d_ff, d_model, 1]  -> expert.fc1.weight [d_ff, d_model]
          conv2: [d_model, d_ff, 1]  -> expert.fc2.weight [d_model, d_ff]
        expert0 完全拷贝，其他专家可加入微小噪声以打破对称。
        """
        # 取出等价 Linear 的权重/偏置
        W1 = conv1.weight.data.squeeze(-1).clone()        # [d_ff, d_model]
        b1 = conv1.bias.data.clone() if conv1.bias is not None else None
        W2 = conv2.weight.data.squeeze(-1).clone()        # [d_model, d_ff]
        b2 = conv2.bias.data.clone() if conv2.bias is not None else None

        for idx, expert in enumerate(self.experts):
            expert.fc1.weight.data.copy_(W1)
            expert.fc2.weight.data.copy_(W2)
            if b1 is not None:
                expert.fc1.bias.data.copy_(b1)
            if b2 is not None:
                expert.fc2.bias.data.copy_(b2)

            # 其余专家添加极小扰动，促进分化
            if noise_std > 0 and idx > 0:
                expert.fc1.weight.data.add_(torch.randn_like(expert.fc1.weight) * noise_std)
                if b1 is not None:
                    expert.fc1.bias.data.add_(torch.randn_like(expert.fc1.bias) * noise_std)
                expert.fc2.weight.data.add_(torch.randn_like(expert.fc2.weight) * noise_std)
                if b2 is not None:
                    expert.fc2.bias.data.add_(torch.randn_like(expert.fc2.bias) * noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  ->  y: [B, L, D]
        Top-1 路由：每个 token 只进一个专家，计算量 ≈ Dense FFN
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)  # [N, D], N=B*L

        # 计算路由
        logits = self.router(x_flat)              # [N, E]
        top1 = torch.argmax(logits, dim=-1)       # [N]
        y_flat = torch.zeros_like(x_flat)         # 预分配输出

        # 逐专家收集/处理/回填
        for e in range(self.num_experts):
            mask = (top1 == e)
            n_tok = mask.sum().item()
            if n_tok == 0:
                continue
            xe = x_flat[mask]                     # [n_tok, D]
            ye = self.experts[e](xe)              # [n_tok, D]
            y_flat[mask] = ye

        y = y_flat.view(B, L, D)
        return y

