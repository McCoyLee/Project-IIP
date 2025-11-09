# layers/Transformer_EncDec.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.moe_ffn import MoEFeedForward  # 现有 MoE（保留）

# 可选导入：共享+路由 Top-2（已提供）
try:
    from layers.moe_shared_routed import SharedRoutedMoE  # 新增
    _HAS_SHARED_ROUTED = True
except Exception:
    SharedRoutedMoE = None
    _HAS_SHARED_ROUTED = False


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            extra_ctx=extra_ctx,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, extra_ctx=None):
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None, extra_ctx=extra_ctx
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta, extra_ctx=extra_ctx
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderOnlyLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class TimerLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 use_moe: bool = False, num_experts: int = 8, moe_init_noise: float = 0.0,
                 moe_topk: int = 1, moe_capacity_factor: float = 1.25,
                 moe_gate_temp: float = 1.0, moe_gate_noise_std: float = 0.0,
                 moe_lb_alpha: float = 0.0, moe_imp_alpha: float = 0.0,
                 moe_zloss_beta: float = 0.0, moe_entropy_reg: float = 0.0,
                 moe_learnable_temp: bool = False, moe_gate_dropout: float = 0.0,
                 moe_kl_alpha: float = 0.0,
                 # === 首选参数命名（shared+routed） ===
                 moe_mode: str = 'vanilla',          # 'vanilla' or 'shared_routed_top2'
                 moe_n_shared: int = 0,
                 moe_r_shared: int = 2,
                 moe_r_routed: int = 3,
                 moe_router_tau: float = 1.5,
                 moe_router_noisy_std: float = 1.0,
                 moe_dropless: bool = False,
                 # === 兼容你在 timer_xl.py 里可能已使用的命名 ===
                 moe_shared_experts: int | None = None,
                 moe_routed_experts: int | None = None):
        super(TimerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # === MoE 基本配置 ===
        self.use_moe_flag = use_moe
        self.moe_init_noise = moe_init_noise
        self._is_moe_enabled = False

        # 训练侧损失权重（shared+routed 需要）
        self._lb_alpha = moe_lb_alpha
        self._z_beta = moe_zloss_beta

        # 统一映射参数（兼容两种命名）
        self._moe_mode = moe_mode
        # 兼容：若传入旧名则覆盖
        if moe_shared_experts is not None:
            moe_n_shared = int(moe_shared_experts)
        routed_E = num_experts
        if moe_routed_experts is not None:
            routed_E = int(moe_routed_experts)

        self._moe_n_shared = int(moe_n_shared)
        self._moe_r_shared = int(moe_r_shared)
        self._moe_r_routed = int(moe_r_routed)
        self._moe_router_tau = float(moe_router_tau)
        self._moe_router_noisy_std = float(moe_router_noisy_std)
        self._moe_dropless = bool(moe_dropless)

        # 构建 FFN / MoE
        if self.use_moe_flag:
            use_shared_routed = (_HAS_SHARED_ROUTED and (
                    self._moe_mode == 'shared_routed_top2' or self._moe_n_shared > 0
            ))
            if use_shared_routed:
                # 共享 + 路由 Top-2
                self._moe = SharedRoutedMoE(
                    d_model=d_model,
                    n_shared=self._moe_n_shared,
                    n_routed=routed_E,
                    r_shared=self._moe_r_shared,
                    r_routed=self._moe_r_routed,
                    expert_dropout=dropout,
                    router_noisy_std=self._moe_router_noisy_std,
                    router_tau=self._moe_router_tau,
                    capacity_factor=None if self._moe_dropless else moe_capacity_factor,
                    use_dropless=self._moe_dropless,
                )
            else:
                # 回退 vanilla MoE
                self._moe = MoEFeedForward(
                    d_model=d_model, d_ff=d_ff, num_experts=routed_E,
                    dropout=dropout, activation=activation,
                    top_k=moe_topk, capacity_factor=moe_capacity_factor,
                    gate_temp=moe_gate_temp, gate_noise_std=moe_gate_noise_std,
                    lb_alpha=moe_lb_alpha, imp_alpha=moe_imp_alpha,
                    zloss_beta=moe_zloss_beta, entropy_reg=moe_entropy_reg,
                    learnable_gate_temp=moe_learnable_temp,
                    gate_dropout=moe_gate_dropout,
                    kl_alpha=moe_kl_alpha
                )
        else:
            self._moe = None

        # 训练侧 aux 聚合
        self.register_buffer("_moe_aux_total", torch.tensor(0.0))

    @torch.no_grad()
    def enable_moe(self):
        """在训练开始前由外部调用，用 dense conv1/conv2 权重初始化 MoE（如实现）。"""
        if self._moe is None or self._is_moe_enabled:
            return
        if hasattr(self._moe, "init_from_conv1x1"):
            self._moe.init_from_conv1x1(self.conv1, self.conv2, noise_std=self.moe_init_noise)
        self._is_moe_enabled = True

    def _dense_ffn(self, y: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        self._moe_aux_total.zero_()
        return y

    def _moe_ffn(self, y: torch.Tensor) -> torch.Tensor:
        out = self._moe(y)
        # 两种形式：
        # (A) vanilla: 返回 y，并在 self._moe._last_aux['total'] 提供标量
        # (B) shared_routed: 返回 (y, aux)，需要我们计算 total
        if isinstance(out, tuple):
            y_out, aux = out
            total = 0.0
            if self._lb_alpha > 0 and isinstance(aux, dict) and ('f_i' in aux) and ('p_i' in aux):
                f_i = aux['f_i']; p_i = aux['p_i']
                if torch.is_tensor(f_i) and torch.is_tensor(p_i):
                    n = f_i.numel()
                    total = total + self._lb_alpha * n * torch.sum(f_i * p_i)
            if self._z_beta > 0 and isinstance(aux, dict) and ('router_logits' in aux):
                z = torch.logsumexp(aux['router_logits'], dim=-1)
                total = total + self._z_beta * torch.mean(z ** 2)

            if not torch.is_tensor(total):
                total = torch.tensor(float(total), device=self._moe_aux_total.device)
            self._moe_aux_total.copy_(total.detach())
            return y_out
        else:
            y_out = out
            aux = getattr(self._moe, "_last_aux", None)
            if isinstance(aux, dict) and "total" in aux:
                val = aux["total"]
                if not torch.is_tensor(val):
                    val = torch.tensor(float(val), device=self._moe_aux_total.device)
                self._moe_aux_total.copy_(val.detach())
            else:
                self._moe_aux_total.zero_()
            return y_out

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        new_x, attn = self.attention(
            x, x, x,
            n_vars=n_vars, n_tokens=n_tokens,
            attn_mask=attn_mask, tau=tau, delta=delta,
            extra_ctx=extra_ctx,
        )
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        if getattr(self, "_is_moe_enabled", False):
            y = self._moe_ffn(y)
        else:
            y = self._dense_ffn(y)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, extra_ctx=extra_ctx)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, extra_ctx=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class DecoderOnly(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(DecoderOnly, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, extra_ctx=extra_ctx)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TimerBlock(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimerBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None, extra_ctx=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, n_vars, n_tokens, tau=tau, delta=None, extra_ctx=extra_ctx)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, n_vars, n_tokens, attn_mask=attn_mask, tau=tau, delta=delta, extra_ctx=extra_ctx)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
