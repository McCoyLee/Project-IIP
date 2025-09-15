# models/timer_xl.py
import torch
import math
from torch import nn
from typing import Dict, Any

from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.unica_cov import UniCAFiLM
from models.revin import RevIN
from utils.adaptive_norm import AdaRevIN                     # <<< 新增
from layers.de_stationary_attention import DeStationaryAdapter  # <<< 新增


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting
    Paper: https://arxiv.org/abs/2410.04803
    GitHub: https://github.com/thuml/Timer-XL
    """
    def __init__(self, configs):
        super().__init__()
        # ---- patch tokenization ----
        self.input_token_len = configs.input_token_len
        self.input_token_stride = getattr(configs, 'input_token_stride', None)
        if self.input_token_stride is None:
            self.input_token_stride = self.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)

        # ---- flags ----
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm  # 与 RevIN/AdaRevIN 互斥；开 AdaRevIN/RevIN 时请关 use_norm

        # ---- AdaRevIN（优先） / RevIN / use_norm ----
        self.use_adanorm = getattr(configs, 'use_adanorm', False)
        if self.use_adanorm:
            self.adanorm = AdaRevIN(
                nvars=getattr(configs, 'n_vars', None),
                eps=1e-5, affine=True,
                learnable_alpha=getattr(configs, 'adanorm_alpha', 'per_channel'),
                learnable_beta=getattr(configs, 'adanorm_beta', 'scalar'),
                use_ema_detrend=getattr(configs, 'adanorm_use_ema', True),
                ema_gamma=getattr(configs, 'ema_gamma', 0.995),
            )
        else:
            self.adanorm = None

        self.revin = RevIN(
            num_features=getattr(configs, 'n_vars', None) if getattr(configs, 'revin', False) else None,
            affine=getattr(configs, 'revin_affine', True),
            eps=getattr(configs, 'revin_eps', 1e-5),
        ) if getattr(configs, 'revin', False) and (not self.use_adanorm) else None

        # ---- MoE 配置（透传到 TimerLayer；保持预训练兼容）----
        self.use_moe = getattr(configs, 'use_moe', False)
        self.num_experts = getattr(configs, 'num_experts', 8)
        self.moe_init_noise = getattr(configs, 'moe_init_noise', 0.0)

        # ---- UniCA 协变量同质化 + 轻量融合（可选）----
        self.use_unica = getattr(configs, 'use_unica', False)
        self.unica_stage = getattr(configs, 'unica_stage', 'post')  # 'pre' / 'post'
        self.target_channel = getattr(configs, 'target_channel', -1)
        if self.use_unica:
            self.unica = UniCAFiLM(
                d_model=configs.d_model,
                bottleneck=getattr(configs, 'unica_bottleneck', 128),
                input_token_len=configs.input_token_len,
                exclude_target=getattr(configs, 'unica_exclude_target', False),
                fusion=getattr(configs, 'unica_fusion', 'res_add'),
                gamma_scale=getattr(configs, 'unica_gamma_scale', 0.1),
                beta_scale=getattr(configs, 'unica_beta_scale', 0.05),
                res_scale=getattr(configs, 'unica_res_scale', 0.1),
                dropout=getattr(configs, 'unica_dropout', 0.0),
                init_gate_bias=getattr(configs, 'unica_init_gate_bias', -2.0),
                init_alpha_bias=getattr(configs, 'unica_init_alpha_bias', -2.0),
                smooth_gate_ks=getattr(configs, 'unica_smooth_gate_ks', 3),
                smooth_beta_ks=getattr(configs, 'unica_smooth_beta_ks', 3),
            )
        else:
            self.unica = None

        # ---- 主干块 ----
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                            d_model=configs.d_model,
                            num_heads=configs.n_heads,
                            covariate=configs.covariate,
                            flash_attention=configs.flash_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_moe=self.use_moe,
                    num_experts=self.num_experts,
                    moe_init_noise=self.moe_init_noise,
                    moe_topk=getattr(configs, 'moe_topk', 1),
                    moe_capacity_factor=getattr(configs, 'moe_capacity_factor', 1.25),
                    moe_gate_temp=getattr(configs, 'moe_gate_temp', 1.0),
                    moe_gate_noise_std=getattr(configs, 'moe_gate_noise_std', 0.0),
                    moe_lb_alpha=getattr(configs, 'moe_lb_alpha', 0.0),
                    moe_imp_alpha=getattr(configs, 'moe_imp_alpha', 0.0),
                    moe_zloss_beta=getattr(configs, 'moe_zloss_beta', 0.0),
                    moe_entropy_reg=getattr(configs, 'moe_entropy_reg', 0.0),
                    # <<< 新增三项透传 >>>
                    moe_learnable_temp=getattr(configs, 'moe_learnable_temp', False),
                    moe_gate_dropout=getattr(configs, 'moe_gate_dropout', 0.0),
                    moe_kl_alpha=getattr(configs, 'moe_kl_alpha', 0.0),
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ---- De-Stationary Attention 适配器（可选）----
        self.use_desta = getattr(configs, 'use_desta', False)
        if self.use_desta:
            self.desta = DeStationaryAdapter(
                nvars=getattr(configs, 'n_vars', None),
                nheads=configs.n_heads,
                hidden=128
            )
            # 安装到每一层 TimeAttention 上
            for m in self.modules():
                if hasattr(m, 'inner_attention') and hasattr(m.inner_attention, 'desta'):
                    m.inner_attention.desta = self.desta
        else:
            self.desta = None

        # ---- 输出头（token -> 原步长）----
        self.head = nn.Linear(configs.d_model, configs.output_token_len)

        # ---- Linear branch (DLinear-style residual) ----
        self.use_linear_branch = getattr(configs, 'use_linear_branch', False)
        if self.use_linear_branch:
            self.linear_head = nn.Linear(self.input_token_len, configs.output_token_len)
            init_gate = getattr(configs, 'linear_init_gate', -2.0)  # 初始偏关
            self.linear_gate = nn.Parameter(torch.tensor(float(init_gate)))
            self.linear_res_scale = getattr(configs, 'linear_res_scale', 0.2)

    def forecast(self, x, x_mark, y_mark):
        """
        x:      [B, L, C]
        x_mark: [B, L, M] 或 None（时间特征）
        y_mark: 兼容占位
        """
        # ---------------- 前置归一化 ----------------
        x_raw = x.clone()  # UniCA 使用原尺度
        revin_stats = None
        extra_ctx = {}

        if self.adanorm is not None:
            x, stats, _ = self.adanorm.forward_in(x)  # AdaRevIN
            extra_ctx['stats'] = stats
        elif self.revin is not None:
            x, revin_stats = self.revin(x, mode='norm')
        elif self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        B, L, C = x.shape
        x_for_cov = x_raw  # UniCA 用原尺度协变量

        # ---- 保证长度可被 patch 大小整除（容错兜底）----
        P = self.input_token_len
        L = x.size(1)
        if L % P != 0:
            Lp = (L // P) * P     # 向下取整
            x = x[:, :Lp, :]
            x_raw = x_raw[:, :Lp, :]
            if x_mark is not None:
                x_mark = x_mark[:, :Lp, :]

        # ---------------- Patch tokenization ----------------
        x_tok = x.permute(0, 2, 1)
        if self.input_token_stride != self.input_token_len:
            raise NotImplementedError("当前实现只支持非重叠 patch；如需 stride < len，请先实现 overlap-add 聚合。")
        x_tok = x_tok.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_stride)  # [B, C, N, P]
        N = x_tok.shape[2]

        # ---------------- Linear embed per token ----------------
        embed_out = self.embedding(x_tok)            # [B, C, N, D]
        embed_out = embed_out.reshape(B, C * N, -1)  # [B, C*N, D]

        # 【可选】前置融合
        if self.unica is not None and self.unica_stage == 'pre':
            embed_out = self.unica(embed_out, x_for_cov, x_mark, target_channel=self.target_channel)

        # ---------------- Timer-XL 主干 ----------------
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N, extra_ctx=extra_ctx)

        # 【推荐】后置融合
        if self.unica is not None and self.unica_stage == 'post':
            embed_out = self.unica(embed_out, x_for_cov, x_mark, target_channel=self.target_channel)

        # ---------------- 两路头 + 融合 ----------------
        dec_main = self.head(embed_out)  # [B, C*N, P]
        if self.use_linear_branch:
            lin_out = self.linear_head(x_tok)               # [B, C, N, P]
            lin_out = lin_out.reshape(B, C * N, -1)         # [B, C*N, P]
            gate = torch.sigmoid(self.linear_gate) * self.linear_res_scale
            dec_out = dec_main + gate * lin_out
        else:
            dec_out = dec_main

        # ---------------- 还原为序列 ----------------
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1).permute(0, 2, 1)  # [B, L, C]

        # ---------------- 后置反归一化 ----------------
        if self.adanorm is not None:
            dec_out = self.adanorm.forward_out(dec_out)
        elif self.revin is not None:
            dec_out = self.revin(dec_out, mode='denorm', stats=revin_stats)
        elif self.use_norm:
            dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)

    @torch.no_grad()
    def convert_dense_ffn_to_moe(self):
        if not self.use_moe:
            return
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                m.enable_moe()

    def moe_aux_loss(self) -> torch.Tensor:
        device = self.head.weight.device
        aux_vals = []
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                aux_val = None
                if hasattr(m, 'ffn') and hasattr(m.ffn, '_last_aux'):
                    last_aux = m.ffn._last_aux
                    if isinstance(last_aux, dict) and 'total_live' in last_aux:
                        aux_val = last_aux['total_live']
                    elif isinstance(last_aux, dict) and 'total' in last_aux:
                        aux_val = last_aux['total']
                if aux_val is None and hasattr(m, '_moe_aux_total'):
                    aux_val = m._moe_aux_total
                if aux_val is not None:
                    if not torch.is_tensor(aux_val):
                        aux_val = torch.tensor(float(aux_val), device=device)
                    aux_vals.append(aux_val)
        if not aux_vals:
            return torch.tensor(0.0, device=device)
        return torch.stack(aux_vals).mean()

    @torch.no_grad()
    def moe_metrics(self) -> Dict[str, Any]:
        device = self.head.weight.device
        acc: Dict[str, float] = {}
        cnt = 0

        def _add(k, v):
            acc[k] = acc.get(k, 0.0) + float(v)

        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                if hasattr(m, 'ffn') and hasattr(m.ffn, '_last_aux'):
                    aux = m.ffn._last_aux
                    if isinstance(aux, dict):
                        for k in ['total', 'assigned_frac', 'balance_switch', 'balance_kl', 'entropy', 'zloss']:
                            if k in aux:
                                v = aux[k]
                                v = v.item() if torch.is_tensor(v) else float(v)
                                _add(f"moe/{k}", v)
                        cnt += 1
                if hasattr(m, '_moe_aux_total'):
                    v = m._moe_aux_total
                    v = v.item() if torch.is_tensor(v) else float(v)
                    _add("moe/total", v)
                    if "moe/assigned_frac" not in acc:
                        acc["moe/assigned_frac"] = 1.0
                    cnt = max(cnt, 1)

        if cnt == 0:
            return {
                "moe/aux_total": 0.0,
                "moe/assigned_frac": 0.0,
                "moe/balance_switch": 0.0,
                "moe/balance_kl": 0.0,
                "moe/entropy": 0.0,
                "moe/zloss": 0.0,
            }

        return {
            "moe/aux_total": acc.get("moe/total", 0.0) / cnt,
            "moe/assigned_frac": acc.get("moe/assigned_frac", 0.0) / cnt,
            "moe/balance_switch": acc.get("moe/balance_switch", 0.0) / cnt,
            "moe/balance_kl": acc.get("moe/balance_kl", 0.0) / cnt,
            "moe/entropy": acc.get("moe/entropy", 0.0) / cnt,
            "moe/zloss": acc.get("moe/zloss", 0.0) / cnt,
        }
