# layers/SelfAttention_Family.py
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import repeat
from layers.Attn_Bias import BinaryAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerCovariateMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # 可由外部注入（De-Stationary Adapter）
        self.desta = None

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None, extra_ctx=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B,H,L,S]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        logits = scale * scores

        # De-Stationary 调制（若已注入且提供 stats）
        if getattr(self, 'desta', None) is not None and extra_ctx is not None and ('stats' in extra_ctx):
            logits = self.desta(logits, extra_ctx['stats'])

        A = self.dropout(torch.softmax(logits, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False,
                 d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(
            dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection,
            kwargs=dict(max_len=max_len), partial_factor=(0.0, 0.5),
        )
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)
        # 由外部模型注入（De-Stationary Adapter）
        self.desta = None

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None, extra_ctx=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars, device=queries.device)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars, device=queries.device), 'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        # 若启用 DeSta（且提供 stats），则不能走 sdpa（不便逐头缩放/偏置）
        use_flash = self.flash_attention and not (
                getattr(self, 'desta', None) is not None and extra_ctx is not None and ('stats' in extra_ctx)
        )

        if use_flash:
            V = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)  # [B,H,L,S]
            logits = scale * scores + attn_mask
            if getattr(self, 'desta', None) is not None and extra_ctx is not None and ('stats' in extra_ctx):
                logits = self.desta(logits, extra_ctx['stats'])
            A = self.dropout(torch.softmax(logits, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None, extra_ctx=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask,
            n_vars=n_vars, n_tokens=n_tokens, tau=tau, delta=delta,
            extra_ctx=extra_ctx,   # 关键：把上层传来的 stats 继续往下传
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
