import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any

from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.unica_cov import UniCAFiLM
from models.revin import RevIN
from utils.adaptive_norm import AdaRevIN, TokenAdaptiveNorm
from layers.de_stationary_attention import DeStationaryAdapter

# ★ PatchTST 风格前端与聚合头
from layers.patch import PatchEmbed1D, PatchUnembedHead

# ★ FIR-MoE / TAN 共享：per-token 频率特征提取
from layers.freq_features import PerTokenFreqExtractor, extract_patch_freq_from_sequence

# ============================================================
# ★★★ 内联实现：AFS-Gate（自适应频率选择门） ★★★
# 为了便于直接替换，本文件内联 AFS，避免额外新建 layers/afs_gate.py
# ============================================================
class _TriangularBank(nn.Module):
    """K个可学习三角带通窗，频率轴归一到 [0,1]。"""
    def __init__(self, K: int, init_centers=None, init_width=0.25):
        super().__init__()
        self.K = K
        if init_centers is None:
            init_centers = torch.linspace(0.05, 0.95, K)
        self.centers = nn.Parameter(init_centers.clone())            # 学习中心 c_k
        self.widths  = nn.Parameter(torch.full((K,), float(init_width)))  # 学习宽度 w_k

    def forward(self, F_bins: int, device=None, dtype=None):
        f = torch.linspace(0, 1, F_bins, device=device, dtype=dtype)    # 频率坐标
        c = torch.sigmoid(self.centers).unsqueeze(1)                    # [K,1]
        w = F.softplus(self.widths).unsqueeze(1) + 1e-6                 # [K,1] > 0
        W = torch.relu(1.0 - torch.abs(f.unsqueeze(0) - c) / w)         # [K,F] 三角窗
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-9)                    # 每带归一化
        return W

class AdaptiveFreqGate(nn.Module):
    """
    AFS-Gate：rFFT→K带能量→alpha=softmax→频谱掩码→iFFT→FiLM/残差
    输入:  x [B, L, D]
    输出:  y [B, L, D], alpha [B, K]
    """
    def __init__(self, d_model: int, K: int = 8, use_film: bool = True,
                 res_scale: float = 0.2, init_gamma: float = 0.0):
        super().__init__()
        self.K = K
        self.use_film = use_film
        self.res_scale = res_scale

        self.bank = _TriangularBank(K)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(K, K), nn.ReLU(inplace=True),
            nn.Linear(K, K)
        )
        self.gamma = nn.Parameter(torch.tensor(init_gamma))  # 频谱掩码强度

        if use_film:
            self.film = nn.Linear(K, 2 * d_model)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xt = x.transpose(1, 2)                       # [B, D, L]
        Xf = torch.fft.rfft(xt, dim=-1)              # [B, D, F]
        Fbins = Xf.shape[-1]
        W = self.bank(Fbins, device=Xf.device, dtype=Xf.real.dtype)  # [K, F]

        power = (Xf.real**2 + Xf.imag**2)            # [B, D, F]
        band_power = torch.einsum('bdf,kf->bdk', power, W)  # [B, D, K]
        band_power = band_power.mean(dim=1)          # [B, K]

        alpha_logits = self.alpha_mlp(band_power)    # [B, K]
        alpha = torch.softmax(alpha_logits, dim=-1)  # [B, K]

        M = torch.einsum('bk,kf->bf', alpha, W).unsqueeze(1)  # [B,1,F]
        Xf_masked = Xf * (1.0 + self.gamma * M)
        xt_filt = torch.fft.irfft(Xf_masked, n=L, dim=-1)     # [B, D, L]
        x_filt = xt_filt.transpose(1, 2)                      # [B, L, D]

        if self.use_film:
            film_params = self.film(alpha)        # [B, 2D]
            scale, bias = film_params.chunk(2, dim=-1)
            y = x_filt * (1 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        else:
            y = x_filt

        out = x + self.res_scale * y              # 残差融合（恒等起步稳）
        return out, alpha
# ======================= AFS 结束 ============================


class Model(nn.Module):
    """
    Timer-XL with PatchTST-style tokenization (+ optional CI backbone and heads)
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # ====== Patch 参数 ======
        self.patch_len = configs.input_token_len
        self.patch_stride = getattr(configs, 'input_token_stride', None)
        if self.patch_stride is None:
            self.patch_stride = self.patch_len

        # ====== CI 主干与预测头类型 ======
        self.ci_backbone = getattr(configs, 'ci_backbone', False)
        self.head_type = getattr(configs, 'head_type', 'last')  # 'last' | 'mean' | 'tokenwise'
        if self.ci_backbone and self.head_type == 'tokenwise':
            print("[Timer-XL] CI 模式下不推荐 tokenwise 头，已强制改为 'last'")
            self.head_type = 'last'

        # ====== 归一化系列 ======
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

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

        # ====== 禁用 Revin ======
        self.revin = None  # 禁用 Revin

        # ====== UniCA ======
        self.use_unica = getattr(configs, 'use_unica', False)
        self.unica_stage = getattr(configs, 'unica_stage', 'post')
        self.target_channel = getattr(configs, 'target_channel', -1)
        if self.use_unica:
            self.unica = UniCAFiLM(
                d_model=configs.d_model,
                bottleneck=getattr(configs, 'unica_bottleneck', 128),
                input_token_len=self.patch_len,
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

        # ====== AFS-Gate（可选）======
        self.use_afs_gate = getattr(configs, 'use_afs_gate', False)
        self.afs_place = getattr(configs, 'afs_place', 'block_post')
        self.afs_as_moe_prior = getattr(configs, 'afs_as_moe_prior', False)
        if self.use_afs_gate:
            self.afs_gate = AdaptiveFreqGate(
                d_model=configs.d_model,
                K=getattr(configs, 'afs_bands', 8),
                res_scale=getattr(configs, 'afs_res_scale', 0.2),
                init_gamma=getattr(configs, 'afs_init_gamma', 0.0),
            )
        else:
            self.afs_gate = None
        self._freq_prior = None  # 缓存 alpha，供需要时读取（MoE 先验等）

        # ====== FIR-MoE（核心贡献：频率引导专家路由）======
        self.use_fir_moe = getattr(configs, 'use_fir_moe', False)
        self.fir_bands = int(getattr(configs, 'fir_bands', 8))
        self.fir_spec_alpha = float(getattr(configs, 'fir_spec_alpha', 0.01))
        self.fir_ortho_alpha = float(getattr(configs, 'fir_ortho_alpha', 0.0))

        # ====== TAN（补充贡献：Token 级自适应归一化）======
        self.use_tan = getattr(configs, 'use_tan', False)
        self.tan_freq_cond = getattr(configs, 'tan_freq_cond', True)
        self.tan_bands = int(getattr(configs, 'tan_bands', 8))

        # FIR-MoE 和 TAN 共享的 per-token 频率提取器（若任一启用则构建）
        self._need_freq_features = self.use_fir_moe or self.use_tan
        if self._need_freq_features:
            # FIR-MoE 和 TAN 可以共享 K；若不同则优先使用较大值
            K_shared = max(self.fir_bands if self.use_fir_moe else 0,
                           self.tan_bands if self.use_tan else 0)
            self._freq_K = K_shared
            self.per_token_freq = PerTokenFreqExtractor(K=K_shared)
        else:
            self._freq_K = 0
            self.per_token_freq = None

        if self.use_tan:
            if self.use_adanorm:
                print("[Timer-XL] 警告：use_tan 与 use_adanorm 同时启用；TAN 将取代 AdaRevIN 归一化路径")
                self.adanorm = None
                self.use_adanorm = False
            self.tan = TokenAdaptiveNorm(
                n_vars=getattr(configs, 'n_vars', None),
                patch_len=self.patch_len,
                stride=self.patch_stride,
                freq_dim=self._freq_K if self.tan_freq_cond else 0,
                use_freq_cond=self.tan_freq_cond,
                eps=1e-5,
                affine=True,
            )
        else:
            self.tan = None

        # ====== 主干（共享）======
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
                    use_moe=getattr(configs, 'use_moe', False),
                    num_experts=getattr(configs, 'num_experts', 8),
                    moe_init_noise=getattr(configs, 'moe_init_noise', 0.0),
                    moe_topk=getattr(configs, 'moe_topk', 1),
                    moe_capacity_factor=getattr(configs, 'moe_capacity_factor', 1.25),
                    moe_gate_temp=getattr(configs, 'moe_gate_temp', 1.0),
                    moe_gate_noise_std=getattr(configs, 'moe_gate_noise_std', 0.0),
                    moe_lb_alpha=getattr(configs, 'moe_lb_alpha', 0.0),
                    moe_imp_alpha=getattr(configs, 'moe_imp_alpha', 0.0),
                    moe_zloss_beta=getattr(configs, 'moe_zloss_beta', 0.0),
                    moe_entropy_reg=getattr(configs, 'moe_entropy_reg', 0.0),

                    moe_mode=getattr(configs, 'moe_mode', 'vanilla'),             # 'vanilla' 或 'shared_routed_top2'
                    moe_n_shared=getattr(configs, 'moe_n_shared', 0),             # 共享专家个数
                    moe_r_shared=getattr(configs, 'moe_r_shared', 2),             # 共享专家扩展倍数 r_shared
                    moe_r_routed=getattr(configs, 'moe_r_routed', 3),             # 路由专家扩展倍数 r_routed
                    moe_router_tau=getattr(configs, 'moe_router_tau', 1.5),
                    moe_router_noisy_std=getattr(configs, 'moe_router_noisy_std', 1.0),
                    moe_dropless=getattr(configs, 'moe_dropless', False),

                    # 方式B（兼容你先前命名）：只给这两个也行
                    moe_shared_experts=getattr(configs, 'moe_shared_experts', None),  # e.g. 2
                    moe_routed_experts=getattr(configs, 'moe_routed_experts', None),  # e.g. 10

                    # FIR-MoE（频率引导专家路由）
                    use_fir_moe=self.use_fir_moe,
                    fir_freq_dim=(self._freq_K if self.use_fir_moe else 0),
                    fir_spec_alpha=self.fir_spec_alpha,
                    fir_ortho_alpha=self.fir_ortho_alpha,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ====== De-Stationary（可选）======
        self.use_desta = getattr(configs, 'use_desta', False)
        if self.use_desta:
            self.desta = DeStationaryAdapter(
                nvars=getattr(configs, 'n_vars', None),
                nheads=configs.n_heads,
                hidden=128
            )
            for m in self.modules():
                if hasattr(m, 'inner_attention') and hasattr(m.inner_attention, 'desta'):
                    m.inner_attention.desta = self.desta
        else:
            self.desta = None

        # ====== Patch 前端 + 预测头 ======
        self.patch = PatchEmbed1D(
            patch_len=self.patch_len,
            stride=self.patch_stride,
            d_model=configs.d_model,
            add_pos=True,
            pos_learnable=False,
        )
        self.head = PatchUnembedHead(
            d_model=configs.d_model,
            pred_len=configs.output_token_len,
            head_type=self.head_type,
            token_out_len=configs.output_token_len if self.head_type == 'tokenwise' else None,
            stride_equals_patch=(self.patch_stride == self.patch_len),
        )

        if getattr(configs, 'use_moe', False):
            self.convert_dense_ffn_to_moe()

    # ---------- 归一化 ----------
    def _compute_n_tokens(self, L: int) -> int:
        """Analytically derive the number of patch tokens for a sequence of length L."""
        L_eff = max(int(L), int(self.patch_len))
        return (L_eff - int(self.patch_len)) // int(self.patch_stride) + 1

    def _pre_norm(self, x, freq_all=None, n_tokens=None):
        x_raw = x
        # TAN 优先：token 级自适应归一化
        if self.tan is not None:
            if n_tokens is None:
                n_tokens = self._compute_n_tokens(x.shape[1])
            x_norm, tan_ctx = self.tan.forward_in(x, freq_all, n_tokens)
            return x_norm, x_raw, ("tan", tan_ctx)
        if self.adanorm is not None:
            x, stats, _ = self.adanorm.forward_in(x)
            return x, x_raw, ("adanorm", stats)
        if self.revin is not None:
            x, revin_stats = self.revin(x, mode='norm')
            return x, x_raw, ("revin", revin_stats)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev
            return x, x_raw, ("stat", (means, stdev))
        return x, x_raw, None

    def _post_denorm(self, y, ctx, freq_features_last=None):
        if ctx is None:
            return y
        kind, obj = ctx
        if kind == "tan":
            return self.tan.forward_out(y, obj, freq_features_last=freq_features_last)
        if kind == "adanorm":
            return self.adanorm.forward_out(y)
        if kind == "revin":
            return self.revin(y, mode='denorm', stats=obj)
        if kind == "stat":
            means, stdev = obj
            return y * stdev + means
        return y

    def forecast(self, x, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        return:
          - 若 head_type in {'last','mean'}: [B, H, C]
          - 若 head_type == 'tokenwise'    : [B, L, C] (仅 stride==patch_len)
        """
        B, L, C = x.shape

        # ---------- 预先抽取 per-token 频率特征（供 TAN 前置归一化与 FIR-MoE 路由共享）----------
        freq_all_bc = None   # [B*C, N, K] 从 raw x 抽取
        N_pre = self._compute_n_tokens(L)
        if self._need_freq_features and self.per_token_freq is not None:
            # 在**原始**输入上抽取频率特征（未归一化），以保证物理频率语义与 TAN/FIR 一致
            freq_all_bc = extract_patch_freq_from_sequence(
                x_raw=x,
                patch_len=self.patch_len,
                stride=self.patch_stride,
                extractor=self.per_token_freq,
                n_tokens=N_pre,
            )  # [B*C, N, K]

        # ---------- 归一化（TAN 或传统路径）----------
        x, x_raw, ctx = self._pre_norm(x, freq_all=freq_all_bc, n_tokens=N_pre)

        # ---------- Patch tokenize ----------
        tokens, meta = self.patch(x)     # tokens: [B*C, N, d], meta=(B,C,N)
        Bm, Cm, N = meta
        assert Bm == B and Cm == C
        # 若解析 N 与真实 patch 后的 N 不一致（例如边界 padding 行为差异），重新抽取以对齐
        if freq_all_bc is not None and freq_all_bc.shape[1] != N:
            freq_all_bc = extract_patch_freq_from_sequence(
                x_raw=x_raw,
                patch_len=self.patch_len,
                stride=self.patch_stride,
                extractor=self.per_token_freq,
                n_tokens=N,
            )

        # ---------- 构造 extra_ctx（FIR-MoE 使用）----------
        moe_extra_ctx = None
        if freq_all_bc is not None and self.use_fir_moe:
            if self.ci_backbone:
                # CI 主干：tokens 为 [B*C, N, d]，freq 也为 [B*C, N, K]
                moe_extra_ctx = {"freq_features": freq_all_bc}
            else:
                # 非 CI：将 tokens reshape 成 [B, C*N, d]，freq 也同构
                freq_for_moe = freq_all_bc.view(B, C, N, -1).reshape(B, C * N, -1)
                moe_extra_ctx = {"freq_features": freq_for_moe}

        # ---------- （可选）UniCA 前置 ----------
        if self.use_unica and self.unica_stage == 'pre':
            z4u = tokens.view(B, C, N, -1).reshape(B, C * N, -1)
            z4u = self.unica(z4u, x_raw, x_mark, target_channel=self.target_channel)
            tokens = z4u.view(B, C, N, -1).view(B * C, N, -1)

        # ---------- （可选）AFS block_pre ----------
        self._freq_prior = None
        if self.afs_gate is not None and self.afs_place == 'block_pre':
            t_in = tokens.view(B, C * N, -1)              # [B, C*N, d] 走一遍时域AFS更自然
            t_in, alpha = self.afs_gate(t_in)             # [B, C*N, d], [B, K]
            tokens = t_in.view(B, C, N, -1).view(B * C, N, -1)
            if self.afs_as_moe_prior:
                self._freq_prior = alpha.detach()         # 缓存；需要时外部可读取

        # ---------- 编码器 ----------
        attns = None
        if self.ci_backbone:
            # CI：batch = B*C, len = N
            ret = self.blocks(tokens, n_vars=1, n_tokens=N, extra_ctx=moe_extra_ctx)
            if isinstance(ret, tuple):
                z, attns = ret
            else:
                z = ret
        else:
            z_in = tokens.view(B, C, N, -1).reshape(B, C * N, -1)
            ret = self.blocks(z_in, n_vars=C, n_tokens=N, extra_ctx=moe_extra_ctx)
            if isinstance(ret, tuple):
                z_in, attns = ret
            else:
                z_in = ret
            z = z_in.view(B, C, N, -1).view(B * C, N, -1)

        assert isinstance(z, torch.Tensor) and z.dim() == 3 and z.shape[0] == B * C and z.shape[1] == N, \
            f"[shapes] z={type(z)} {getattr(z,'shape',None)}, expected (B*C,N,d)"

        # ---------- （可选）UniCA 后置 ----------
        if self.use_unica and self.unica_stage == 'post':
            if self.ci_backbone:
                pass
            else:
                z4u = z.view(B, C, N, -1).reshape(B, C * N, -1)
                z4u = self.unica(z4u, x_raw, x_mark, target_channel=self.target_channel)
                z = z4u.view(B, C, N, -1).view(B * C, N, -1)

        # ---------- （可选）AFS block_post ----------
        if self.afs_gate is not None and self.afs_place == 'block_post':
            z_in = z.view(B, C, N, -1).reshape(B, C * N, -1)    # [B, C*N, d]
            z_out, alpha = self.afs_gate(z_in)                  # [B, C*N, d], [B, K]
            z = z_out.view(B, C, N, -1).view(B * C, N, -1)
            if self.afs_as_moe_prior:
                self._freq_prior = alpha.detach()

        # ---------- 预测头 ----------
        out = self.head(z, meta)  # 'last/mean' -> (B,C,H) ; 'tokenwise' -> (B,L,C)

        if self.head_type in ("last", "mean"):
            out = out.transpose(1, 2)  # (B,C,H) -> (B,H,C)

            # CI + post-UniCA：在 (B,H,C) 上做轻量跨变量融合（可选）
            if self.use_unica and self.unica_stage == 'post' and self.ci_backbone:
                if not hasattr(self, 'post_ci_fuse'):
                    self.post_ci_fuse = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=True).to(out.device)
                elif self.post_ci_fuse.weight.device != out.device:
                    self.post_ci_fuse = self.post_ci_fuse.to(out.device)
                out = out.permute(0, 2, 1)     # (B,H,C) -> (B,C,H)
                out = self.post_ci_fuse(out)   # (B,C,H)
                out = out.permute(0, 2, 1)     # (B,C,H) -> (B,H,C)

        # ---------- 反归一化 ----------
        freq_last_for_tan = None
        if self.tan is not None and freq_all_bc is not None:
            # 取最后一个输入 token 的频率特征：[B*C, K] -> [B, C, K]
            last_k = freq_all_bc[:, -1, :]
            freq_last_for_tan = last_k.view(B, C, -1)
        out = self._post_denorm(out, ctx, freq_features_last=freq_last_for_tan)

        if self.output_attention:
            return out, attns
        return out

    def forward(self, x, x_mark=None, y_mark=None):
        return self.forecast(x, x_mark, y_mark)

    # ====== MoE 工具保持不变 ======
    @torch.no_grad()
    def convert_dense_ffn_to_moe(self):
        if not getattr(self.configs, 'use_moe', False):
            return
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                m.enable_moe()

    def moe_aux_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        aux_vals = []
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                aux = None
                if hasattr(m, 'ffn') and hasattr(m.ffn, '_last_aux'):
                    last_aux = m.ffn._last_aux
                    if isinstance(last_aux, dict) and 'total_live' in last_aux:
                        aux = last_aux['total_live']
                    elif isinstance(last_aux, dict) and 'total' in last_aux:
                        aux = last_aux['total']
                if aux is None and hasattr(m, '_moe_aux_total'):
                    aux = m._moe_aux_total
                if aux is not None:
                    aux_vals.append(torch.as_tensor(aux, device=device, dtype=torch.float32))
        if not aux_vals:
            return torch.tensor(0.0, device=device)
        return torch.stack(aux_vals).mean()

    @torch.no_grad()
    def moe_metrics(self) -> Dict[str, Any]:
        acc, cnt = {}, 0
        def _add(k, v):
            acc[k] = acc.get(k, 0.0) + float(v)
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                if hasattr(m, 'ffn') and hasattr(m.ffn, '_last_aux'):
                    aux = m.ffn._last_aux
                    if isinstance(aux, dict):
                        for k in ['total', 'assigned_frac', 'balance_switch', 'balance_kl', 'entropy', 'zloss']:
                            if k in aux:
                                vv = aux[k]
                                _add(f"moe/{k}", vv.item() if torch.is_tensor(vv) else float(vv))
                        cnt += 1
                if hasattr(m, '_moe_aux_total'):
                    vv = m._moe_aux_total
                    _add("moe/total", vv.item() if torch.is_tensor(vv) else float(vv))
                    if "moe/assigned_frac" not in acc:
                        acc["moe/assigned_frac"] = 1.0
                    cnt = max(cnt, 1)
        if cnt == 0:
            return {k: 0.0 for k in [
                "moe/aux_total","moe/assigned_frac","moe/balance_switch","moe/balance_kl","moe/entropy","moe/zloss"
            ]}
        return {
            "moe/aux_total": acc.get("moe/total", 0.0) / cnt,
            "moe/assigned_frac": acc.get("moe/assigned_frac", 0.0) / cnt,
            "moe/balance_switch": acc.get("moe/balance_switch", 0.0) / cnt,
            "moe/balance_kl": acc.get("moe/balance_kl", 0.0) / cnt,
            "moe/entropy": acc.get("moe/entropy", 0.0) / cnt,
            "moe/zloss": acc.get("moe/zloss", 0.0) / cnt,
        }
