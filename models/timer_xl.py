import torch
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.unica_cov import UniCAFiLM
from models.revin import RevIN


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting

    Paper: https://arxiv.org/abs/2410.04803
    GitHub: https://github.com/thuml/Timer-XL

    Citation:
    @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm  # 与 RevIN 二选一，建议开 RevIN 时关闭 use_norm

        # === RevIN（可逆实例归一化）===
        # 注意：num_features 需与数据的通道数一致；建议通过 --n_vars 显式传入
        self.revin = RevIN(
            num_features=getattr(configs, 'n_vars', None) if getattr(configs, 'revin', False) else None,
            affine=getattr(configs, 'revin_affine', True),
            eps=getattr(configs, 'revin_eps', 1e-5),
        ) if getattr(configs, 'revin', False) else None

        # === MoE 配置（透传到 TimerLayer；保持预训练兼容）===
        self.use_moe = getattr(configs, 'use_moe', False)
        self.num_experts = getattr(configs, 'num_experts', 8)
        self.moe_init_noise = getattr(configs, 'moe_init_noise', 0.0)

        # === UniCA 协变量同质化 + 轻量融合（可选）===
        self.use_unica = getattr(configs, 'use_unica', False)
        self.target_channel = getattr(configs, 'target_channel', -1)  # ETT: OT 常为 6；Weather 需按列索引设置
        if self.use_unica:
            self.unica = UniCAFiLM(
                d_model=configs.d_model,
                bottleneck=getattr(configs, 'unica_bottleneck', 128),
                input_token_len=configs.input_token_len,
                exclude_target=getattr(configs, 'unica_exclude_target', False),
                fusion=getattr(configs, 'unica_fusion', 'film_gate'),
            )
        else:
            self.unica = None

        # === 主干块 ===
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
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # === 输出头（token -> 原步长）===
        self.head = nn.Linear(configs.d_model, configs.output_token_len)

    def forecast(self, x, x_mark, y_mark):
        """
        x:      [B, L, C]
        x_mark: [B, L, M] 或 None（时间特征）
        y_mark: 兼容占位
        """
        # ----------------
        # 前置归一化（RevIN 优先；否则用 use_norm）
        # ----------------
        revin_stats = None
        if self.revin is not None:
            # RevIN: (norm, stats)
            x, revin_stats = self.revin(x, mode='norm')
        elif self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        B, L, C = x.shape
        x_for_cov = x  # UniCA 用的原序列分支（已做与上面一致的归一化）

        # ----------------
        # Tokenization: [B, L, C] -> [B, C, L] -> unfold -> [B, C, N, P]
        # ----------------
        x_tok = x.permute(0, 2, 1)
        x_tok = x_tok.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x_tok.shape[2]

        # ----------------
        # Linear embed per token: [B, C, N, P] --Linear--> [B, C, N, D] -> [B, C*N, D]
        # ----------------
        embed_out = self.embedding(x_tok)
        embed_out = embed_out.reshape(B, C * N, -1)

        # ----------------
        # UniCA 融合（FiLM/gate 等），轻量 + 与预训练兼容
        # ----------------
        if self.unica is not None:
            embed_out = self.unica(embed_out, x_for_cov, x_mark, target_channel=self.target_channel)

        # ----------------
        # Timer-XL 主干
        # ----------------
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)

        # ----------------
        # 还原为序列： [B, C*N, D] --head--> [B, C*N, P] -> [B, C, N, P] -> [B, C, L] -> [B, L, C]
        # ----------------
        dec_out = self.head(embed_out)
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        dec_out = dec_out.permute(0, 2, 1)

        # ----------------
        # 后置反归一化（与前置对称）
        # ----------------
        if self.revin is not None:
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
        """
        在加载原始 Timer-XL 预训练权重后调用，将每层 1x1 FFN 转为 MoE（保持 expert0 权重对齐）。
        需要 TimerLayer 实现 enable_moe()（已在 Transformer_EncDec.py 中添加）。
        """
        if not self.use_moe:
            return
        for m in self.modules():
            if isinstance(m, TimerLayer) and getattr(m, 'use_moe_flag', False):
                m.enable_moe()
