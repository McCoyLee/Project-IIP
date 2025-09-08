主要改动：
layers/unica_cov.py：UniCA 协变量自适应模块；支持 FiLM 门控/残差融合、pre/post 两种插入点，含瓶颈维度与残差缩放等可调参数。

layers/moe_ffn.py：把原 FFN 换成 MoE 专家层；支持 Top-k 路由、capacity factor、gate temperature，并输出 负载均衡等 aux loss。

layers/Transformer_EncDec.py：把 Dense FFN 抽象成接口，--use_moe 时自动切到 MoE；同时在 Block 内暴露 pre/post hooks 接 UniCA；对 Norm/残差路径做了细化，减小数值漂移。

models/timer_xl.py：在每个 Block 的 FFN 位置可切换 MoE，并汇总回传 MoE auxiliary load-balancing loss；按 pre/post 插入 UniCA，--unica_fusion 选择 FiLM 或 res_add。

run.py：新增整套开关与参数：--use_unica 系（如 --unica_bottleneck/--unica_fusion/--unica_stage/--unica_res_scale）、--use_moe 系（如 --num_experts/--moe_topk/--moe_capacity_factor/--moe_gate_temp）；保留并透传 ReVIN（--revin/--revin_affine/--revin_eps）；加入 单变量评测（--eval_target_only --target_channel）；支持 token 级长度（--input_token_len/--output_token_len）。


exp/exp_forecast.py：把 UniCA/MoE 串进训练与评测流程，记录 aux/门控统计；开启 --eval_target_only 时仅对目标通道计分。

data_provider/*：支持 target_channel 切片 与 协变量打包（供 UniCA 使用）。

utils/metrics.py：补充“目标通道”版指标与汇总打印。
