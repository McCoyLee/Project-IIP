#!/usr/bin/env bash
# ============================================================================
# Phase 3.1 — Core Comparison (ECL, DDP on 8×V100 16GB)
#
# 对比如下 5 个模型，顺序串行训练（每个实验各跑 50 epoch，占满 8 张卡）：
#   1) baseline           : Timer-XL + 无任何新增
#   2) moe                : Timer-XL + 标准 MoE
#   3) fir_moe            : Timer-XL + FIR-MoE（频率引导路由 + 特化正则）
#   4) tan                : Timer-XL + TAN（Token 自适应归一化）
#   5) fir_moe + tan      : Timer-XL + FIR-MoE + TAN（完整方法）
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets    # 含 Electricity/ECL.csv
#   bash scripts/jobs/phase3_1_core_ddp.sh
#
# 环境：conda 环境 timerxl ；单机 8 卡 V100 ；PyTorch ≥ 2.0 （用 torchrun）
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 禁用 tokenizers/并行库的线程警告
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# 让 torchrun 挑选空闲端口（多实验顺序跑时避免端口冲突）
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

# ---------- 用户需要设置的路径 ----------
: "${DATA_ROOT:?请先 export DATA_ROOT=/你的/数据根目录，里面需有 Electricity/ECL.csv}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/../.." && pwd)}"

cd "$PROJECT_ROOT"

# ---------- 激活 conda 环境 ----------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate timerxl

echo "==============================="
echo " Phase 3.1 — Core Comparison  "
echo " DATA_ROOT   = $DATA_ROOT"
echo " PROJECT     = $PROJECT_ROOT"
echo " GPUs        = $CUDA_VISIBLE_DEVICES"
echo " Master port = $MASTER_PORT"
echo " Python      = $(python -c 'import sys; print(sys.executable)')"
echo " Torch       = $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"
echo "==============================="

NGPU=8
LOG_DIR="logs/phase3_1"
CKPT_BASE="checkpoints/Electricity/phase3_1"
mkdir -p "$LOG_DIR" "$CKPT_BASE"

# ---------- 通用超参（与基线脚本对齐）----------
# 注意：batch_size=2 是 per-GPU；DDP 有效 batch = 2 × 8 = 16
#       learning_rate 与原单卡基线相同（1e-4），若训练曲线震荡可考虑 warmup 或调到 3e-4
COMMON_ARGS=(
    --task_name forecast
    --is_training 1
    --model timer_xl
    --data Electricity
    --root_path "${DATA_ROOT}/Electricity"
    --data_path ECL.csv
    --seq_len 672
    --input_token_len 112
    --input_token_stride 24
    --output_token_len 24
    --e_layers 3
    --d_model 160
    --n_heads 2
    --d_ff 640
    --dropout 0.1
    --learning_rate 1e-4
    --batch_size 2
    --train_epochs 50
    --patience 5
    --seed 42
    --cosine
    --tmax 50
    --n_vars 321
    --patch_size 0
    --stride 0
    --eval_target_only
    --target_channel 0
    --num_workers 4
    --ddp
)

# 运行一个实验的辅助函数
run_one () {
    local model_id="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${LOG_DIR}/${model_id}__${ts}.log"
    local ckpt="${CKPT_BASE}/${model_id}"
    mkdir -p "$ckpt"
    echo ""
    echo "[$(date)] >>> Launching: ${model_id}" | tee -a "$log"
    echo "    ckpt: $ckpt"  | tee -a "$log"
    echo "    log : $log"   | tee -a "$log"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py \
        "${COMMON_ARGS[@]}" \
        --model_id "${model_id}" \
        --checkpoints "${ckpt}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Done: ${model_id}" | tee -a "$log"
    # 顺序跑每个实验后递增 master port，避免残留 socket 冲突
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ============================================================================
# 1) Baseline — 不加任何新增模块
# ============================================================================
run_one ecl_p31_baseline

# ============================================================================
# 2) Standard MoE — 仅启用标准 MoE（无频率感知）
# ============================================================================
run_one ecl_p31_moe \
    --use_moe --num_experts 4 --moe_topk 2 \
    --moe_capacity_factor 1.25 \
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 \
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp

# ============================================================================
# 3) FIR-MoE — 频率引导专家路由 + 频率特化正则（核心贡献）
# ============================================================================
run_one ecl_p31_fir_moe \
    --use_moe --num_experts 4 --moe_topk 2 \
    --moe_capacity_factor 1.25 \
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 \
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01

# ============================================================================
# 4) TAN — Token 自适应归一化（补充贡献）
# ============================================================================
run_one ecl_p31_tan \
    --use_tan --tan_bands 8

# ============================================================================
# 5) FIR-MoE + TAN — 完整方法
# ============================================================================
run_one ecl_p31_fir_moe_tan \
    --use_moe --num_experts 4 --moe_topk 2 \
    --moe_capacity_factor 1.25 \
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 \
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 \
    --use_tan --tan_bands 8

echo ""
echo "[$(date)] Phase 3.1 ALL DONE."
echo "所有日志 → $LOG_DIR"
echo "所有 checkpoints → $CKPT_BASE"
echo "汇总结果请查看 result_long_term_forecast.txt"
