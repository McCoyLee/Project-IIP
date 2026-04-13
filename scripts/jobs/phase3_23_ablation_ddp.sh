#!/usr/bin/env bash
# ============================================================================
# Phase 3.2 + 3.3 — Combined Ablation (ECL, DDP on 8×V100 16GB)
#
# 本脚本包含两组消融实验，顺序串行：
#
# ---------- 3.2 FIR-MoE 消融 ----------
#   A1) fir_moe_no_spec      : FIR-MoE without spec loss (--fir_spec_alpha 0)
#   A2) fir_moe_ortho        : FIR-MoE with orthogonality reg (替代 spec 的另一种特化)
#   A3) fir_moe_bands4       : 频带数 K=4 （对比主跑的 K=8）
#   A4) fir_moe_bands16      : 频带数 K=16
#   A5) fir_moe_spec_low     : spec_alpha=0.001 （弱正则）
#   A6) fir_moe_spec_high    : spec_alpha=0.1   （强正则）
#
#   注：与"标准 MoE"(w/o freq input) 的对比已在 Phase 3.1 的 ecl_p31_moe 中覆盖。
#   注：FIR-MoE full 参考值直接用 Phase 3.1 的 ecl_p31_fir_moe 结果。
#
# ---------- 3.3 TAN 消融 ----------
#   B1) tan_no_freq_cond     : TAN without freq conditioning (固定 α/β)
#   B2) adanorm_ref          : 现有 AdaRevIN（整序列自适应）作为"无 per-token local stats"参照
#
#   注：TAN full 参考值直接用 Phase 3.1 的 ecl_p31_tan 结果。
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/phase3_23_ablation_ddp.sh
#
# 环境：conda 环境 timerxl ；8×V100 ；必须先跑完 Phase 3.1 才能做完整对比
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-$((30500 + RANDOM % 1000))}

: "${DATA_ROOT:?请先 export DATA_ROOT=/你的/数据根目录}"
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
echo " Phase 3.2+3.3 — Ablations    "
echo " DATA_ROOT   = $DATA_ROOT"
echo " PROJECT     = $PROJECT_ROOT"
echo " GPUs        = $CUDA_VISIBLE_DEVICES"
echo " Master port = $MASTER_PORT"
echo "==============================="

NGPU=8
LOG_DIR="logs/phase3_23"
CKPT_BASE="checkpoints/Electricity/phase3_23"
mkdir -p "$LOG_DIR" "$CKPT_BASE"

# ---------- 通用超参（与 Phase 3.1 完全一致以保证可比性）----------
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

# FIR-MoE 消融所用的 MoE 基础配置（与 3.1 fir_moe 对齐，仅改 FIR 相关项）
MOE_BASE=(
    --use_moe --num_experts 4 --moe_topk 2
    --moe_capacity_factor 1.25
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp
)

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
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ============================================================================
# 3.2 — FIR-MoE 消融
# ============================================================================

# A1) 关闭频率特化正则（保留频率输入）
run_one ecl_p32_fir_moe_no_spec \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.0

# A2) 用正交正则替代 KL-to-uniform 特化正则
run_one ecl_p32_fir_moe_ortho \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.0 --fir_ortho_alpha 0.01

# A3) 频带数 K=4
run_one ecl_p32_fir_moe_bands4 \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 4 --fir_spec_alpha 0.01

# A4) 频带数 K=16
run_one ecl_p32_fir_moe_bands16 \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 16 --fir_spec_alpha 0.01

# A5) 弱特化正则 spec_alpha=0.001
run_one ecl_p32_fir_moe_spec_low \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.001

# A6) 强特化正则 spec_alpha=0.1
run_one ecl_p32_fir_moe_spec_high \
    "${MOE_BASE[@]}" \
    --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.1

# ============================================================================
# 3.3 — TAN 消融
# ============================================================================

# B1) TAN 关闭频率条件（退化为固定 α、仅做局部/全局混合）
run_one ecl_p33_tan_no_freq_cond \
    --use_tan --tan_bands 8 --tan_no_freq_cond

# B2) 用现有 AdaRevIN 做参照（整序列级自适应，无 per-token 局部统计）
#     （与 TAN 的"w/o local stats"在精神上等价）
run_one ecl_p33_adanorm_ref \
    --use_adanorm \
    --adanorm_alpha per_channel --adanorm_beta scalar --adanorm_use_ema

echo ""
echo "[$(date)] Phase 3.2+3.3 ABLATION DONE."
echo "所有日志 → $LOG_DIR"
echo "所有 checkpoints → $CKPT_BASE"
echo ""
echo "完整对比表请参考："
echo "  Phase 3.1: ecl_p31_moe, ecl_p31_fir_moe, ecl_p31_tan, ecl_p31_fir_moe_tan"
echo "  Phase 3.2: A1-A6 (FIR-MoE 消融)"
echo "  Phase 3.3: B1-B2 (TAN 消融)"
