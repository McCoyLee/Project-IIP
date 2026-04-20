#!/usr/bin/env bash
# ============================================================================
# Phase 4 — Full Benchmark: 所有数据集 × 多预测长度 × 核心变体
#
# 数据集 (8):
#   ECL (321 vars), ETTh1 (7), ETTh2 (7), ETTm1 (7), ETTm2 (7),
#   Weather (21), Traffic (862), Solar (137), Exchange (8)
#
# 预测长度: {96, 192, 336, 720}
#
# 模型变体:
#   1) baseline     : TimerXL (无增强)
#   2) tan          : TimerXL + TAN v2
#   3) fir_moe_tan  : TimerXL + FIR-MoE + TAN v2
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/phase4_full_benchmark.sh
#
# 筛选数据集/预测长度/变体（可组合）：
#   ONLY_DATASET=ecl bash scripts/jobs/phase4_full_benchmark.sh
#   ONLY_PRED=96 bash scripts/jobs/phase4_full_benchmark.sh
#   ONLY_VARIANT=tan bash scripts/jobs/phase4_full_benchmark.sh
#   ONLY_DATASET=etth1 ONLY_PRED=96 ONLY_VARIANT=baseline bash ...
#
# 环境：conda 环境 timerxl ；单机 8 卡 V100 16GB ；PyTorch ≥ 2.0
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

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

NGPU=8
ONLY_DS="${ONLY_DATASET:-all}"
ONLY_PL="${ONLY_PRED:-all}"
ONLY_VAR="${ONLY_VARIANT:-all}"

echo "=========================================="
echo " Phase 4 — Full Benchmark"
echo " DATA_ROOT = $DATA_ROOT"
echo " PROJECT   = $PROJECT_ROOT"
echo " GPUs      = $CUDA_VISIBLE_DEVICES"
echo " Filter DS = $ONLY_DS"
echo " Filter PL = $ONLY_PL"
echo " Filter VA = $ONLY_VAR"
echo "=========================================="

# ---------- 辅助函数 ----------
run_one () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"
    echo ""
    echo "[$(date)] >>> Launching: ${model_id}" | tee -a "$log"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py \
        --model_id "${model_id}" \
        --checkpoints "${ckpt_dir}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Done: ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ---------- MoE 公共参数 ----------
MOE_ARGS=(
    --use_moe --num_experts 4 --moe_topk 2
    --moe_capacity_factor 1.25
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp
    --moe_init_noise 0.02
)

# ---------- 对某数据集跑全部变体 × 某个预测长度 ----------
run_variants_for_pred_len () {
    local ds_tag="$1"; shift        # e.g. ecl
    local pred_len="$1"; shift
    local log_base="$1"; shift
    local ckpt_base="$1"; shift
    # 剩余 $@ 是该数据集的公共参数（不含 output_token_len）

    local log_dir="${log_base}/pl${pred_len}"
    local ckpt_dir="${ckpt_base}/pl${pred_len}"

    local common_args=("$@" --output_token_len "$pred_len" --test_pred_len "$pred_len")

    # 1) baseline
    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "baseline" ]]; then
        run_one "${ds_tag}_p4_baseline_pl${pred_len}" "$log_dir" "${ckpt_dir}/baseline" \
            "${common_args[@]}"
    fi

    # 2) TAN
    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "tan" ]]; then
        run_one "${ds_tag}_p4_tan_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan" \
            "${common_args[@]}" --use_tan --tan_bands 8
    fi

    # 3) FIR-MoE + TAN
    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "fir_moe_tan" ]]; then
        run_one "${ds_tag}_p4_fir_moe_tan_pl${pred_len}" "$log_dir" "${ckpt_dir}/fir_moe_tan" \
            "${common_args[@]}" "${MOE_ARGS[@]}" \
            --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 \
            --use_tan --tan_bands 8
    fi
}

# ---------- 对某数据集跑全部预测长度 ----------
run_all_pred_lens () {
    local ds_tag="$1"; shift
    local log_base="$1"; shift
    local ckpt_base="$1"; shift
    # 剩余 $@ 是该数据集的公共参数

    for pl in 96 192 336 720; do
        if [[ "$ONLY_PL" == "all" || "$ONLY_PL" == "$pl" ]]; then
            run_variants_for_pred_len "$ds_tag" "$pl" "$log_base" "$ckpt_base" "$@"
        fi
    done
}

# ---------- 公共模型参数 ----------
MODEL_COMMON=(
    --task_name forecast --is_training 1 --model timer_xl
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640
    --dropout 0.1 --learning_rate 1e-4
    --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --ci_backbone
    --patch_size 0 --stride 0
    --eval_target_only
    --num_workers 4 --ddp
)

# ============================================================================
#                           各数据集配置
# ============================================================================

# ---- ECL (321 vars, 周期性强) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
echo ""
echo "========== ECL (321 vars) =========="
run_all_pred_lens ecl "logs/phase4/ecl" "checkpoints/phase4/ecl" \
    "${MODEL_COMMON[@]}" \
    --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" \
    --data_path ECL.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 2 \
    --n_vars 321 --target_channel 0
fi

# ---- ETTh1 (7 vars, 小时级, 趋势为主) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
echo ""
echo "========== ETTh1 (7 vars) =========="
run_all_pred_lens etth1 "logs/phase4/etth1" "checkpoints/phase4/etth1" \
    "${MODEL_COMMON[@]}" \
    --data ETTh1 \
    --root_path "${DATA_ROOT}/ETT" \
    --data_path ETTh1.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 \
    --n_vars 7 --target_channel 6
fi

# ---- ETTh2 (7 vars, 小时级) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
echo ""
echo "========== ETTh2 (7 vars) =========="
run_all_pred_lens etth2 "logs/phase4/etth2" "checkpoints/phase4/etth2" \
    "${MODEL_COMMON[@]}" \
    --data ETTh2 \
    --root_path "${DATA_ROOT}/ETT" \
    --data_path ETTh2.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 \
    --n_vars 7 --target_channel 6
fi

# ---- ETTm1 (7 vars, 15分钟级) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
echo ""
echo "========== ETTm1 (7 vars) =========="
run_all_pred_lens ettm1 "logs/phase4/ettm1" "checkpoints/phase4/ettm1" \
    "${MODEL_COMMON[@]}" \
    --data ETTm1 \
    --root_path "${DATA_ROOT}/ETT" \
    --data_path ETTm1.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 \
    --n_vars 7 --target_channel 6
fi

# ---- ETTm2 (7 vars, 15分钟级) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
echo ""
echo "========== ETTm2 (7 vars) =========="
run_all_pred_lens ettm2 "logs/phase4/ettm2" "checkpoints/phase4/ettm2" \
    "${MODEL_COMMON[@]}" \
    --data ETTm2 \
    --root_path "${DATA_ROOT}/ETT" \
    --data_path ETTm2.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 \
    --n_vars 7 --target_channel 6
fi

# ---- Weather (21 vars, 突变多) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
echo ""
echo "========== Weather (21 vars) =========="
run_all_pred_lens weather "logs/phase4/weather" "checkpoints/phase4/weather" \
    "${MODEL_COMMON[@]}" \
    --data Weather \
    --root_path "${DATA_ROOT}/Weather" \
    --data_path WTH.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 8 \
    --n_vars 21 --target_channel 20
fi

# ---- Traffic (862 vars, 高维交通流) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
echo ""
echo "========== Traffic (862 vars) =========="
run_all_pred_lens traffic "logs/phase4/traffic" "checkpoints/phase4/traffic" \
    "${MODEL_COMMON[@]}" \
    --data Traffic \
    --root_path "${DATA_ROOT}/traffic" \
    --data_path traffic.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 1 \
    --n_vars 862 --target_channel 0
fi

# ---- Solar-Energy (137 vars, 强周期性) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
echo ""
echo "========== Solar (137 vars) =========="
run_all_pred_lens solar "logs/phase4/solar" "checkpoints/phase4/solar" \
    "${MODEL_COMMON[@]}" \
    --data Solar \
    --root_path "${DATA_ROOT}/Solar" \
    --data_path solar_AL.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 4 \
    --n_vars 137 --target_channel 0
fi

# ---- Exchange (8 vars, 日频金融) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
echo ""
echo "========== Exchange (8 vars) =========="
run_all_pred_lens exchange "logs/phase4/exchange" "checkpoints/phase4/exchange" \
    "${MODEL_COMMON[@]}" \
    --data Exchange \
    --root_path "${DATA_ROOT}/ExchangeRate" \
    --data_path exchange_rate.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 \
    --n_vars 8 --target_channel 7
fi

echo ""
echo "[$(date)] Phase 4 Full Benchmark ALL DONE."
echo "日志     → logs/phase4/{dataset}/pl{96,192,336,720}/"
echo "检查点   → checkpoints/phase4/{dataset}/pl{96,192,336,720}/"
echo "汇总结果 → result_long_term_forecast.txt"
