#!/usr/bin/env bash
# ============================================================================
# Phase 3.1 — Multi-Dataset Core Comparison (DDP on 8×V100 16GB)
#
# 在 3 个异质数据集上跑 5 个模型变体，验证跨数据集泛化能力：
#
#   数据集:
#     ECL       (321 vars, 周期性强, 局部平稳段长)
#     ETTh1     (7 vars, 趋势为主, 低频信号)
#     Weather   (21 vars, 突变多, 高频噪声)
#
#   模型:
#     1) baseline           : Timer-XL 不加任何新增
#     2) moe                : Timer-XL + 标准 MoE
#     3) fir_moe            : Timer-XL + FIR-MoE
#     4) tan                : Timer-XL + TAN v2
#     5) fir_moe_tan        : Timer-XL + FIR-MoE + TAN v2
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/phase3_1_multi_dataset_ddp.sh
#
# 可选：只跑某个数据集（跳过其余）
#   ONLY_DATASET=ecl bash scripts/jobs/phase3_1_multi_dataset_ddp.sh
#   ONLY_DATASET=etth1 bash scripts/jobs/phase3_1_multi_dataset_ddp.sh
#   ONLY_DATASET=weather bash scripts/jobs/phase3_1_multi_dataset_ddp.sh
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

echo "==============================="
echo " Phase 3.1 — Multi-Dataset    "
echo " DATA_ROOT = $DATA_ROOT"
echo " PROJECT   = $PROJECT_ROOT"
echo " GPUs      = $CUDA_VISIBLE_DEVICES"
echo "==============================="

NGPU=8

# ---------- 运行单个实验的辅助函数 ----------
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
)

# ---------- 跑全部 5 个变体 ----------
run_all_variants () {
    local ds_tag="$1"; shift     # e.g. ecl, etth1, weather
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift
    # 剩余 $@ 是该数据集的公共参数

    # 1) Baseline
    run_one "${ds_tag}_p31_baseline" "$log_dir" "${ckpt_base}/baseline" "$@"

    # 2) Standard MoE
    run_one "${ds_tag}_p31_moe" "$log_dir" "${ckpt_base}/moe" \
        "$@" "${MOE_ARGS[@]}"

    # 3) FIR-MoE
    run_one "${ds_tag}_p31_fir_moe" "$log_dir" "${ckpt_base}/fir_moe" \
        "$@" "${MOE_ARGS[@]}" \
        --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01

    # 4) TAN v2
    run_one "${ds_tag}_p31_tan" "$log_dir" "${ckpt_base}/tan" \
        "$@" --use_tan --tan_bands 8

    # 5) FIR-MoE + TAN v2
    run_one "${ds_tag}_p31_fir_moe_tan" "$log_dir" "${ckpt_base}/fir_moe_tan" \
        "$@" "${MOE_ARGS[@]}" \
        --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 \
        --use_tan --tan_bands 8
}

ONLY="${ONLY_DATASET:-all}"

# ============================================================================
# ECL — 321 vars, 周期性强
# ============================================================================
if [[ "$ONLY" == "all" || "$ONLY" == "ecl" ]]; then
echo ""
echo "========== ECL (321 vars) =========="

ECL_ARGS=(
    --task_name forecast --is_training 1 --model timer_xl
    --data Electricity
    --root_path "${DATA_ROOT}/Electricity"
    --data_path ECL.csv
    --seq_len 672 --input_token_len 112 --input_token_stride 24
    --output_token_len 24
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640
    --dropout 0.1 --learning_rate 1e-4
    --batch_size 2 --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --n_vars 321 --ci_backbone
    --patch_size 0 --stride 0
    --eval_target_only --target_channel 0
    --num_workers 4 --ddp
)
run_all_variants ecl "logs/phase3_1/ecl" "checkpoints/phase3_1/ecl" "${ECL_ARGS[@]}"
fi

# ============================================================================
# ETTh1 — 7 vars, 趋势为主
# ============================================================================
if [[ "$ONLY" == "all" || "$ONLY" == "etth1" ]]; then
echo ""
echo "========== ETTh1 (7 vars) =========="

ETTH1_ARGS=(
    --task_name forecast --is_training 1 --model timer_xl
    --data ETTh1
    --root_path "${DATA_ROOT}/ETT"
    --data_path ETTh1.csv
    --seq_len 672 --input_token_len 96 --input_token_stride 24
    --output_token_len 24
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640
    --dropout 0.1 --learning_rate 1e-4
    --batch_size 16 --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --n_vars 7 --ci_backbone
    --patch_size 0 --stride 0
    --eval_target_only --target_channel 6
    --num_workers 4 --ddp
)
run_all_variants etth1 "logs/phase3_1/etth1" "checkpoints/phase3_1/etth1" "${ETTH1_ARGS[@]}"
fi

# ============================================================================
# Weather — 21 vars, 突变多
# ============================================================================
if [[ "$ONLY" == "all" || "$ONLY" == "weather" ]]; then
echo ""
echo "========== Weather (21 vars) =========="

WEATHER_ARGS=(
    --task_name forecast --is_training 1 --model timer_xl
    --data Weather
    --root_path "${DATA_ROOT}/Weather"
    --data_path WTH.csv
    --seq_len 672 --input_token_len 96 --input_token_stride 24
    --output_token_len 24
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640
    --dropout 0.1 --learning_rate 1e-4
    --batch_size 8 --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --n_vars 21 --ci_backbone
    --patch_size 0 --stride 0
    --eval_target_only --target_channel 20
    --num_workers 4 --ddp
)
run_all_variants weather "logs/phase3_1/weather" "checkpoints/phase3_1/weather" "${WEATHER_ARGS[@]}"
fi

echo ""
echo "[$(date)] Phase 3.1 Multi-Dataset ALL DONE."
echo "日志 → logs/phase3_1/{ecl,etth1,weather}/"
echo "检查点 → checkpoints/phase3_1/{ecl,etth1,weather}/"
echo "汇总结果 → result_long_term_forecast.txt"
