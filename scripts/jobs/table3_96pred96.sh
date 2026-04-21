#!/usr/bin/env bash
# ============================================================================
# Table 3 — Multivariate forecasting (96-pred-96), trained from scratch
#
# 对齐论文 Table 3：seq_len=96, pred_len=96, 直接预测（无 rolling）
# 所有模型从头训练，评估全部通道的 MSE/MAE
#
# 数据集: ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Solar, Exchange
# 变体:   baseline, tan, fir_moe_tan
#
# 参数: d_model=256, d_ff=1024, n_heads=4, e_layers=3
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/table3_96pred96.sh
#
#   ONLY_DATASET=ecl bash scripts/jobs/table3_96pred96.sh
#   ONLY_VARIANT=tan bash scripts/jobs/table3_96pred96.sh
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

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate timerxl

NGPU="${NGPU:-8}"
SEED_ENV="${SEED:-42}"
ONLY_DS="${ONLY_DATASET:-all}"
ONLY_VAR="${ONLY_VARIANT:-all}"

echo "=========================================="
echo " Table 3 — 96-pred-96, from scratch"
echo " d_model=256, d_ff=1024, n_heads=4, e_layers=3"
echo " DATA_ROOT = $DATA_ROOT"
echo " Filter DS = $ONLY_DS  |  Filter VA = $ONLY_VAR"
echo "=========================================="

run_one () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"
    echo ""
    echo "[$(date)] >>> ${model_id}" | tee -a "$log"
    torchrun \
        --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py --model_id "${model_id}" --checkpoints "${ckpt_dir}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

MOE_ARGS=(
    --use_moe --num_experts 4 --moe_topk 2
    --moe_capacity_factor 1.25
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp
    --moe_init_noise 0.02
)

run_variants () {
    local ds_tag="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift

    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "baseline" ]]; then
        run_one "${ds_tag}_t3_baseline" "$log_dir" "${ckpt_base}/baseline" "$@"
    fi
    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "tan" ]]; then
        run_one "${ds_tag}_t3_tan" "$log_dir" "${ckpt_base}/tan" \
            "$@" --use_tan --tan_bands 8
    fi
    if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "fir_moe_tan" ]]; then
        run_one "${ds_tag}_t3_fir_moe_tan" "$log_dir" "${ckpt_base}/fir_moe_tan" \
            "$@" "${MOE_ARGS[@]}" \
            --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 \
            --use_tan --tan_bands 8
    fi
}

# ---- 公共参数：Table 3 (96-pred-96, 从头训练) ----
COMMON=(
    --task_name forecast --is_training 1 --model timer_xl
    --seq_len 96 --input_token_len 96 --input_token_stride 96
    --output_token_len 96 --test_pred_len 96
    --e_layers 3 --d_model 256 --n_heads 4 --d_ff 1024
    --dropout 0.1 --learning_rate 1e-4
    --train_epochs 50 --patience 5
    --seed "${SEED_ENV}" --cosine --tmax 50
    --ci_backbone --patch_size 0 --stride 0
    --num_workers 4 --ddp
)

# ============================================================================
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
echo "========== ECL =========="
run_variants ecl "logs/table3/ecl" "checkpoints/table3/ecl" \
    "${COMMON[@]}" --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
    --batch_size 4 --n_vars 321
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
echo "========== ETTh1 =========="
run_variants etth1 "logs/table3/etth1" "checkpoints/table3/etth1" \
    "${COMMON[@]}" --data ETTh1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv \
    --batch_size 32 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
echo "========== ETTh2 =========="
run_variants etth2 "logs/table3/etth2" "checkpoints/table3/etth2" \
    "${COMMON[@]}" --data ETTh2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh2.csv \
    --batch_size 32 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
echo "========== ETTm1 =========="
run_variants ettm1 "logs/table3/ettm1" "checkpoints/table3/ettm1" \
    "${COMMON[@]}" --data ETTm1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv \
    --batch_size 32 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
echo "========== ETTm2 =========="
run_variants ettm2 "logs/table3/ettm2" "checkpoints/table3/ettm2" \
    "${COMMON[@]}" --data ETTm2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm2.csv \
    --batch_size 32 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
echo "========== Weather =========="
run_variants weather "logs/table3/weather" "checkpoints/table3/weather" \
    "${COMMON[@]}" --data Weather \
    --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
    --batch_size 16 --n_vars 21
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
echo "========== Traffic =========="
run_variants traffic "logs/table3/traffic" "checkpoints/table3/traffic" \
    "${COMMON[@]}" --data Traffic \
    --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
    --batch_size 2 --n_vars 862
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
echo "========== Solar =========="
run_variants solar "logs/table3/solar" "checkpoints/table3/solar" \
    "${COMMON[@]}" --data Solar \
    --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv \
    --batch_size 8 --n_vars 137
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
echo "========== Exchange =========="
run_variants exchange "logs/table3/exchange" "checkpoints/table3/exchange" \
    "${COMMON[@]}" --data Exchange \
    --root_path "${DATA_ROOT}/ExchangeRate" --data_path exchange_rate.csv \
    --batch_size 32 --n_vars 8
fi

echo ""
echo "[$(date)] Table 3 ALL DONE. → result_long_term_forecast.txt"
