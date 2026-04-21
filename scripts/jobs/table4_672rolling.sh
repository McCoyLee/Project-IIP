#!/usr/bin/env bash
# ============================================================================
# Table 4 — Multivariate forecasting (672-pred-{96,192,336,720})
#           One-for-all: rolling forecasting with one model
#
# 对齐论文 Table 4：
#   - seq_len=672, output_token_len=24（训练时的单步预测长度）
#   - 训一个模型 → rolling 测 pred_len ∈ {96,192,336,720}
#   - 评估全部通道 MSE/MAE
#   - 最终取 4 个 pred_len 的平均
#
# 参数: d_model=256, d_ff=1024, n_heads=4, e_layers=3
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/table4_672rolling.sh
#
#   ONLY_DATASET=ecl bash scripts/jobs/table4_672rolling.sh
#   ONLY_VARIANT=baseline bash scripts/jobs/table4_672rolling.sh
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

NGPU=8
ONLY_DS="${ONLY_DATASET:-all}"
ONLY_VAR="${ONLY_VARIANT:-all}"

echo "=========================================="
echo " Table 4 — 672-pred-{96,192,336,720} Rolling"
echo " d_model=256, d_ff=1024, n_heads=4, e_layers=3"
echo " DATA_ROOT = $DATA_ROOT"
echo " Filter DS = $ONLY_DS  |  Filter VA = $ONLY_VAR"
echo "=========================================="

# ---------- 训练 + 测试（第一个 pred_len）----------
train_and_test () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_train__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"
    echo ""
    echo "[$(date)] >>> Training: ${model_id}" | tee -a "$log"
    torchrun \
        --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py --is_training 1 \
        --model_id "${model_id}" --checkpoints "${ckpt_dir}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Training done: ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ---------- 仅测试（加载 checkpoint，rolling 到指定 pred_len）----------
# 用单 GPU 运行，避免 DDP 在 is_training=0 时的 rank 同步问题
test_only () {
    local model_id="$1"; shift
    local test_pred_len="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local test_dir_name="$1"; shift  # checkpoint 子目录名
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_test_pl${test_pred_len}__${ts}.log"
    echo "[$(date)] >>> Rolling test: ${model_id} pred_len=${test_pred_len}" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --is_training 0 \
        --model_id "${model_id}" \
        --checkpoints "${ckpt_dir}" \
        --test_dir "${test_dir_name}" \
        --test_file_name checkpoint.pth \
        --test_pred_len "${test_pred_len}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Test done: pred_len=${test_pred_len}" | tee -a "$log"
}

MOE_ARGS=(
    --use_moe --num_experts 4 --moe_topk 2
    --moe_capacity_factor 1.25
    --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3
    --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp
    --moe_init_noise 0.02
)

# ---------- 对某变体：训练 + 4 个 rolling 测试 ----------
run_one_variant () {
    local ds_tag="$1"; shift
    local variant="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift
    # 剩余 $@ = 数据集公共参数（含 --output_token_len 24）

    local ckpt_dir="${ckpt_base}/${variant}"
    local extra_args=()

    case "$variant" in
        baseline) ;;
        tan)
            extra_args=(--use_tan --tan_bands 8) ;;
        fir_moe_tan)
            extra_args=("${MOE_ARGS[@]}" --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 --use_tan --tan_bands 8) ;;
    esac

    # Step 1: 训练（output_token_len=24），训练结束自动测 test_pred_len=96
    train_and_test "${ds_tag}_t4_${variant}" "$log_dir" "$ckpt_dir" \
        "$@" --test_pred_len 96 "${extra_args[@]}"

    # Step 2: 定位 checkpoint 目录
    local setting_dir
    setting_dir=$(find "${ckpt_dir}" -maxdepth 1 -name "forecast_*" -type d | head -1)
    if [[ -z "$setting_dir" || ! -f "${setting_dir}/checkpoint.pth" ]]; then
        echo "[WARN] 找不到训练检查点 ${ckpt_dir}，跳过 rolling test"
        return
    fi
    local setting_name
    setting_name=$(basename "$setting_dir")
    echo "Found checkpoint: ${setting_name}"

    # Step 3: Rolling 测试 pred_len=192, 336, 720（96 已在训练后测过）
    # 去掉 --ddp 和 DDP 相关参数用于单 GPU 测试
    local test_args=()
    for arg in "$@" "${extra_args[@]}"; do
        [[ "$arg" == "--ddp" ]] && continue
        test_args+=("$arg")
    done

    for pl in 192 336 720; do
        test_only "${ds_tag}_t4_${variant}" "$pl" "$log_dir" "$ckpt_dir" \
            "$setting_name" "${test_args[@]}" --test_pred_len "$pl"
    done
}

run_dataset () {
    local ds_tag="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift

    for var in baseline tan fir_moe_tan; do
        if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "$var" ]]; then
            run_one_variant "$ds_tag" "$var" "$log_dir" "$ckpt_base" "$@"
        fi
    done
}

# ---- 公共参数：Table 4 (672-rolling, one-for-all) ----
COMMON=(
    --task_name forecast --model timer_xl
    --seq_len 672 --input_token_len 96 --input_token_stride 24
    --output_token_len 24
    --e_layers 3 --d_model 256 --n_heads 4 --d_ff 1024
    --dropout 0.1 --learning_rate 1e-4
    --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --ci_backbone --patch_size 0 --stride 0
    --num_workers 4 --ddp
)

# ============================================================================
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
echo "========== ECL =========="
run_dataset ecl "logs/table4/ecl" "checkpoints/table4/ecl" \
    "${COMMON[@]}" --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
    --batch_size 2 --n_vars 321
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
echo "========== ETTh1 =========="
run_dataset etth1 "logs/table4/etth1" "checkpoints/table4/etth1" \
    "${COMMON[@]}" --data ETTh1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv \
    --batch_size 16 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
echo "========== ETTh2 =========="
run_dataset etth2 "logs/table4/etth2" "checkpoints/table4/etth2" \
    "${COMMON[@]}" --data ETTh2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh2.csv \
    --batch_size 16 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
echo "========== ETTm1 =========="
run_dataset ettm1 "logs/table4/ettm1" "checkpoints/table4/ettm1" \
    "${COMMON[@]}" --data ETTm1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv \
    --batch_size 16 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
echo "========== ETTm2 =========="
run_dataset ettm2 "logs/table4/ettm2" "checkpoints/table4/ettm2" \
    "${COMMON[@]}" --data ETTm2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm2.csv \
    --batch_size 16 --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
echo "========== Weather =========="
run_dataset weather "logs/table4/weather" "checkpoints/table4/weather" \
    "${COMMON[@]}" --data Weather \
    --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
    --batch_size 8 --n_vars 21
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
echo "========== Traffic =========="
run_dataset traffic "logs/table4/traffic" "checkpoints/table4/traffic" \
    "${COMMON[@]}" --data Traffic \
    --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
    --batch_size 1 --n_vars 862
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
echo "========== Solar =========="
run_dataset solar "logs/table4/solar" "checkpoints/table4/solar" \
    "${COMMON[@]}" --data Solar \
    --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv \
    --batch_size 4 --n_vars 137
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
echo "========== Exchange =========="
run_dataset exchange "logs/table4/exchange" "checkpoints/table4/exchange" \
    "${COMMON[@]}" --data Exchange \
    --root_path "${DATA_ROOT}/ExchangeRate" --data_path exchange_rate.csv \
    --batch_size 16 --n_vars 8
fi

echo ""
echo "[$(date)] Table 4 ALL DONE."
echo "结果 → result_long_term_forecast.txt"
echo "每个数据集有 4 个 pred_len 的结果，取平均后对比论文 Table 4"
