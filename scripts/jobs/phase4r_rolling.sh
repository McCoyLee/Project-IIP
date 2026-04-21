#!/usr/bin/env bash
# ============================================================================
# Phase 4r — 对齐 TimerXL 论文评估协议（One-for-All Rolling Forecasting）
#
# 论文协议 (Table 4):
#   1. 训一个模型（output_token_len=24，短步预测）
#   2. 测试时做 rolling forecast，分别评估 pred_len ∈ {96,192,336,720}
#   3. 评估 ALL channels 的 MSE/MAE（不是 eval_target_only）
#   4. 取 4 个 pred_len 的平均
#
# 数据集: ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic
# 变体: baseline, tan, fir_moe_tan
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/phase4r_rolling.sh
#
#   ONLY_DATASET=ecl bash scripts/jobs/phase4r_rolling.sh
#   ONLY_VARIANT=tan bash scripts/jobs/phase4r_rolling.sh
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
echo " Phase 4r — Rolling Forecast (Paper Protocol)"
echo " DATA_ROOT = $DATA_ROOT"
echo " Filter DS = $ONLY_DS"
echo " Filter VA = $ONLY_VAR"
echo "=========================================="

# ---------- 训练函数：训一个模型 ----------
train_one () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_train__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"
    echo ""
    echo "[$(date)] >>> Training: ${model_id}" | tee -a "$log"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py \
        --is_training 1 \
        --model_id "${model_id}" \
        --checkpoints "${ckpt_dir}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Training Done: ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ---------- 测试函数：对已训好的模型做 rolling forecast ----------
test_rolling () {
    local model_id="$1"; shift
    local test_pred_len="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local setting="$1"; shift  # 训练时的 setting 字符串（= ckpt 子目录名）
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_test_pl${test_pred_len}__${ts}.log"
    echo "[$(date)] >>> Testing: ${model_id} pred_len=${test_pred_len}" | tee -a "$log"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py \
        --is_training 0 \
        --model_id "${model_id}" \
        --checkpoints "${ckpt_dir}" \
        --test_dir "${setting}" \
        --test_file_name checkpoint.pth \
        --test_pred_len "${test_pred_len}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Test Done: ${model_id} pred_len=${test_pred_len}" | tee -a "$log"
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

# ---------- 对某数据集: 训一个模型 + 测 4 个 pred_len ----------
run_variant () {
    local ds_tag="$1"; shift
    local variant="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift
    # 剩余 $@ 是该数据集的公共参数（含 --output_token_len 24）

    local ckpt_dir="${ckpt_base}/${variant}"
    local extra_args=()

    case "$variant" in
        baseline)
            ;;
        tan)
            extra_args=(--use_tan --tan_bands 8)
            ;;
        fir_moe_tan)
            extra_args=("${MOE_ARGS[@]}" --use_fir_moe --fir_bands 8 --fir_spec_alpha 0.01 --use_tan --tan_bands 8)
            ;;
    esac

    # 训练（output_token_len=24, test_pred_len=96 做训练期验证）
    # 训练期的 test 会自动以 test_pred_len=96 评估一次
    train_one "${ds_tag}_r_${variant}" "$log_dir" "$ckpt_dir" \
        "$@" --test_pred_len 96 "${extra_args[@]}"

    # 构造 setting 字符串（必须与 run.py 中生成的一致，用于定位 checkpoint）
    # 格式: forecast_{model_id}_{model}_{data}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_test_0
    # 由于 setting 格式较复杂且易出错，我们直接查找 ckpt_dir 下的子目录
    local setting_dir
    setting_dir=$(ls -1d "${ckpt_dir}"/forecast_* 2>/dev/null | head -1 || true)
    if [[ -z "$setting_dir" || ! -f "${setting_dir}/checkpoint.pth" ]]; then
        echo "[WARN] 找不到训练检查点，跳过 rolling test: ${ckpt_dir}"
        return
    fi
    local setting_name
    setting_name=$(basename "$setting_dir")

    # Rolling 测试：4 个预测长度（192, 336, 720；96 已在训练时测过）
    for pl in 192 336 720; do
        test_rolling "${ds_tag}_r_${variant}" "$pl" "$log_dir" "$ckpt_dir" \
            "$setting_name" \
            "$@" --test_pred_len "$pl" "${extra_args[@]}"
    done
}

# ---------- 对某数据集跑全部变体 ----------
run_dataset () {
    local ds_tag="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift

    for var in baseline tan fir_moe_tan; do
        if [[ "$ONLY_VAR" == "all" || "$ONLY_VAR" == "$var" ]]; then
            run_variant "$ds_tag" "$var" "$log_dir" "$ckpt_base" "$@"
        fi
    done
}

# ---------- 公共模型参数（对齐论文协议）----------
# 关键差异 vs Phase 4:
#   1. output_token_len=24（短步训练 → rolling 测试）
#   2. 不用 --eval_target_only（评估全部通道）
#   3. 保持 d_model=160 先跑；如果仍差距大再试 d_model=256
MODEL_COMMON=(
    --task_name forecast --model timer_xl
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640
    --dropout 0.1 --learning_rate 1e-4
    --train_epochs 50 --patience 5
    --seed 42 --cosine --tmax 50
    --ci_backbone
    --patch_size 0 --stride 0
    --num_workers 4 --ddp
    --output_token_len 24
)

# ============================================================================
#                           各数据集配置
# ============================================================================

# ---- ECL (321 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
echo "========== ECL (Rolling) =========="
run_dataset ecl "logs/phase4r/ecl" "checkpoints/phase4r/ecl" \
    "${MODEL_COMMON[@]}" \
    --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 2 --n_vars 321
fi

# ---- ETTh1 (7 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
echo "========== ETTh1 (Rolling) =========="
run_dataset etth1 "logs/phase4r/etth1" "checkpoints/phase4r/etth1" \
    "${MODEL_COMMON[@]}" \
    --data ETTh1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 7
fi

# ---- ETTh2 (7 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
echo "========== ETTh2 (Rolling) =========="
run_dataset etth2 "logs/phase4r/etth2" "checkpoints/phase4r/etth2" \
    "${MODEL_COMMON[@]}" \
    --data ETTh2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh2.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 7
fi

# ---- ETTm1 (7 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
echo "========== ETTm1 (Rolling) =========="
run_dataset ettm1 "logs/phase4r/ettm1" "checkpoints/phase4r/ettm1" \
    "${MODEL_COMMON[@]}" \
    --data ETTm1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 7
fi

# ---- ETTm2 (7 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
echo "========== ETTm2 (Rolling) =========="
run_dataset ettm2 "logs/phase4r/ettm2" "checkpoints/phase4r/ettm2" \
    "${MODEL_COMMON[@]}" \
    --data ETTm2 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm2.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 7
fi

# ---- Weather (21 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
echo "========== Weather (Rolling) =========="
run_dataset weather "logs/phase4r/weather" "checkpoints/phase4r/weather" \
    "${MODEL_COMMON[@]}" \
    --data Weather \
    --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 8 --n_vars 21
fi

# ---- Traffic (862 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
echo "========== Traffic (Rolling) =========="
run_dataset traffic "logs/phase4r/traffic" "checkpoints/phase4r/traffic" \
    "${MODEL_COMMON[@]}" \
    --data Traffic \
    --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 1 --n_vars 862
fi

# ---- Solar (137 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
echo "========== Solar (Rolling) =========="
run_dataset solar "logs/phase4r/solar" "checkpoints/phase4r/solar" \
    "${MODEL_COMMON[@]}" \
    --data Solar \
    --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 4 --n_vars 137
fi

# ---- Exchange (8 vars) ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
echo "========== Exchange (Rolling) =========="
run_dataset exchange "logs/phase4r/exchange" "checkpoints/phase4r/exchange" \
    "${MODEL_COMMON[@]}" \
    --data Exchange \
    --root_path "${DATA_ROOT}/ExchangeRate" --data_path exchange_rate.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 8
fi

echo ""
echo "[$(date)] Phase 4r Rolling Forecast ALL DONE."
echo "结果 → result_long_term_forecast.txt"
echo ""
echo "与论文对比方法："
echo "  1. 找到 result_long_term_forecast.txt 中带 '_r_baseline' 的行"
echo "  2. 每个数据集有 4 个 pred_len 的结果，取平均"
echo "  3. 与论文 Table 4 对比"
