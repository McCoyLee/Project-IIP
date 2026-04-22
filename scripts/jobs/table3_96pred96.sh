#!/usr/bin/env bash
# ============================================================================
# Table 3 runner (paper-aligned baseline/experiment settings)
#
# 目标：在原 table3.sh 基础上，把 baseline/tan/fir_moe_tan 的主干参数
# 对齐到与论文 Table 3 更一致的设置，避免旧脚本（sl96, dm256, el3）造成系统性偏差。
#
# 用法：
#   export DATA_ROOT=/path/to/datasets
#   SEED=2021 NGPU=8 ONLY_DATASET=etth1 ONLY_VARIANT=baseline \
#     bash scripts/jobs/table3_96pred96.sh
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}
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

NGPU="${NGPU:-}"
SEED_ENV="${SEED:-2021}"
ONLY_DS="${ONLY_DATASET:-all}"
ONLY_VAR="${ONLY_VARIANT:-all}"

# 可选：自动挑选空闲显卡，避免 GPU0 被占用导致 OOM
# AUTO_GPU=1 且未显式传 CUDA_VISIBLE_DEVICES/GPU_IDS 时生效。
if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_IDS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t _free_ids < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 1024) print $1}')
    if [[ ${#_free_ids[@]} -gt 0 ]]; then
        export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_free_ids[*]}")"
    fi
fi
if [[ -n "${GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _ngpu_vis=$(python - <<'PY2'
import os
v=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
print(len([x for x in v.split(',') if x!='']) if v else 0)
PY2
)
    if [[ "${NGPU:-}" == "" ]]; then
        NGPU="${_ngpu_vis}"
    fi
fi
NGPU="${NGPU:-8}" # fallback

run_one () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"
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

# 通用 IO：Table 3 为 96-pred-96，但 lookback 采用 672（7*96）
COMMON_IO=(
    --task_name forecast --is_training 1 --model timer_xl
    --seq_len 672 --input_token_len 96 --input_token_stride 96
    --output_token_len 96 --test_pred_len 96
    --dropout 0.1 --train_epochs 10 --patience 10
    --seed "${SEED_ENV}" --patch_size 0 --stride 0
    --num_workers 4 --ddp
)

# --------------------------- 论文主对齐的 5 个数据集 ---------------------------
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
run_variants ecl "logs/table3/ecl" "checkpoints/table3/ecl" \
    "${COMMON_IO[@]}" \
    --e_layers 5 --d_model 512 --n_heads 8 --d_ff 2048 \
    --learning_rate 5e-4 --batch_size 4 \
    --data Electricity --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv --n_vars 321
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
run_variants etth1 "logs/table3/etth1" "checkpoints/table3/etth1" \
    "${COMMON_IO[@]}" \
    --e_layers 1 --d_model 1024 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 32 \
    --use_norm --valid_last \
    --data ETTh1 --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
run_variants traffic "logs/table3/traffic" "checkpoints/table3/traffic" \
    "${COMMON_IO[@]}" \
    --e_layers 4 --d_model 512 --n_heads 8 --d_ff 2048 \
    --learning_rate 5e-4 --batch_size 4 \
    --data Traffic --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv --n_vars 862
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
run_variants weather "logs/table3/weather" "checkpoints/table3/weather" \
    "${COMMON_IO[@]}" \
    --e_layers 4 --d_model 512 --n_heads 8 --d_ff 2048 \
    --learning_rate 5e-4 --batch_size 32 \
    --data Weather --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv --n_vars 21
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
run_variants solar "logs/table3/solar" "checkpoints/table3/solar" \
    "${COMMON_IO[@]}" \
    --e_layers 6 --d_model 512 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 16 \
    --data Solar --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv --n_vars 137
fi

# --------------------------- 扩展到原脚本其余 4 个数据集 ---------------------------
# 这些数据集不在你贴的 Table 3 图中，用 ETTh1 同族配置作为默认扩展。
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
run_variants etth2 "logs/table3/etth2" "checkpoints/table3/etth2" \
    "${COMMON_IO[@]}" \
    --e_layers 1 --d_model 1024 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 32 \
    --use_norm --valid_last \
    --data ETTh2 --root_path "${DATA_ROOT}/ETT" --data_path ETTh2.csv --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
run_variants ettm1 "logs/table3/ettm1" "checkpoints/table3/ettm1" \
    "${COMMON_IO[@]}" \
    --e_layers 1 --d_model 1024 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 32 \
    --use_norm --valid_last \
    --data ETTm1 --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
run_variants ettm2 "logs/table3/ettm2" "checkpoints/table3/ettm2" \
    "${COMMON_IO[@]}" \
    --e_layers 1 --d_model 1024 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 32 \
    --use_norm --valid_last \
    --data ETTm2 --root_path "${DATA_ROOT}/ETT" --data_path ETTm2.csv --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
run_variants exchange "logs/table3/exchange" "checkpoints/table3/exchange" \
    "${COMMON_IO[@]}" \
    --e_layers 4 --d_model 512 --n_heads 8 --d_ff 2048 \
    --learning_rate 1e-4 --batch_size 32 \
    --data Exchange --root_path "${DATA_ROOT}/ExchangeRate" --data_path exchange_rate.csv --n_vars 8
fi

echo "[$(date)] done. results appended in result_long_term_forecast.txt"
