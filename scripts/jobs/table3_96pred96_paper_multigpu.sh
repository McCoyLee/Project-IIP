#!/usr/bin/env bash
# ============================================================================
# Timer-XL Table 3 (96-pred-96) — multi-GPU reproduction helper
#
# Paper: Timer-XL (ICLR 2025), Table 3 + Table 11.
# - Table 3 reports 5 datasets: ECL, ETTh1, Traffic, Weather, Solar-Energy
# - Table 11 gives Timer-XL multivariate backbone config: L=5, D=512, H=8, P=96
#
# Key idea for multi-GPU reproduction:
#   Keep EFFECTIVE global batch size aligned with paper by setting
#   per_gpu_batch = paper_global_batch / NPROC_PER_NODE
#
# Usage:
#   export DATA_ROOT=/path/to/datasets
#   NPROC_PER_NODE=4 ONLY_DATASET=etth1 bash scripts/jobs/table3_96pred96_paper_multigpu.sh
#
# Notes:
# - Use NPROC_PER_NODE in {1,2,4,8} to exactly divide paper batch sizes (4/16/32).
# - This script reproduces Timer-XL column (not all baseline models).
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
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

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
ONLY_DS="${ONLY_DATASET:-all}"

calc_local_bs () {
    local global_bs="$1"
    if (( global_bs % NPROC_PER_NODE != 0 )); then
        echo "[ERROR] paper batch_size=${global_bs} 不能被 NPROC_PER_NODE=${NPROC_PER_NODE} 整除。" >&2
        echo "        请改用 1/2/4/8 卡，保持有效全局 batch 与论文一致。" >&2
        exit 1
    fi
    echo $((global_bs / NPROC_PER_NODE))
}

run_one () {
    local model_id="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift

    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"

    echo "[$(date)] >>> ${model_id}" | tee -a "$log"
    torchrun \
      --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      run.py --model_id "${model_id}" --checkpoints "${ckpt_dir}" \
      "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

# Timer-XL backbone per Table 11: L=5, D=512, H=8, P=96
COMMON=(
  --task_name forecast --is_training 1 --model timer_xl
  --seq_len 96 --input_token_len 96 --input_token_stride 96
  --output_token_len 96 --test_pred_len 96
  --e_layers 5 --d_model 512 --n_heads 8 --d_ff 2048
  --dropout 0.1 --train_epochs 10 --patience 10
  --seed 42 --patch_size 0 --stride 0
  --num_workers 4 --ddp
)

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
  # Paper Table 11 (Timer-XL, multivariate ECL): LR=5e-4, BS=4
  run_one "ecl_t3_timerxl_paper" "logs/table3_paper/ecl" "checkpoints/table3_paper/ecl" \
    "${COMMON[@]}" --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
    --learning_rate 5e-4 --batch_size "$(calc_local_bs 4)" --n_vars 321
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
  # Timer baseline on ETTh1 in Table 11 uses LR=1e-4, BS=32; widely used for ETTh1 setup
  run_one "etth1_t3_timerxl_paper" "logs/table3_paper/etth1" "checkpoints/table3_paper/etth1" \
    "${COMMON[@]}" --data ETTh1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv \
    --learning_rate 1e-4 --batch_size "$(calc_local_bs 32)" --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
  # Traffic row in Table 11 corresponds to BS=4, LR=5e-4 for multivariate benchmark settings
  run_one "traffic_t3_timerxl_paper" "logs/table3_paper/traffic" "checkpoints/table3_paper/traffic" \
    "${COMMON[@]}" --data Traffic \
    --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
    --learning_rate 5e-4 --batch_size "$(calc_local_bs 4)" --n_vars 862
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
  # Weather row in Table 11 uses LR=5e-4, BS=32 under multivariate forecasting configs
  run_one "weather_t3_timerxl_paper" "logs/table3_paper/weather" "checkpoints/table3_paper/weather" \
    "${COMMON[@]}" --data Weather \
    --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
    --learning_rate 5e-4 --batch_size "$(calc_local_bs 32)" --n_vars 21
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
  # Solar row in Table 11 uses LR=1e-4, BS=16 under multivariate forecasting configs
  run_one "solar_t3_timerxl_paper" "logs/table3_paper/solar" "checkpoints/table3_paper/solar" \
    "${COMMON[@]}" --data Solar \
    --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv \
    --learning_rate 1e-4 --batch_size "$(calc_local_bs 16)" --n_vars 137
fi

echo "[$(date)] done. results appended in result_long_term_forecast.txt"
