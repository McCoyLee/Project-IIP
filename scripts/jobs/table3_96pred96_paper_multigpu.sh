#!/usr/bin/env bash
# ============================================================================
# Timer-XL Table 3 (96-pred-96) — multi-GPU reproduction helper
#
# NOTE:
# Table 11 gives dataset-level hyper-parameter rows under multivariate forecasting.
# In practice, ETTh1/Traffic/Weather/Solar use different backbone depths/widths than
# ECL. This script follows those dataset-level rows to improve reproducibility.
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
    torchrun       --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}"       --master_port="${MASTER_PORT}"       run.py --model_id "${model_id}" --checkpoints "${ckpt_dir}"       "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

COMMON_IO=(
  --task_name forecast --is_training 1 --model timer_xl
  --seq_len 672 --input_token_len 96 --input_token_stride 96
  --output_token_len 96 --test_pred_len 96
  --dropout 0.1 --train_epochs 10 --patience 10
  --seed 42 --patch_size 0 --stride 0 --use_norm --valid_last
  --num_workers 4 --ddp
)

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
  run_one "ecl_t3_timerxl_paper" "logs/table3_paper/ecl" "checkpoints/table3_paper/ecl"     "${COMMON_IO[@]}"     --e_layers 5 --d_model 512 --n_heads 8 --d_ff 2048     --data Electricity --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv     --learning_rate 5e-4 --batch_size "$(calc_local_bs 4)" --n_vars 321
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
  run_one "etth1_t3_timerxl_paper" "logs/table3_paper/etth1" "checkpoints/table3_paper/etth1"     "${COMMON_IO[@]}"     --e_layers 1 --d_model 1024 --n_heads 8 --d_ff 2048     --data ETTh1 --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv     --learning_rate 1e-4 --batch_size "$(calc_local_bs 32)" --n_vars 7
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
  run_one "traffic_t3_timerxl_paper" "logs/table3_paper/traffic" "checkpoints/table3_paper/traffic"     "${COMMON_IO[@]}"     --e_layers 4 --d_model 512 --n_heads 8 --d_ff 2048     --data Traffic --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv     --learning_rate 5e-4 --batch_size "$(calc_local_bs 4)" --n_vars 862
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
  run_one "weather_t3_timerxl_paper" "logs/table3_paper/weather" "checkpoints/table3_paper/weather"     "${COMMON_IO[@]}"     --e_layers 4 --d_model 512 --n_heads 8 --d_ff 2048     --data Weather --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv     --learning_rate 5e-4 --batch_size "$(calc_local_bs 32)" --n_vars 21
fi

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
  run_one "solar_t3_timerxl_paper" "logs/table3_paper/solar" "checkpoints/table3_paper/solar"     "${COMMON_IO[@]}"     --e_layers 6 --d_model 512 --n_heads 8 --d_ff 2048     --data Solar --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv     --learning_rate 1e-4 --batch_size "$(calc_local_bs 16)" --n_vars 137
fi

echo "[$(date)] done. results appended in result_long_term_forecast.txt"
