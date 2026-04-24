#!/usr/bin/env bash
# Table 4 runner for Stable Koopman Spectral Residual (SKSR).
# Fair setting: same Timer-XL backbone, same seq_len=672, same output_token_len=24,
# same rolling evaluation at pred_len={96,192,336,720}. The only change for the
# SKSR variant is --model timer_xl_sksr plus a bounded residual gate.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

: "${DATA_ROOT:?please export DATA_ROOT=/path/to/datasets}"
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

NGPU=${NGPU:-8}
SEED=${SEED:-42}
ONLY_DS="${ONLY_DATASET:-all}"
# Default runs only the new method. Use VARIANTS=baseline,sksr for a fresh fair pair.
VARIANTS="${VARIANTS:-sksr}"

IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"

echo "=========================================="
echo " Table 4 SKSR rolling benchmark"
echo " seed=$SEED ngpu=$NGPU variants=$VARIANTS"
echo " DATA_ROOT=$DATA_ROOT"
echo "=========================================="

train_and_test () {
    local model_id="$1"; shift
    local model_name="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_train__${ts}.log"
    mkdir -p "$ckpt_dir" "$log_dir"

    echo "[$(date)] >>> Training: ${model_id}" | tee -a "$log"
    torchrun \
        --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT}" \
        run.py --is_training 1 \
        --model "$model_name" \
        --model_id "${model_id}" --checkpoints "${ckpt_dir}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Training done: ${model_id}" | tee -a "$log"
    MASTER_PORT=$((MASTER_PORT + 1))
}

test_only () {
    local model_id="$1"; shift
    local model_name="$1"; shift
    local test_pred_len="$1"; shift
    local log_dir="$1"; shift
    local ckpt_dir="$1"; shift
    local test_dir_name="$1"; shift
    local ts; ts=$(date +%F_%H-%M-%S)
    local log="${log_dir}/${model_id}_test_pl${test_pred_len}__${ts}.log"

    echo "[$(date)] >>> Rolling test: ${model_id} pred_len=${test_pred_len}" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --is_training 0 \
        --model "$model_name" \
        --model_id "${model_id}" \
        --checkpoints "${ckpt_dir}" \
        --test_dir "${test_dir_name}" \
        --test_file_name checkpoint.pth \
        --test_pred_len "${test_pred_len}" \
        "$@" 2>&1 | tee -a "$log"
    echo "[$(date)] <<< Test done: pred_len=${test_pred_len}" | tee -a "$log"
}

variant_args () {
    local variant="$1"
    case "$variant" in
        baseline)
            echo "timer_xl"
            ;;
        sksr)
            echo "timer_xl_sksr --linear_res_scale 0.35 --linear_init_gate -3.0"
            ;;
        sksr_small)
            echo "timer_xl_sksr --linear_res_scale 0.20 --linear_init_gate -3.5"
            ;;
        sksr_tan)
            echo "timer_xl_sksr --linear_res_scale 0.30 --linear_init_gate -3.0 --use_tan --tan_bands 8"
            ;;
        *)
            echo "unknown variant: $variant" >&2
            return 1
            ;;
    esac
}

run_one_variant () {
    local ds_tag="$1"; shift
    local variant="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift

    local spec model_name extra_string
    spec=$(variant_args "$variant")
    read -r model_name extra_string <<< "$spec"

    local extra_args=()
    if [[ -n "${extra_string:-}" ]]; then
        read -r -a extra_args <<< "$extra_string"
    fi

    local ckpt_dir="${ckpt_base}/${variant}"
    train_and_test "${ds_tag}_t4_${variant}" "$model_name" "$log_dir" "$ckpt_dir" \
        "$@" --test_pred_len 96 "${extra_args[@]}"

    local setting_dir
    setting_dir=$(find "${ckpt_dir}" -maxdepth 1 -name "forecast_*" -type d | sort | tail -1)
    if [[ -z "$setting_dir" || ! -f "${setting_dir}/checkpoint.pth" ]]; then
        echo "[WARN] checkpoint not found under ${ckpt_dir}; skip rolling tests"
        return
    fi
    local setting_name
    setting_name=$(basename "$setting_dir")
    echo "Found checkpoint: ${setting_name}"

    local test_args=()
    for arg in "$@" "${extra_args[@]}"; do
        [[ "$arg" == "--ddp" ]] && continue
        test_args+=("$arg")
    done

    for pl in 192 336 720; do
        test_only "${ds_tag}_t4_${variant}" "$model_name" "$pl" "$log_dir" "$ckpt_dir" \
            "$setting_name" "${test_args[@]}" --test_pred_len "$pl"
    done
}

run_dataset () {
    local ds_tag="$1"; shift
    local log_dir="$1"; shift
    local ckpt_base="$1"; shift

    for var in "${VARIANT_LIST[@]}"; do
        run_one_variant "$ds_tag" "$var" "$log_dir" "$ckpt_base" "$@"
    done
}

COMMON=(
    --task_name forecast
    --seq_len 672 --input_token_len 96 --input_token_stride 24
    --output_token_len 24
    --e_layers 3 --d_model 256 --n_heads 4 --d_ff 1024
    --dropout 0.1 --learning_rate 1e-4
    --train_epochs 50 --patience 5
    --seed "$SEED" --cosine --tmax 50
    --ci_backbone --patch_size 0 --stride 0
    --num_workers 4 --ddp
)

if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
    run_dataset ecl "logs/table4_sksr/ecl" "checkpoints/table4_sksr/ecl" \
        "${COMMON[@]}" --data Electricity --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
        --batch_size 2 --n_vars 321
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth1" ]]; then
    run_dataset etth1 "logs/table4_sksr/etth1" "checkpoints/table4_sksr/etth1" \
        "${COMMON[@]}" --data ETTh1 --root_path "${DATA_ROOT}/ETT" --data_path ETTh1.csv \
        --batch_size 16 --n_vars 7
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "etth2" ]]; then
    run_dataset etth2 "logs/table4_sksr/etth2" "checkpoints/table4_sksr/etth2" \
        "${COMMON[@]}" --data ETTh2 --root_path "${DATA_ROOT}/ETT" --data_path ETTh2.csv \
        --batch_size 16 --n_vars 7
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
    run_dataset ettm1 "logs/table4_sksr/ettm1" "checkpoints/table4_sksr/ettm1" \
        "${COMMON[@]}" --data ETTm1 --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv \
        --batch_size 16 --n_vars 7
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm2" ]]; then
    run_dataset ettm2 "logs/table4_sksr/ettm2" "checkpoints/table4_sksr/ettm2" \
        "${COMMON[@]}" --data ETTm2 --root_path "${DATA_ROOT}/ETT" --data_path ETTm2.csv \
        --batch_size 16 --n_vars 7
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
    run_dataset weather "logs/table4_sksr/weather" "checkpoints/table4_sksr/weather" \
        "${COMMON[@]}" --data Weather --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
        --batch_size 8 --n_vars 21
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
    run_dataset traffic "logs/table4_sksr/traffic" "checkpoints/table4_sksr/traffic" \
        "${COMMON[@]}" --data Traffic --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
        --batch_size 1 --n_vars 862
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "solar" ]]; then
    run_dataset solar "logs/table4_sksr/solar" "checkpoints/table4_sksr/solar" \
        "${COMMON[@]}" --data Solar --root_path "${DATA_ROOT}/Solar" --data_path solar_AL.csv \
        --batch_size 4 --n_vars 137
fi
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "exchange" ]]; then
    run_dataset exchange "logs/table4_sksr/exchange" "checkpoints/table4_sksr/exchange" \
        "${COMMON[@]}" --data Exchange --root_path "${DATA_ROOT}/ExchangeRate" --data_path exchange_rate.csv \
        --batch_size 16 --n_vars 8
fi

echo "[$(date)] Table 4 SKSR jobs finished. Results are appended to result_long_term_forecast.txt"
