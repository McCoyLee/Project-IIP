#!/usr/bin/env bash
# ============================================================================
# Phase 5 — TAN 消融实验 + 频谱一致性 Loss
#
# 在 4 个代表性数据集上跑 TAN 消融变体，验证各组件贡献：
#   - ECL (321 vars, 高维周期)  — TAN 在短期赢但长期输，需要理解
#   - ETTm1 (7 vars)           — TAN 稳定领先
#   - Traffic (862 vars)       — TAN 最强
#   - Weather (21 vars)        — 三者接近，区分度低
#
# 消融变体 (6):
#   1) baseline         : TimerXL 无增强
#   2) tan              : 完整 TAN（频率条件 + per-token stats）
#   3) tan_no_freq      : TAN 去掉频率条件（固定门控 g）
#   4) tan_K4           : TAN 频带数 K=4（vs 默认 K=8）
#   5) tan_K16          : TAN 频带数 K=16
#   6) tan_spec_loss    : TAN + 频谱一致性 loss
#
# 预测长度: {96, 336}（代表短期和长期）
#
# 使用方法：
#   export DATA_ROOT=/home/你的用户名/datasets
#   bash scripts/jobs/phase5_ablation.sh
#
#   ONLY_DATASET=ecl bash scripts/jobs/phase5_ablation.sh
#   ONLY_PRED=96 bash scripts/jobs/phase5_ablation.sh
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
ONLY_PL="${ONLY_PRED:-all}"

echo "=========================================="
echo " Phase 5 — TAN Ablation + Spectral Loss"
echo " DATA_ROOT = $DATA_ROOT"
echo " Filter DS = $ONLY_DS"
echo " Filter PL = $ONLY_PL"
echo "=========================================="

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

# ---------- 对某数据集跑消融变体 × 某个预测长度 ----------
run_ablation () {
    local ds_tag="$1"; shift
    local pred_len="$1"; shift
    local log_base="$1"; shift
    local ckpt_base="$1"; shift

    local log_dir="${log_base}/pl${pred_len}"
    local ckpt_dir="${ckpt_base}/pl${pred_len}"
    local common=("$@" --output_token_len "$pred_len" --test_pred_len "$pred_len")

    # 1) baseline（可复用 Phase 4 结果，此处为方便也跑一次）
    run_one "${ds_tag}_p5_baseline_pl${pred_len}" "$log_dir" "${ckpt_dir}/baseline" \
        "${common[@]}"

    # 2) TAN 完整版（K=8, freq_cond=True）
    run_one "${ds_tag}_p5_tan_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan" \
        "${common[@]}" --use_tan --tan_bands 8

    # 3) TAN 去掉频率条件（仅 per-token stats + 固定 gate）
    run_one "${ds_tag}_p5_tan_no_freq_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan_no_freq" \
        "${common[@]}" --use_tan --tan_bands 8 --tan_no_freq_cond

    # 4) TAN K=4
    run_one "${ds_tag}_p5_tan_K4_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan_K4" \
        "${common[@]}" --use_tan --tan_bands 4

    # 5) TAN K=16
    run_one "${ds_tag}_p5_tan_K16_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan_K16" \
        "${common[@]}" --use_tan --tan_bands 16

    # 6) TAN + 频谱一致性 loss
    run_one "${ds_tag}_p5_tan_spec_pl${pred_len}" "$log_dir" "${ckpt_dir}/tan_spec" \
        "${common[@]}" --use_tan --tan_bands 8 \
        --use_spectral_loss --spectral_lambda 0.1
}

run_ablation_all_preds () {
    local ds_tag="$1"; shift
    local log_base="$1"; shift
    local ckpt_base="$1"; shift
    for pl in 96 336; do
        if [[ "$ONLY_PL" == "all" || "$ONLY_PL" == "$pl" ]]; then
            run_ablation "$ds_tag" "$pl" "$log_base" "$ckpt_base" "$@"
        fi
    done
}

# ---- ECL ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ecl" ]]; then
echo "========== ECL Ablation =========="
run_ablation_all_preds ecl "logs/phase5/ecl" "checkpoints/phase5/ecl" \
    "${MODEL_COMMON[@]}" \
    --data Electricity \
    --root_path "${DATA_ROOT}/Electricity" --data_path ECL.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 2 --n_vars 321 --target_channel 0
fi

# ---- ETTm1 ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "ettm1" ]]; then
echo "========== ETTm1 Ablation =========="
run_ablation_all_preds ettm1 "logs/phase5/ettm1" "checkpoints/phase5/ettm1" \
    "${MODEL_COMMON[@]}" \
    --data ETTm1 \
    --root_path "${DATA_ROOT}/ETT" --data_path ETTm1.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 16 --n_vars 7 --target_channel 6
fi

# ---- Traffic ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "traffic" ]]; then
echo "========== Traffic Ablation =========="
run_ablation_all_preds traffic "logs/phase5/traffic" "checkpoints/phase5/traffic" \
    "${MODEL_COMMON[@]}" \
    --data Traffic \
    --root_path "${DATA_ROOT}/traffic" --data_path traffic.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 1 --n_vars 862 --target_channel 0
fi

# ---- Weather ----
if [[ "$ONLY_DS" == "all" || "$ONLY_DS" == "weather" ]]; then
echo "========== Weather Ablation =========="
run_ablation_all_preds weather "logs/phase5/weather" "checkpoints/phase5/weather" \
    "${MODEL_COMMON[@]}" \
    --data Weather \
    --root_path "${DATA_ROOT}/Weather" --data_path WTH.csv \
    --seq_len 672 --input_token_len 96 --input_token_stride 24 \
    --batch_size 8 --n_vars 21 --target_channel 20
fi

echo ""
echo "[$(date)] Phase 5 Ablation ALL DONE."
echo "日志 → logs/phase5/{ecl,ettm1,traffic,weather}/pl{96,336}/"
