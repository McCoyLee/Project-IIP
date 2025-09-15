#!/usr/bin/env bash
# Usage: bash run_all.sh [GPU_ID]
# Default GPU_ID=0
set -euo pipefail

GPU="${1:-0}"

# -------- paths --------
ETT_ROOT="$HOME/datasets/ETT"
WTH_ROOT="$HOME/datasets/TSLib/Weather"
ETT_FILE="ETTm1.csv"
WTH_FILE="WTH.csv"

mkdir -p logs results

# ========= helpers =========
now() { date "+%F %T"; }

run_eval_weather () {
  # $1 = RESULTS_DIR, $2 = MODEL_ID (for logging)
  local RESULTS_DIR="$1"
  local MODEL_ID="$2"
  local TESTDIR
  TESTDIR=$(ls -t "$RESULTS_DIR" | head -n1)
  echo "[$(now)] Eval Weather -> $MODEL_ID | test_dir=$TESTDIR"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name long_term_forecast --is_training 0 \
    --model_id "$MODEL_ID" --model timer_xl \
    --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
    --seq_len 672 --input_token_len 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --test_seq_len 672 --test_pred_len 96 \
    --eval_target_only --target_channel 2 --n_vars 12 --seed 42 \
    --checkpoints "$RESULTS_DIR" \
    --test_dir "$TESTDIR" --test_file_name checkpoint.pth \
    > "logs/${MODEL_ID}.eval.log" 2>&1
}

run_eval_etthm1 () {
  # 简洁评测（你的分支对 ETTm1 通常无需 test_dir）
  # $1 = RESULTS_DIR, $2 = MODEL_ID (for logging)
  local RESULTS_DIR="$1"
  local MODEL_ID="$2"
  echo "[$(now)] Eval ETTm1 -> $MODEL_ID"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name forecast --is_training 0 \
    --model_id "$MODEL_ID" --model timer_xl \
    --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --eval_target_only --target_channel 6 --num_workers 0 \
    --checkpoints "$RESULTS_DIR" \
    > "logs/${MODEL_ID}.eval.log" 2>&1
}

# ========= ETTm1 =========
echo "========== ETTm1 (target_channel=6) =========="

# 1) MoE (safe)
MODEL_ID="ettm1_timerxl_moe_safe"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 6 --num_workers 0 \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 \
  --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 \
  --moe_lb_alpha 1e-2 --moe_imp_alpha 0.0 --moe_zloss_beta 0.0 --moe_entropy_reg 0.0 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_etthm1 "$RESULTS_DIR" "$MODEL_ID"

# 2) UniCA + Patch (post + res_add)
MODEL_ID="ettm1_timerxl_unica_post"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --use_norm --learning_rate 1e-4 \
  --batch_size 32 --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 6 --num_workers 0 \
  --use_unica --unica_bottleneck 128 --unica_fusion res_add \
  --unica_stage post --unica_res_scale 0.05 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_etthm1 "$RESULTS_DIR" "$MODEL_ID"

# 3) RevIN
MODEL_ID="ettm1_timerxl_revin"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 6 --num_workers 0 \
  --revin --revin_affine --revin_eps 1e-5 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_etthm1 "$RESULTS_DIR" "$MODEL_ID"

# 4) AdaRevIN (AdaNorm)
MODEL_ID="ettm1_timerxl_adanorm"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 6 --num_workers 0 \
  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_etthm1 "$RESULTS_DIR" "$MODEL_ID"

# 5) AdaRevIN + DeSTA
MODEL_ID="ettm1_timerxl_ada_desta"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 6 --num_workers 0 \
  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 \
  --use_desta \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_etthm1 "$RESULTS_DIR" "$MODEL_ID"

# ========= Weather =========
echo "========== Weather (target_channel=2, n_vars=12) =========="

# 1) MoE (safe) + RevIN
MODEL_ID="weather_moe_safe"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
  --seq_len 672 --input_token_len 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 \
  --learning_rate 2e-4 --batch_size 32 --train_epochs 30 --patience 5 \
  --test_seq_len 672 --test_pred_len 96 \
  --eval_target_only --target_channel 2 \
  --n_vars 12 --seed 42 \
  --revin --revin_affine --revin_eps 1e-5 \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 \
  --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 \
  --moe_lb_alpha 1e-2 --moe_imp_alpha 0.0 --moe_zloss_beta 0.0 --moe_entropy_reg 0.0 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_weather "$RESULTS_DIR" "$MODEL_ID"

# 2) UniCA + Patch（FiLM 门更适合 Weather）
MODEL_ID="weather_timerxl_unica_patch"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
  --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 5e-5 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 2 --num_workers 0 --n_vars 12 \
  --use_unica --unica_bottleneck 128 --unica_fusion film_gate \
  --unica_stage post --unica_res_scale 0.05 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_weather "$RESULTS_DIR" "$MODEL_ID"

# 3) RevIN
MODEL_ID="weather_timerxl_revin"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
  --seq_len 672 --input_token_len 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 5e-5 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 2 --num_workers 0 --n_vars 12 \
  --revin --revin_affine --revin_eps 1e-5 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_weather "$RESULTS_DIR" "$MODEL_ID"

# 4) AdaRevIN (AdaNorm)
MODEL_ID="weather_timerxl_adanorm"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
  --seq_len 672 --input_token_len 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 5e-5 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 2 --num_workers 0 --n_vars 12 \
  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_weather "$RESULTS_DIR" "$MODEL_ID"

# 5) AdaRevIN + DeSTA
MODEL_ID="weather_timerxl_ada_desta"
RESULTS_DIR="./results/${MODEL_ID}"
echo "[$(now)] Train -> $MODEL_ID"
CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id "$MODEL_ID" --model timer_xl \
  --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
  --seq_len 672 --input_token_len 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
  --dropout 0.1 --learning_rate 5e-5 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --eval_target_only --target_channel 2 --num_workers 0 --n_vars 12 \
  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 \
  --use_desta \
  --checkpoints "$RESULTS_DIR" \
  > "logs/${MODEL_ID}.train.log" 2>&1
run_eval_weather "$RESULTS_DIR" "$MODEL_ID"

echo "[$(now)] ALL DONE."
