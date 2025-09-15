#!/usr/bin/env bash
# Full re-train + eval for ETTm1 & Weather on 5 lines
# (MoE / UniCA+Patch / RevIN / AdaRevIN / AdaRevIN+DeSTA)
# Usage: bash run_all_fresh_v3.sh [GPU_ID] [CLEAN=1]
set -euo pipefail

GPU="${1:-0}"
CLEAN="${2:-1}"   # 1=删除旧结果与日志, 0=保留

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"; mkdir -p logs results

ETT_ROOT="$HOME/datasets/ETT"
WTH_ROOT="$HOME/datasets/TSLib/Weather"
ETT_FILE="ETTm1.csv"      # 代码不支持 ETThm1；如需小时版改成 ETTh1.csv 且把 run_ettt 的 --data 改成 ETTh1
WTH_FILE="WTH.csv"

# -------- helpers --------
now(){ date "+%F %T"; }
touch_flag(){ : > "$1"; }
clear_flag(){ rm -f "$1" 2>/dev/null || true; }
assert_file(){ [[ -f "$1" ]] || { echo "[FATAL] missing $1"; exit 2; }; }
latest_ckpt(){
  local d="$1"
  [[ -f "$d/checkpoint.pth" ]] && { echo "checkpoint.pth"; return; }
  local f; f=$(ls -t "$d"/*.pth 2>/dev/null | head -n1 || true)
  [[ -n "${f:-}" ]] && basename "$f" || echo "checkpoint.pth"
}

# -------- runners (数组方式，不拼字符串) --------
train_ettt(){
  local MID="$1"; shift
  local FLAGS=( "$@" )
  local LOG="logs/${MID}.train.log"; local RUN="logs/${MID}.RUNNING"; local RES="results/${MID}"
  [[ "$CLEAN" == "1" ]] && { rm -rf "$RES" "$LOG" "logs/${MID}.eval.log"; }
  mkdir -p "$RES"
  echo "[`now`] TRAIN start -> $MID"; echo "  log: $LOG"
  touch_flag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name forecast --is_training 1 \
    --model_id "$MID" --model timer_xl \
    --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
    --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
    --train_epochs 50 --patience 5 --seed 42 \
    --eval_target_only --target_channel 6 --num_workers 0 \
    "${FLAGS[@]}" \
    --checkpoints "$RES" > "$LOG" 2>&1
  clear_flag "$RUN"; echo "[`now`] TRAIN done  -> $MID"
}
eval_ettt(){
  local MID="$1"; shift
  local FLAGS=( "$@" )
  local LOG="logs/${MID}.eval.log"; local RUN="logs/${MID}.EVAL.RUNNING"; local RES="results/${MID}"
  echo "[`now`] EVAL  start -> $MID"; echo "  log: $LOG"
  touch_flag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name forecast --is_training 0 \
    --model_id "$MID" --model timer_xl \
    --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE" \
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --eval_target_only --target_channel 6 --num_workers 0 \
    "${FLAGS[@]}" \
    --checkpoints "$RES" > "$LOG" 2>&1
  clear_flag "$RUN"; echo "[`now`] EVAL  done  -> $MID"
}

train_weather(){
  local MID="$1"; shift
  local FLAGS=( "$@" )
  local LOG="logs/${MID}.train.log"; local RUN="logs/${MID}.RUNNING"; local RES="results/${MID}"
  [[ "$CLEAN" == "1" ]] && { rm -rf "$RES" "$LOG" "logs/${MID}.eval.log"; }
  mkdir -p "$RES"
  local LR="5e-5"; [[ "$MID" == "weather_moe_safe" ]] && LR="2e-4"
  echo "[`now`] TRAIN start -> $MID"; echo "  log: $LOG"
  touch_flag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model_id "$MID" --model timer_xl \
    --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
    --seq_len 672 --input_token_len 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
    --dropout 0.1 --learning_rate "$LR" --batch_size 32 \
    --train_epochs 50 --patience 5 --seed 42 \
    --test_seq_len 672 --test_pred_len 96 \
    --eval_target_only --target_channel 2 --num_workers 0 --n_vars 12 \
    "${FLAGS[@]}" \
    --checkpoints "$RES" > "$LOG" 2>&1
  clear_flag "$RUN"; echo "[`now`] TRAIN done  -> $MID"
}
eval_weather(){
  local MID="$1"; shift
  local FLAGS=( "$@" )
  local LOG="logs/${MID}.eval.log"; local RUN="logs/${MID}.EVAL.RUNNING"; local RES="results/${MID}"
  local TESTDIR; TESTDIR=$(ls -t "$RES" | head -n1)
  local CKPT; CKPT=$(latest_ckpt "$RES/$TESTDIR")
  echo "[`now`] EVAL  start -> $MID"; echo "  log: $LOG"; echo "  use: test_dir=$TESTDIR, ckpt=$CKPT"
  touch_flag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name long_term_forecast --is_training 0 \
    --model_id "$MID" --model timer_xl \
    --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
    --seq_len 672 --input_token_len 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --test_seq_len 672 --test_pred_len 96 \
    --eval_target_only --target_channel 2 --n_vars 12 --seed 42 \
    "${FLAGS[@]}" \
    --checkpoints "$RES" --test_dir "$TESTDIR" --test_file_name "$CKPT" > "$LOG" 2>&1
  clear_flag "$RUN"; echo "[`now`] EVAL  done  -> $MID"
}

# -------- sanity --------
assert_file "${ETT_ROOT}/${ETT_FILE}"
assert_file "${WTH_ROOT}/${WTH_FILE}"

# -------- ETTm1: 5 条线 --------
train_ettt ettm1_timerxl_moe_safe      --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2
eval_ettt  ettm1_timerxl_moe_safe      --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2

train_ettt ettm1_timerxl_unica_post    --use_unica --unica_bottleneck 128 --unica_fusion res_add --unica_stage post --unica_res_scale 0.05 --use_norm
eval_ettt  ettm1_timerxl_unica_post    --use_unica --unica_bottleneck 128 --unica_fusion res_add --unica_stage post --unica_res_scale 0.05 --use_norm

train_ettt ettm1_timerxl_revin         --revin --revin_affine --revin_eps 1e-5
eval_ettt  ettm1_timerxl_revin         --revin --revin_affine --revin_eps 1e-5

train_ettt ettm1_timerxl_adanorm       --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995
eval_ettt  ettm1_timerxl_adanorm       --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995

train_ettt ettm1_timerxl_ada_desta     --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta
eval_ettt  ettm1_timerxl_ada_desta     --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta

# -------- Weather: 5 条线 --------
train_weather weather_moe_safe         --revin --revin_affine --revin_eps 1e-5 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2
eval_weather   weather_moe_safe        --revin --revin_affine --revin_eps 1e-5 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2

train_weather weather_timerxl_unica_patch --use_unica --unica_bottleneck 128 --unica_fusion film_gate --unica_stage post --unica_res_scale 0.05
eval_weather   weather_timerxl_unica_patch --use_unica --unica_bottleneck 128 --unica_fusion film_gate --unica_stage post --unica_res_scale 0.05

train_weather weather_timerxl_revin    --revin --revin_affine --revin_eps 1e-5
eval_weather  weather_timerxl_revin    --revin --revin_affine --revin_eps 1e-5

train_weather weather_timerxl_adanorm  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995
eval_weather  weather_timerxl_adanorm  --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995

train_weather weather_timerxl_ada_desta --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta
eval_weather   weather_timerxl_ada_desta --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta

# -------- summary --------
OUT="results/summary.csv"
echo "model,dataset,mse,mae" > "$OUT"
for e in logs/*eval.log; do
  base=$(basename "$e" .eval.log)
  ds="UNKNOWN"
  [[ "$base" == ettm1_* ]] && ds="ETTm1"
  [[ "$base" == weather_* || "$base" == w_* ]] && ds="Weather"
  mse=$(egrep -io 'mse[^0-9]*[0-9.]+' "$e" | tail -n1 | grep -oE '[0-9.]+' || echo NA)
  mae=$(egrep -io 'mae[^0-9]*[0-9.]+' "$e" | tail -n1 | grep -oE '[0-9.]+' || echo NA)
  echo "$base,$ds,$mse,$mae" >> "$OUT"
done

echo; echo "=== 摘要 ==="
awk -F, 'BEGIN{printf("%-30s %-8s %-12s %-12s\n","model","dataset","mse","mae")} NR>1{printf("%-30s %-8s %-12s %-12s\n",$1,$2,$3,$4)}' "$OUT"

echo; echo "=== 各数据集 MSE 最优/次优 ==="
for ds in ETTm1 Weather; do
  echo ">> $ds"
  awk -F, -v target="$ds" 'NR>1 && $2==target && $3!="NA"{print $0}' "$OUT" \
  | sort -t, -k3,3g \
  | awk -F, '
    NR==1{printf("  1st: **%s**  mse=%s  mae=%s\n",$1,$3,$4)}
    NR==2{printf("  2nd: _%s_  mse=%s  mae=%s\n",$1,$3,$4)}
  '
done

echo; echo "CSV 写入: $OUT"
echo "[`now`] ALL DONE."
