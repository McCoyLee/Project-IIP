#!/usr/bin/env bash
# Full re-train + eval for ETTm1 & Weather on 5 lines:
# (MoE / UniCA+Patch / RevIN / AdaRevIN / AdaRevIN+DeSTA)
# Usage: bash run_all_fresh.sh [GPU_ID] [CLEAN=1]
set -euo pipefail

GPU="${1:-0}"
CLEAN="${2:-1}"   # 1=删除旧 results/<id> 与 logs/<id>*.log；0=保留

# ---------- paths ----------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs results

ETT_ROOT="${HOME}/datasets/ETT"
WTH_ROOT="${HOME}/datasets/TSLib/Weather"
ETT_FILE_M="ETTm1.csv"   # 代码不认识 ETThm1；如需小时版把它改成 ETTh1.csv + DS=ETTh1
WTH_FILE="WTH.csv"

# ---------- helpers ----------
now(){ date "+%F %T"; }
touch_runflag(){ : > "$1"; } # create/overwrite RUNNING flag
end_runflag(){ rm -f "$1" 2>/dev/null || true; }

assert_file(){
  if [[ ! -f "$1" ]]; then
    echo "[FATAL] Missing file: $1"
    exit 2
  fi
}

latest_ckpt_in_dir(){
  local d="$1"
  [[ -f "$d/checkpoint.pth" ]] && { echo "checkpoint.pth"; return; }
  local x
  x=$(ls -t "$d"/*.pth 2>/dev/null | head -n1 || true)
  [[ -n "${x:-}" ]] && basename "$x" || echo "checkpoint.pth"
}

# 打印本次要跑的清单
PLAN=(
  "ETTm1 ettm1_timerxl_moe_safe        train '--use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2'    eval '--use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2'"
  "ETTm1 ettm1_timerxl_unica_post      train '--use_unica --unica_bottleneck 128 --unica_fusion res_add --unica_stage post --unica_res_scale 0.05 --use_norm'                      eval  '--use_unica --unica_bottleneck 128 --unica_fusion res_add --unica_stage post --unica_res_scale 0.05 --use_norm'"
  "ETTm1 ettm1_timerxl_revin           train '--revin --revin_affine --revin_eps 1e-5'                                                                                              eval  '--revin --revin_affine --revin_eps 1e-5'"
  "ETTm1 ettm1_timerxl_adanorm         train '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995'                                                    eval  '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995'"
  "ETTm1 ettm1_timerxl_ada_desta       train '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta'                                        eval  '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta'"

  "Weather weather_moe_safe            train '--revin --revin_affine --revin_eps 1e-5 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2' eval '--revin --revin_affine --revin_eps 1e-5 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2'"
  "Weather weather_timerxl_unica_patch train '--use_unica --unica_bottleneck 128 --unica_fusion film_gate --unica_stage post --unica_res_scale 0.05'                               eval '--use_unica --unica_bottleneck 128 --unica_fusion film_gate --unica_stage post --unica_res_scale 0.05'"
  "Weather weather_timerxl_revin       train '--revin --revin_affine --revin_eps 1e-5'                                                                                              eval '--revin --revin_affine --revin_eps 1e-5'"
  "Weather weather_timerxl_adanorm     train '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995'                                                    eval '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995'"
  "Weather weather_timerxl_ada_desta   train '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta'                                        eval '--use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 --use_desta'"
)

echo "========== PLAN (${#PLAN[@]} jobs) =========="
printf "%-8s  %-30s  %s\n" "DS" "MODEL_ID" "FLAGS"
for item in "${PLAN[@]}"; do
  set -- $item
  echo "$1  $2  (${5})"
done
echo "============================================="

# ---------- sanity checks ----------
assert_file "${ETT_ROOT}/${ETT_FILE_M}"
assert_file "${WTH_ROOT}/${WTH_FILE}"

# ---------- runners ----------
train_ettt(){
  local MID="$1"; local TRAIN_FLAGS="$2"
  local LOG="logs/${MID}.train.log"; local RUN="logs/${MID}.RUNNING"
  local RES="results/${MID}"
  [[ "$CLEAN" == "1" ]] && rm -rf "$RES" "$LOG" "logs/${MID}.eval.log" 2>/dev/null || true
  mkdir -p "$RES"

  echo "[`now`] TRAIN start -> $MID"
  echo "   log: $LOG"
  touch_runflag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name forecast --is_training 1 \
    --model_id "$MID" --model timer_xl \
    --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE_M" \
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 \
    --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
    --train_epochs 50 --patience 5 --seed 42 \
    --eval_target_only --target_channel 6 --num_workers 0 \
    $TRAIN_FLAGS \
    --checkpoints "$RES" \
    > "$LOG" 2>&1
  end_runflag "$RUN"
  echo "[`now`] TRAIN done  -> $MID"
}

eval_ettt(){
  local MID="$1"; local EVAL_FLAGS="$2"
  local LOG="logs/${MID}.eval.log"; local RUN="logs/${MID}.EVAL.RUNNING"
  local RES="results/${MID}"
  echo "[`now`] EVAL  start -> $MID"
  echo "   log: $LOG"
  touch_runflag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name forecast --is_training 0 \
    --model_id "$MID" --model timer_xl \
    --data ETTm1 --root_path "$ETT_ROOT" --data_path "$ETT_FILE_M" \
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --eval_target_only --target_channel 6 --num_workers 0 \
    $EVAL_FLAGS \
    --checkpoints "$RES" \
    > "$LOG" 2>&1
  end_runflag "$RUN"
  echo "[`now`] EVAL  done  -> $MID"
}

train_weather(){
  local MID="$1"; local TRAIN_FLAGS="$2"
  local LOG="logs/${MID}.train.log"; local RUN="logs/${MID}.RUNNING"
  local RES="results/${MID}"
  [[ "$CLEAN" == "1" ]] && rm -rf "$RES" "$LOG" "logs/${MID}.eval.log" 2>/dev/null || true
  mkdir -p "$RES"

  # 不同线稍有学习率差异：MoE 用 2e-4；其它 5e-5。由外部传 train_flags 控制也可。
  local LR="5e-5"; [[ "$MID" == "weather_moe_safe" ]] && LR="2e-4"

  echo "[`now`] TRAIN start -> $MID"
  echo "   log: $LOG"
  touch_runflag "$RUN"
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
    $TRAIN_FLAGS \
    --checkpoints "$RES" \
    > "$LOG" 2>&1
  end_runflag "$RUN"
  echo "[`now`] TRAIN done  -> $MID"
}

eval_weather(){
  local MID="$1"; local EVAL_FLAGS="$2"
  local LOG="logs/${MID}.eval.log"; local RUN="logs/${MID}.EVAL.RUNNING"
  local RES="results/${MID}"
  local TESTDIR; TESTDIR=$(ls -t "$RES" | head -n1)
  local CKPT;    CKPT=$(latest_ckpt_in_dir "$RES/$TESTDIR")
  echo "[`now`] EVAL  start -> $MID"
  echo "   log: $LOG"
  echo "   use: test_dir=$TESTDIR, ckpt=$CKPT"
  touch_runflag "$RUN"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
    --task_name long_term_forecast --is_training 0 \
    --model_id "$MID" --model timer_xl \
    --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE" \
    --seq_len 672 --input_token_len 24 --output_token_len 24 \
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1 \
    --test_seq_len 672 --test_pred_len 96 \
    --eval_target_only --target_channel 2 --n_vars 12 --seed 42 \
    $EVAL_FLAGS \
    --checkpoints "$RES" --test_dir "$TESTDIR" --test_file_name "$CKPT" \
    > "$LOG" 2>&1
  end_runflag "$RUN"
  echo "[`now`] EVAL  done  -> $MID"
}

# ---------- run ----------
for item in "${PLAN[@]}"; do
  # fields: DS MID train FLAGS  eval FLAGS
  DS=$(awk '{print $1}' <<<"$item")
  MID=$(awk '{print $2}' <<<"$item")
  TFLAGS=$(awk -F"train " '{print $2}' <<<"$item" | awk -F" eval " '{print $1}')
  EFLAGS=$(awk -F" eval " '{print $2}' <<<"$item")

  if [[ "$DS" == "ETTm1" ]]; then
    train_ettt "$MID" "$TFLAGS"
    eval_ettt  "$MID" "$EFLAGS"
  else
    train_weather "$MID" "$TFLAGS"
    eval_weather  "$MID" "$EFLAGS"
  fi
done

# ---------- summary ----------
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

echo
echo "=== 摘要（终端显示） ==="
awk -F, 'BEGIN{printf("%-30s %-8s %-12s %-12s\n","model","dataset","mse","mae")}
NR>1{printf("%-30s %-8s %-12s %-12s\n",$1,$2,$3,$4)}' "$OUT"

echo
echo "=== 各数据集 MSE 最优/次优 ==="
for ds in ETTm1 Weather; do
  echo ">> $ds"
  awk -F, -v target="$ds" 'NR>1 && $2==target && $3!="NA"{print $0}' "$OUT" \
  | sort -t, -k3,3g \
  | awk -F, '
    NR==1{printf("  1st: **%s**  mse=%s  mae=%s\n",$1,$3,$4)}
    NR==2{printf("  2nd: _%s_  mse=%s  mae=%s\n",$1,$3,$4)}
  '
done

echo
echo "CSV 写入: $OUT"
echo "[`now`] ALL DONE."
