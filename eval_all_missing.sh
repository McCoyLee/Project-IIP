#!/usr/bin/env bash
# Usage: bash eval_all_missing.sh [GPU_ID]
set -euo pipefail
GPU="${1:-0}"

ETT_ROOT="$HOME/datasets/ETT"
WTH_ROOT="$HOME/datasets/TSLib/Weather"
ETT_H_FILE="ETTh1.csv"
ETT_M_FILE="ETTm1.csv"
WTH_FILE="WTH.csv"

mkdir -p logs results

# ---------- helpers ----------
has_results_dir () { [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null || true)" ]]; }

has_flag () {
  # only add flags run.py actually supports
  python -u run.py -h 2>/dev/null | grep -q -- "$1"
}

latest_ckpt_in_dir () {
  local dir="$1"
  local ckpt="checkpoint.pth"
  if [[ -f "$dir/$ckpt" ]]; then echo "$ckpt"; return 0; fi
  local cand
  cand=$(ls -t "$dir"/*.pth 2>/dev/null | head -n1 || true)
  if [[ -n "${cand:-}" ]]; then basename "$cand"; else echo "$ckpt"; fi
}

eval_etth_like () {
  # $1=model_id, $2=data_name(ETTh1|ETTm1), $3=extra keywords
  local MID="$1"; local DATA="$2"; local EXTRA="$3"
  local FILE="$ETT_H_FILE"; [[ "$DATA" == "ETTm1" ]] && FILE="$ETT_M_FILE"
  local RDIR="./results/$MID"
  if ! has_results_dir "$RDIR"; then
    echo "[SKIP] $MID: no results dir"; return 0; fi

  echo "[EVAL] $MID on $DATA"
  local CMD=( python -u run.py
    --task_name forecast --is_training 0
    --model_id "$MID" --model timer_xl
    --data "$DATA" --root_path "$ETT_ROOT" --data_path "$FILE"
    --seq_len 672 --input_token_len 24 --input_token_stride 24 --output_token_len 24
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1
    --eval_target_only --target_channel 6 --num_workers 0
    --checkpoints "$RDIR"
  )
  has_flag "--revin"        && [[ "$EXTRA" == *"revin"*      ]] && CMD+=( --revin --revin_affine --revin_eps 1e-5 )
  has_flag "--use_unica"    && [[ "$EXTRA" == *"unica_res"*  ]] && CMD+=( --use_unica --unica_bottleneck 128 --unica_stage post --unica_res_scale 0.05 --unica_fusion res_add )
  has_flag "--use_unica"    && [[ "$EXTRA" == *"unica_film"* ]] && CMD+=( --use_unica --unica_bottleneck 128 --unica_stage post --unica_res_scale 0.05 --unica_fusion film_gate )
  has_flag "--use_adanorm"  && [[ "$EXTRA" == *"adanorm"*    ]] && CMD+=( --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 )
  has_flag "--use_desta"    && [[ "$EXTRA" == *"desta"*      ]] && CMD+=( --use_desta )
  has_flag "--use_moe"      && [[ "$EXTRA" == *"moe"*        ]] && CMD+=( --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2 )

  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "logs/${MID}.eval.log" 2>&1 \
    || echo "[WARN] $MID eval failed, see logs/${MID}.eval.log"
}

eval_weather_like () {
  # $1=model_id, $2=extra keywords
  local MID="$1"; local EXTRA="$2"; local RDIR="./results/$MID"
  if ! has_results_dir "$RDIR"; then echo "[SKIP] $MID: no results dir"; return 0; fi
  local TESTDIR; TESTDIR=$(ls -t "$RDIR" | head -n1)
  local CKPT;   CKPT=$(latest_ckpt_in_dir "$RDIR/$TESTDIR")
  echo "[EVAL] $MID on Weather (test_dir=$TESTDIR, ckpt=$CKPT)"

  local CMD=( python -u run.py
    --task_name long_term_forecast --is_training 0
    --model_id "$MID" --model timer_xl
    --data Weather --root_path "$WTH_ROOT" --data_path "$WTH_FILE"
    --seq_len 672 --input_token_len 24 --output_token_len 24
    --e_layers 3 --d_model 256 --n_heads 8 --d_ff 1024 --dropout 0.1
    --test_seq_len 672 --test_pred_len 96
    --eval_target_only --target_channel 2 --n_vars 12 --seed 42
    --checkpoints "$RDIR" --test_dir "$TESTDIR" --test_file_name "$CKPT"
  )
  has_flag "--revin"        && [[ "$EXTRA" == *"revin"*      ]] && CMD+=( --revin --revin_affine --revin_eps 1e-5 )
  has_flag "--use_unica"    && [[ "$EXTRA" == *"unica_res"*  ]] && CMD+=( --use_unica --unica_bottleneck 128 --unica_stage post --unica_res_scale 0.05 --unica_fusion res_add )
  has_flag "--use_unica"    && [[ "$EXTRA" == *"unica_film"* ]] && CMD+=( --use_unica --unica_bottleneck 128 --unica_stage post --unica_res_scale 0.05 --unica_fusion film_gate )
  has_flag "--use_adanorm"  && [[ "$EXTRA" == *"adanorm"*    ]] && CMD+=( --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar --ema_gamma 0.995 )
  has_flag "--use_desta"    && [[ "$EXTRA" == *"desta"*      ]] && CMD+=( --use_desta )
  has_flag "--use_moe"      && [[ "$EXTRA" == *"moe"*        ]] && CMD+=( --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 2.0 --moe_gate_temp 1.0 --moe_gate_noise_std 0.0 --moe_lb_alpha 1e-2 )

  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "logs/${MID}.eval.log" 2>&1 \
    || echo "[WARN] $MID eval failed, see logs/${MID}.eval.log"
}

# ---------- collect missing ----------
MISSING=()
# 1) 所有已有 train.log 但缺 eval.log 的
for t in logs/*train.log; do
  base="${t%.train.log}"
  [[ ! -f "${base}.eval.log" ]] && MISSING+=("$(basename "$base")")
done
# 2) 常见前缀的 train.log（兼容 etthm1_ 前缀）
for f in logs/etth1_timerxl_*.train.log logs/ettm1_timerxl_*.train.log logs/etthm1_timerxl_*.train.log logs/weather_timerxl_*.train.log logs/w_*.train.log; do
  [[ -e "$f" ]] || continue
  base="$(basename "${f%.train.log}")"
  [[ ! -f "logs/${base}.eval.log" ]] && MISSING+=("$base")
done
# 去重
MISSING=($(printf "%s\n" "${MISSING[@]}" | awk '!seen[$0]++'))

echo "待评测数量: ${#MISSING[@]}"
(IFS=$'\n'; printf '%s\n' "${MISSING[@]}")

# ---------- eval loop ----------
for mid in "${MISSING[@]}"; do
  case "$mid" in
    etth1_*)
      EXTRA=""
      [[ "$mid" == *"_unica_resadd"* || "$mid" == *"_unica_post"* ]] && EXTRA+=" unica_res"
      [[ "$mid" == *"_revin"*   ]] && EXTRA+=" revin"
      [[ "$mid" == *"_adanorm"* ]] && EXTRA+=" adanorm"
      [[ "$mid" == *"_ada_desta"* ]] && EXTRA+=" adanorm desta"
      [[ "$mid" == *"_moe"*     ]] && EXTRA+=" moe"
      eval_etth_like "$mid" "ETTh1" "$EXTRA"
      ;;
    etthm1_*|ettm1_*)
      EXTRA=""
      [[ "$mid" == *"_unica_resadd"* || "$mid" == *"_unica_post"* ]] && EXTRA+=" unica_res"
      [[ "$mid" == *"_revin"*   ]] && EXTRA+=" revin"
      [[ "$mid" == *"_adanorm"* ]] && EXTRA+=" adanorm"
      [[ "$mid" == *"_ada_desta"* ]] && EXTRA+=" adanorm desta"
      [[ "$mid" == *"_moe"*     ]] && EXTRA+=" moe"
      eval_etth_like "$mid" "ETTm1" "$EXTRA"
      ;;
    weather_*|w_*)
      EXTRA=""
      [[ "$mid" == *"_unica_resadd"* || "$mid" == *"_unica_post"* || "$mid" == "w_unica" ]] && EXTRA+=" unica_film"
      [[ "$mid" == *"_revin"* || "$mid" == "w_unica_revin" || "$mid" == "w_ur" ]] && EXTRA+=" revin"
      [[ "$mid" == *"_adanorm"* || "$mid" == *"_ada_desta"* ]] && EXTRA+=" adanorm"
      [[ "$mid" == *"_ada_desta"* ]] && EXTRA+=" desta"
      [[ "$mid" == *"_moe"* || "$mid" == "w_moe" || "$mid" == "w_moe_lb" || "$mid" == "weather_moe_safe" ]] && EXTRA+=" moe"
      eval_weather_like "$mid" "$EXTRA"
      ;;
    *)
      echo "[SKIP] Unknown prefix for $mid"
      ;;
  esac
done

# ---------- summary ----------
OUT="results/summary.csv"
echo "model,dataset,mse,mae" > "$OUT"
for e in logs/*eval.log; do
  base=$(basename "$e" .eval.log)
  ds="UNKNOWN"
  [[ "$base" == etth1_* ]] && ds="ETTh1"
  [[ "$base" == ettm1_* || "$base" == etthm1_* ]] && ds="ETTm1"
  [[ "$base" == weather_* || "$base" == w_* ]] && ds="Weather"
  mse=$(egrep -io 'mse[^0-9]*[0-9.]+' "$e" | tail -n1 | grep -oE '[0-9.]+' || echo NA)
  mae=$(egrep -io 'mae[^0-9]*[0-9.]+' "$e" | tail -n1 | grep -oE '[0-9.]+' || echo NA)
  echo "$base,$ds,$mse,$mae" >> "$OUT"
done

awk -F, '
BEGIN{printf("model,dataset,mse,mae\n")}
NR>1{print $0}
' "$OUT"

echo
echo "=== 各数据集 MSE 最优/次优 ==="
for ds in ETTh1 ETTm1 Weather; do
  echo ">> $ds"
  awk -F, -v target="$ds" 'NR>1 && $2==target && $3!="NA"{print $0}' "$OUT" \
  | sort -t, -k3,3g \
  | awk -F, '
    NR==1{printf("  1st: **%s**  mse=%s  mae=%s\n",$1,$3,$4)}
    NR==2{printf("  2nd: _%s_  mse=%s  mae=%s\n",$1,$3,$4)}
  '
done

echo "CSV 写入: $OUT"
