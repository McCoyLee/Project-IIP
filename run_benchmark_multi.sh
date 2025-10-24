#!/usr/bin/env bash
set -euo pipefail

# ========= 全局超参 =========
SEED=42
EPOCHS=50
DM=256; DFF=1024; HEADS=8; EL=3
BATCH=32; LR=1e-4
PREDS=(96 192 336 720)

COMMON="--seq_len 672 --input_token_len 24 --output_token_len 24 \
        --e_layers ${EL} --d_model ${DM} --d_ff ${DFF} --n_heads ${HEADS} \
        --dropout 0.1 --learning_rate ${LR} --batch_size ${BATCH} \
        --train_epochs ${EPOCHS} --patience 5 --cosine --tmax 50 --seed ${SEED} \
        --model timer_xl --task_name forecast --is_training 1"

# RevIN 两边统一开/关
USE_REVIN=1
if [[ "${USE_REVIN}" == "1" ]]; then
  COMMON="${COMMON} --revin --revin_affine --revin_eps 1e-5"
fi

# target-only（按需打开）
TARGET_ONLY=0
TARGET_CHANNEL=6
if [[ "${TARGET_ONLY}" == "1" ]]; then
  COMMON="${COMMON} --eval_target_only --target_channel ${TARGET_CHANNEL}"
fi

# 你的改进版（UniCA + AdaNorm + MoE）
IMPROVED_OPTS="--use_unica --unica_stage post --unica_fusion res_add \
               --unica_bottleneck 128 --unica_res_scale 0.01 \
               --use_adanorm --adanorm_alpha per_channel --adanorm_beta scalar \
               --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
               --moe_lb_alpha 0.003 --moe_gate_temp 1.0"

# ========= 数据集（name, root, csv, n_vars）一行一张卡 =========
DS_0=("ETTh1"       "$HOME/datasets/ETT"         "ETTh1.csv"          7)
DS_1=("ETTh2"       "$HOME/datasets/ETT"         "ETTh2.csv"          7)
DS_2=("ETTm1"       "$HOME/datasets/ETT"         "ETTm1.csv"          7)
DS_3=("ETTm2"       "$HOME/datasets/ETT"         "ETTm2.csv"          7)
DS_4=("Electricity" "$HOME/datasets/Electricity" "ECL.csv"            321)
DS_5=("Weather"     "$HOME/datasets/Weather"     "WTH.csv"            12)   # 13列=日期+12特征
DS_6=("Exchange"    "$HOME/datasets/ExchangeRate" "exchange_rate.csv"  8)
DS_7=("Solar"       "$HOME/datasets/Solar"       "solar_AL.csv"       137)

mkdir -p logs results checkpoints

run_one_dataset () {
  local gpu_id="$1"
  local data="$2"
  local root="$3"
  local dpath="$4"
  local nvars="$5"

  export CUDA_VISIBLE_DEVICES="${gpu_id}"

  for P in "${PREDS[@]}"; do
    # Baseline
    local mid_base="${data}_timerxl_base_p${P}"
    echo "[GPU ${gpu_id}] ${data} baseline p=${P} ..."
    python run.py ${COMMON} \
      --data "${data}" --root_path "${root}" --data_path "${dpath}" \
      --patch_size "${P}" --n_vars "${nvars}" \
      --model_id "${mid_base}" \
      --checkpoints "./checkpoints/${data}/${mid_base}" \
      | tee -a "logs/${mid_base}.log"

    # 改进版
    local mid_imp="${data}_unica_adanorm_moe_p${P}"
    echo "[GPU ${gpu_id}] ${data} improved p=${P} ..."
    python run.py ${COMMON} \
      --data "${data}" --root_path "${root}" --data_path "${dpath}" \
      --patch_size "${P}" --n_vars "${nvars}" \
      ${IMPROVED_OPTS} \
      --model_id "${mid_imp}" \
      --checkpoints "./checkpoints/${data}/${mid_imp}" \
      | tee -a "logs/${mid_imp}.log"
  done
  echo "[GPU ${gpu_id}] ${data} DONE."
}

# ========= 并行启动（8 张 GPU）=========
run_one_dataset 0 "${DS_0[@]}" &
run_one_dataset 1 "${DS_1[@]}" &
run_one_dataset 2 "${DS_2[@]}" &
run_one_dataset 3 "${DS_3[@]}" &
run_one_dataset 4 "${DS_4[@]}" &
run_one_dataset 5 "${DS_5[@]}" &
run_one_dataset 6 "${DS_6[@]}" &
run_one_dataset 7 "${DS_7[@]}" &

wait
echo "== ALL FINISHED =="
