#!/usr/bin/env bash
set -euo pipefail

# 可按需改 GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p logs checkpoints

# 4 个预测步长
PREDS=(96 192 336 720)

# 通用训练超参（保持你“正常指令”的风格）
COMMON="--task_name forecast --is_training 1 \
  --model timer_xl \
  --seq_len 672 --input_token_len 24 --output_token_len 24 \
  --e_layers 3 --d_model 256 --d_ff 1024 --n_heads 8 \
  --dropout 0.1 --learning_rate 1e-4 --batch_size 32 \
  --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 \
  --revin --revin_affine --revin_eps 1e-5 \
  --use_moe --num_experts 4 --moe_topk 1 \
  --moe_capacity_factor 1.1 \
  --moe_lb_alpha 0.003 \
  --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp \
  --moe_kl_alpha 1e-3 \
  --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
  --eval_target_only --target_channel 6 \
  --checkpoints ./checkpoints"

run_one () {
  local DATA=$1    # 数据集名（ETTh1 / ETTm1 / Electricity / Weather / Exchange / Solar）
  local DP=$2      # 文件名
  local ROOT=$3    # 根目录路径

  for P in "${PREDS[@]}"; do
    MID="${DATA}_revin_moe_lin_gate_p${P}"
    echo "===> Training ${MID}"
    python run.py \
      ${COMMON} \
      --model_id "${MID}" \
      --data "${DATA}" --root_path "${ROOT}" --data_path "${DP}" \
      | tee -a "logs/${MID}.log"
  done
}

# 逐个数据集开跑（路径按你当前目录结构）
run_one ETTh1       ETTh1.csv             ~/datasets/ETT
run_one ETTh2       ETTh2.csv             ~/datasets/ETT
run_one ETTm1       ETTm1.csv             ~/datasets/ETT
run_one ETTm2       ETTm2.csv             ~/datasets/ETT
run_one Electricity ECL.csv               ~/datasets/Electricity
run_one Weather     WTH.csv               ~/datasets/Weather
run_one Exchange    exchange_rate.csv     ~/datasets/ExchangeRate
run_one Solar       solar_AL.csv          ~/datasets/Solar
