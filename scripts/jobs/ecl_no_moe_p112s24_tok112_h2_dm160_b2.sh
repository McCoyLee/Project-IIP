#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8'
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd "/home/limaocheng/OpenLTM"

# 激活 conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate openltm

ts=$(date +%F_%H-%M-%S)
log="/home/limaocheng/OpenLTM/logs/ecl_no_moe_p112s24_tok112_h2_dm160_b2__${ts}.log"
echo "[$(date)] GPU=4 任务：ecl_no_moe_p112s24_tok112_h2_dm160_b2" | tee -a "$log"
stdbuf -oL -eL python -u ./run.py --task_name forecast --is_training 1 --model_id ecl_no_moe_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path /home/limaocheng/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_moe_p112s24_tok112_h2_dm160_b2 2>&1 | tee -a "$log"
