#!/usr/bin/env bash
set -e
SESSION="ecl"; GPU="${1:-5}"   # 默认用 5 号卡，可传参改
PROJ="$HOME/OpenLTM"; LOGDIR="$PROJ/logs"; DATESTR="$(date +%F_%H-%M-%S)"
ALLOC_CONF="expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8"
tmux new-window -t "${SESSION}:" -n "gpu${GPU}_norevin" bash -lc "
  cd '$PROJ' && eval \"\$(conda shell.bash hook)\" && conda activate openltm && \
  export PYTORCH_CUDA_ALLOC_CONF='$ALLOC_CONF' && export CUDA_VISIBLE_DEVICES=$GPU && \
  stdbuf -oL -eL python -u run.py --task_name forecast --is_training 1 \
    --model_id ecl_no_revin_p112s24_tok112_h2_dm160_b2 --model timer_xl \
    --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
    --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 \
    --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
    --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
    --cosine --tmax 50 --n_vars 321 --flash_attention \
    --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 \
    --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
    --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
    --patch_size 0 --stride 0 \
    --eval_target_only --target_channel 0 \
    --checkpoints ./checkpoints/Electricity/ablation/ecl_no_revin_p112s24_tok112_h2_dm160_b2 \
  2>&1 | tee -a '$LOGDIR/'\"ecl_no_revin_p112s24_tok112_h2_dm160_b2\"'__'\"$DATESTR\"'.log'
"
