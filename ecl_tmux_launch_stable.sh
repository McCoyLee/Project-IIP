#!/usr/bin/env bash
set -euo pipefail

# ==== 基本路径 ====
PROJ="$HOME/OpenLTM"
LOGDIR="$PROJ/logs"
JOBDIR="$PROJ/scripts/jobs"
SESSION="ecl"

mkdir -p "$LOGDIR" "$JOBDIR"

# ==== 显存分配（OOM 友好） ====
ALLOC_CONF="expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8"

# ==== 写出单个 job 脚本 ====
make_job() {
  local name="$1" gpu="$2" args="$3"
  local job="$JOBDIR/${name}.sh"
  cat > "$job" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='$ALLOC_CONF'
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$PROJ"

# 激活 conda (兼容不同安装路径)
if command -v conda >/dev/null 2>&1; then
  eval "\$(conda shell.bash hook)"
else
  [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "\$HOME/miniconda3/etc/profile.d/conda.sh" || true
  [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "\$HOME/anaconda3/etc/profile.d/conda.sh" || true
fi
conda activate openltm

ts=\$(date +%F_%H-%M-%S)
log="$LOGDIR/${name}__\${ts}.log"

echo "[\$(date)] GPU=$gpu 任务：$name" | tee -a "\$log"
stdbuf -oL -eL python -u ./run.py $args 2>&1 | tee -a "\$log"
EOF
  chmod +x "$job"
}

# ==== 各实验参数（全部一行，最稳） ====
make_job "ecl_baseline_tok112_h2_dm160_b2" 1 \
"--task_name forecast --is_training 1 --model_id ecl_baseline_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --n_vars 321 --flash_attention --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_baseline_tok112_h2_dm160_b2"

make_job "ecl_our_full_p112s24_tok112_h2_dm160_b2" 2 \
"--task_name forecast --is_training 1 --model_id ecl_our_full_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_our_full_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_unica_p112s24_tok112_h2_dm160_b2" 3 \
"--task_name forecast --is_training 1 --model_id ecl_no_unica_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_unica_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_moe_p112s24_tok112_h2_dm160_b2" 4 \
"--task_name forecast --is_training 1 --model_id ecl_no_moe_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_moe_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_linear_p112s24_tok112_h2_dm160_b2" 6 \
"--task_name forecast --is_training 1 --model_id ecl_no_linear_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_linear_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_patch_p112s112_tok112_h2_dm160_b2" 7 \
"--task_name forecast --is_training 1 --model_id ecl_no_patch_p112s112_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $PROJ/datasets/Electricity --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_patch_p112s112_tok112_h2_dm160_b2"

# ==== 启动 tmux ====
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n "gpu1_baseline" "bash '$JOBDIR/ecl_baseline_tok112_h2_dm160_b2.sh'"
else
  tmux new-window -t "$SESSION" -n "gpu1_baseline" "bash '$JOBDIR/ecl_baseline_tok112_h2_dm160_b2.sh'"
fi

tmux new-window -t "$SESSION" -n "gpu2_full"     "bash '$JOBDIR/ecl_our_full_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION" -n "gpu3_nounica"  "bash '$JOBDIR/ecl_no_unica_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION" -n "gpu4_nomoe"    "bash '$JOBDIR/ecl_no_moe_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION" -n "gpu6_nolinear" "bash '$JOBDIR/ecl_no_linear_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION" -n "gpu7_nopatch"  "bash '$JOBDIR/ecl_no_patch_p112s112_tok112_h2_dm160_b2.sh'"

tmux select-window -t "$SESSION:0"

echo "✅ 已启动 tmux 会话: $SESSION"
echo "👉 进入会话： tmux attach -t $SESSION"
echo "👉 日志目录： $LOGDIR"
