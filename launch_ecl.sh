#!/usr/bin/env bash
set -euo pipefail

SESSION="ecl"
PROJ="$HOME/OpenLTM"
LOGDIR="$PROJ/logs"
DATESTR="$(date +%F_%H-%M-%S)"
mkdir -p "$LOGDIR"

# —— 显存分配优化：可扩展分段 + 合理切片大小 + 垃圾回收阈值 ——
ALLOC_CONF="expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8"

new_win() {
  local win="$1" gpu="$2" title="$3" cmd="$4"
  tmux new-window -t "${SESSION}:" -n "${win}" bash -lc "
    cd '$PROJ' && \
    { command -v conda >/dev/null 2>&1 && eval \"\$(conda shell.bash hook)\"; } || true && \
    conda activate openltm && \
    export PYTORCH_CUDA_ALLOC_CONF='$ALLOC_CONF' && \
    export CUDA_VISIBLE_DEVICES=$gpu && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    mkdir -p '$LOGDIR' && \
    echo '[$(date)] GPU='$gpu' 任务：$title' && \
    stdbuf -oL -eL python -u $cmd 2>&1 | tee -a '$LOGDIR/'\"${title}\"'__'\"$DATESTR\"'.log'
  "
}

# 创建/复用 tmux 会话
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n "stub" bash -lc "echo 'Session $SESSION started'; sleep 1"
fi

# 1️⃣ GPU-1 : baseline
new_win "gpu1_baseline" 1 "ecl_baseline_tok112_h2_dm160_b2" "
run.py \
  --task_name forecast --is_training 1 --model_id ecl_baseline_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --n_vars 321 --flash_attention \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_baseline_tok112_h2_dm160_b2
"

# 2️⃣ GPU-2 : our_full
new_win "gpu2_full" 2 "ecl_our_full_p112s24_tok112_h2_dm160_b2" "
run.py \
  --task_name forecast --is_training 1 --model_id ecl_our_full_p112s24_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention \
  --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
  --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp \
  --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_our_full_p112s24_tok112_h2_dm160_b2
"

# 3️⃣ GPU-3 : w/o UniCA
new_win "gpu3_nounica" 3 "ecl_no_unica_p112s24_tok112_h2_dm160_b2" "
run.py --task_name forecast --is_training 1 \
  --model_id ecl_no_unica_p112s24_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
  --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_no_unica_p112s24_tok112_h2_dm160_b2
"

# 4️⃣ GPU-4 : w/o MoE
new_win "gpu4_nomoE" 4 "ecl_no_moe_p112s24_tok112_h2_dm160_b2" "
run.py --task_name forecast --is_training 1 \
  --model_id ecl_no_moe_p112s24_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention \
  --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 \
  --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_no_moe_p112s24_tok112_h2_dm160_b2
"

# 6️⃣ GPU-6 : w/o Linear
new_win "gpu6_nolinear" 6 "ecl_no_linear_p112s24_tok112_h2_dm160_b2" "
run.py --task_name forecast --is_training 1 \
  --model_id ecl_no_linear_p112s24_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention \
  --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_no_linear_p112s24_tok112_h2_dm160_b2
"

# 7️⃣ GPU-7 : w/o Patch（内存更友好）
new_win "gpu7_nopatch" 7 "ecl_no_patch_p112s112_tok112_h2_dm160_b2" "
run.py --task_name forecast --is_training 1 \
  --model_id ecl_no_patch_p112s112_tok112_h2_dm160_b2 --model timer_xl \
  --data Electricity --root_path ~/datasets/Electricity --data_path ECL.csv \
  --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 \
  --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 \
  --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 \
  --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention \
  --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 \
  --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 \
  --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 \
  --patch_size 0 --stride 0 \
  --eval_target_only --target_channel 0 \
  --checkpoints ./checkpoints/Electricity/ablation/ecl_no_patch_p112s112_tok112_h2_dm160_b2
"

# 可选：w/o RevIN（等有空闲卡再开，或把上面某个替换到它）
cat > "$LOGDIR/_manual__ecl_no_revin.sh" <<'EOF'
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
EOF
chmod +x "$LOGDIR/_manual__ecl_no_revin.sh"

# 清理占位窗口并列出窗口
tmux kill-window -t "${SESSION}:stub" 2>/dev/null || true
tmux list-windows -t "$SESSION"
echo
echo "✅ 已在 tmux 会话: $SESSION 中启动 6 个训练。"
echo "👉 查看： tmux attach -t $SESSION"
echo "👉 分屏切换：Ctrl+b 然后按窗口编号（0..）或 p/n"
echo "👉 单看日志： tail -f $LOGDIR/<对应名称>__${DATESTR}.log"
echo "👉 额外启动 w/o RevIN： $LOGDIR/_manual__ecl_no_revin.sh  <GPU号，默认5>"
