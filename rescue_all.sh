#!/usr/bin/env bash
set -euo pipefail

SESSION="ecl"
PROJ="$HOME/OpenLTM"
LOGDIR="$PROJ/logs"
JOBDIR="$PROJ/scripts/jobs"
DATA_ROOT="$HOME/datasets/Electricity"   # ✅ 用你的数据路径
ALLOC_CONF="expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8"

mkdir -p "$LOGDIR" "$JOBDIR"

# 简单健检
[ -f "$PROJ/run.py" ] || { echo "❌ $PROJ/run.py 不存在"; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "❌ 数据目录不存在：$DATA_ROOT"; exit 1; }

make_job() {
  local name="$1" gpu="$2" args="$3"
  local job="$JOBDIR/${name}.sh"
  cat > "$job" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='$ALLOC_CONF'
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd "$PROJ"

# 激活 conda
if command -v conda >/dev/null 2>&1; then
  eval "\$(conda shell.bash hook)"
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate openltm

ts=\$(date +%F_%H-%M-%S)
log="$LOGDIR/${name}__\${ts}.log"
echo "[\$(date)] GPU=$gpu 任务：$name" | tee -a "\$log"
stdbuf -oL -eL python -u ./run.py $args 2>&1 | tee -a "\$log"
EOF2
  chmod +x "$job"
}

# ====== 六个任务（root_path 改为 $DATA_ROOT）======
make_job "ecl_baseline_tok112_h2_dm160_b2" 1 \
"--task_name forecast --is_training 1 --model_id ecl_baseline_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --n_vars 321 --flash_attention --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_baseline_tok112_h2_dm160_b2"

make_job "ecl_our_full_p112s24_tok112_h2_dm160_b2" 2 \
"--task_name forecast --is_training 1 --model_id ecl_our_full_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --moe_lb_alpha 0.003 --moe_kl_alpha 1e-3 --moe_gate_temp 1.3 --moe_gate_noise_std 0.05 --moe_learnable_temp --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_our_full_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_unica_p112s24_tok112_h2_dm160_b2" 3 \
"--task_name forecast --is_training 1 --model_id ecl_no_unica_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_unica_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_moe_p112s24_tok112_h2_dm160_b2" 4 \
"--task_name forecast --is_training 1 --model_id ecl_no_moe_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_moe_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_linear_p112s24_tok112_h2_dm160_b2" 6 \
"--task_name forecast --is_training 1 --model_id ecl_no_linear_p112s24_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 24 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_linear_p112s24_tok112_h2_dm160_b2"

make_job "ecl_no_patch_p112s112_tok112_h2_dm160_b2" 7 \
"--task_name forecast --is_training 1 --model_id ecl_no_patch_p112s112_tok112_h2_dm160_b2 --model timer_xl --data Electricity --root_path $DATA_ROOT --data_path ECL.csv --seq_len 672 --input_token_len 112 --input_token_stride 112 --output_token_len 24 --e_layers 3 --d_model 160 --n_heads 2 --d_ff 640 --dropout 0.1 --learning_rate 1e-4 --batch_size 2 --train_epochs 50 --patience 5 --seed 42 --cosine --tmax 50 --revin --revin_affine --revin_eps 1e-5 --n_vars 321 --flash_attention --use_unica --unica_stage post --unica_fusion res_add --unica_bottleneck 128 --unica_res_scale 0.01 --use_moe --num_experts 4 --moe_topk 1 --moe_capacity_factor 1.1 --use_linear_branch --linear_res_scale 0.2 --linear_init_gate -2.0 --patch_size 0 --stride 0 --eval_target_only --target_channel 0 --checkpoints ./checkpoints/Electricity/ablation/ecl_no_patch_p112s112_tok112_h2_dm160_b2"

# ====== tmux 会话：先开 keepalive，防止窗口失败导致会话退出 ======
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" -n keepalive "bash -lc 'while true; do sleep 3600; done'"

# 创建各任务窗口
tmux new-window -t "$SESSION:" -n gpu1_baseline "bash '$JOBDIR/ecl_baseline_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION:" -n gpu2_full     "bash '$JOBDIR/ecl_our_full_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION:" -n gpu3_nounica  "bash '$JOBDIR/ecl_no_unica_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION:" -n gpu4_nomoe    "bash '$JOBDIR/ecl_no_moe_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION:" -n gpu6_nolinear "bash '$JOBDIR/ecl_no_linear_p112s24_tok112_h2_dm160_b2.sh'"
tmux new-window -t "$SESSION:" -n gpu7_nopatch  "bash '$JOBDIR/ecl_no_patch_p112s112_tok112_h2_dm160_b2.sh'"

tmux select-window -t "$SESSION:gpu1_baseline"
echo "✅ rescue: 会话/窗口已创建。使用： tmux attach -t $SESSION"
