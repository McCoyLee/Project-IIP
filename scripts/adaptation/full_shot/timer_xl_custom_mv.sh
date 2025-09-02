set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate openltm
echo Using python: $(which python)
/home/limaocheng/conda/envs/openltm/bin/python - <<'PY'
import sys,torch; print('PY:', sys.executable); print('TORCH:', torch.__version__)
PY
export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
token_num=30
token_len=96
seq_len=$[$token_num*$token_len]

/home/limaocheng/conda/envs/openltm/bin/python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/CustomPV \
  --data_path custom_multivariate.csv \
  --covariate \
  --n_vars 6 \
  --model_id ETTh1_full_shot \
  --model timer_xl \
  --data MultivariateDatasetBenchmark  \
  --seq_len 2880 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 8 \
  --n_heads 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 32 \
  --learning_rate 5e-6 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --adaptation \
  --pretrain_model_path ./checkpoints/timer_xl/checkpoint.pth