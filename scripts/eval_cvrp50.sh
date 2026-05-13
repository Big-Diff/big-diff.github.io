#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train.py \
  --task cvrp \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/cvrp \
  --train_split train/50_with_solution.npz \
  --validation_split val/50_with_solution.npz \
  --test_split test/cvrp50_test_1000_norm_solved.npz \
  --num_vehicles 9 \
  --batch_size 64 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --dropout 0.05 \
  --use_v2v \
  --v2v_every 2 \
  --v2v_heads 4 \
  --v2v_dropout 0.05 \
  --v2v_ffn_mult 2 \
  --n2n_mode attn \
  --n2n_knn_k 16 \
  --n2n_attn_heads 4 \
  --n2n_attn_dropout 0.05 \
  --n2n_attn_ffn_mult 2 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 1 \
  --consistency \
  --alpha 0.5 \
  --refine_threads 8 \
  --eval_deterministic \
  --eval_seed 12345 \
  --num_workers 8 \
  --read_pyvrp_budget_ms 500 \
  --test_examples 128 \
  --ckpt_path /home/aiworker/code/Fast-T2T-main/data/cvrp/models/cvrp_o3.12_v4_train/checkpoints/epoch=31-step=74976.ckpt \
  --do_test \
  --report_time \
  --report_time_split test \
  --report_time_only_cost 1 \
  --offline