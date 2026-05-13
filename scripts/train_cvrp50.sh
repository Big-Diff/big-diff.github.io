#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

torchrun --standalone --nproc_per_node=2 train.py \
  --task cvrp \
  --fp16 \
  --project_name cvrp50_full_pilot \
  --wandb_logger_name cvrp50_release \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler cosine-decay \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/cvrp \
  --train_split train/50_with_solution.npz \
  --validation_split val/50_with_solution.npz \
  --test_split val/50_with_solution.npz \
  --num_vehicles 9 \
  --batch_size 128 \
  --num_epochs 5 \
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
  --n2n_attn_heads 4 \
  --n2n_attn_dropout 0.05 \
  --n2n_attn_ffn_mult 2 \
  --n2n_knn_k 16 \
  --validation_examples 128 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 1 \
  --consistency \
  --alpha 0.5 \
  --lam_pair 1.0 \
  --lam_row 0.1 \
  --lam_cons 0.01 \
  --pair_pos_samples 256 \
  --pair_neg_samples 256 \
  --refine_threads 2 \
  --eval_deterministic \
  --eval_seed 12345 \
  --ckpt_monitor val/cost_refined \
  --num_workers 8 \
  --read_pyvrp_budget_ms 20

