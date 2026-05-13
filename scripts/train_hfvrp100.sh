#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

torchrun --standalone --nproc_per_node=2 train.py \
  --task hfvrp \
  --fp16 \
  --project_name hfvrp100 \
  --wandb_logger_name hfvrp_100_loss \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler cosine-decay \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/hfvrp \
  --train_split train/hfvrp100_train_solved.npz \
  --validation_split val/hfvrp100_hfv_val_solved.npz \
  --test_split test/hfvrp100_hfv_test_solved.npz \
  --sparse_factor -1 \
  --hf_slot_order attribute \
  --batch_size 64 \
  --num_epochs 100 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --assignment_backbone hf_lite_edgeupd \
  --graph_in_dim 10 \
  --dyn_refresh_every 2 \
  --intra_every 2 \
  --n2n_knn_k 24 \
  --validation_examples 256 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 1 \
  --consistency \
  --alpha 0.5 \
  --xt_jitter 0.0 \
  --hf_lam_type 1.0 \
  --hf_lam_pair 1.0 \
  --hf_lam_row 0.10 \
  --hf_lam_cons 0.01 \
  --hf_pair_pos_samples 256 \
  --hf_pair_neg_samples 256 \
  --refine_threads 8 \
  --eval_cost_every 1 \
  --eval_cost_batches 1 \
  --eval_deterministic \
  --eval_seed 12345 \
  --ckpt_monitor val/cost_refined \
  --num_workers 8 \
  --check_val_every_n_epoch 1 \
  --use_n2n \
  --use_global \
  --no_use_adaln \
  --use_v2v \
  --v2v_every 2 \
  --v2v_heads 4 \
  --v2v_dropout 0.05 \
  --v2v_ffn_mult 2 \
  --n2n_mode attn \
  --n2n_attn_heads 4 \
  --n2n_attn_dropout 0.05 \
  --n2n_attn_ffn_mult 2 \
  --read_pyvrp_budget_ms 10 \
  --hf_log_cost_gap