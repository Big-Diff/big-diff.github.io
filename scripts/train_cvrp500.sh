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
  --project_name cvrp200_pairwise_partition \
  --wandb_logger_name cvrp200_from_cvrp100_kmax13_finetune \
  --do_train \
  --resume_weight_only \
  --ckpt_path /home/aiworker/code/Fast-T2T-main/data/cvrp/models/cvrp200_from_cvrp100_kmax13_finetune/checkpoints/epoch=29-step=9360.ckpt \
  --learning_rate 0.00003 \
  --weight_decay 0.0001 \
  --lr_scheduler cosine-decay \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/cvrp \
  --train_split train/cvrp500_train_10000_norm_solved_k64.npz \
  --validation_split val/cvrp500_val_1000_norm_solved_k64.npz \
  --test_split test/cvrp500_test_4000_norm_solved_k64.npz \
  --num_vehicles 64 \
  --batch_size 4 \
  --num_epochs 50 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --use_v2v \
  --v2v_every 2 \
  --v2v_heads 4 \
  --v2v_dropout 0.05 \
  --v2v_ffn_mult 2 \
  --use_n2n \
  --n2n_mode attn \
  --n2n_attn_heads 4 \
  --n2n_attn_dropout 0.05 \
  --n2n_attn_ffn_mult 2 \
  --dropout 0.05 \
  --validation_examples 128 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 1 \
  --alpha 0.5 \
  --consistency \
  --lam_pair 1.0 \
  --lam_row 0.1 \
  --lam_cons 0.01 \
  --pair_pos_samples 512 \
  --pair_neg_samples 512 \
  --refine_threads 2 \
  --eval_deterministic \
  --eval_seed 12345 \
  --ckpt_monitor val/cost_refined \
  --num_workers 8 \
  --check_val_every_n_epoch 1 \
  --route_two_opt \
  --two_opt_iter 128 \
  --read_use_pyvrp \
  --read_pyvrp_budget_ms 30 \
  --read_pyvrp_space_shortlist 96 \
  --read_pyvrp_min_k 16 \
  --read_pyvrp_max_k 64 \
  --read_pyvrp_score_alpha 1.0 \
  --read_projector_topk 10 \
  --read_projector_cum_prob 0.90 \
  --read_projector_lam_balance 0.10 \
  --read_projector_lam_compact 0.15 \
  --n2n_knn_k 32 \
  --no_use_k_predictor \
  --offline