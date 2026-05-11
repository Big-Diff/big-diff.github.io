#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train.py \
  --task cvrp \
  --fp16 \
  --project_name consistency_co \
  --wandb_logger_name cvrp_100_test \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/cvrp \
  --train_split train/cvrp100_test_all_solved_k13.npz \
  --validation_split val/cvrp100_val_128_solved_k13.npz \
  --test_split test/cvrp100_test_memmap \
  --num_vehicles 13 \
  --batch_size 64 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --use_v2v \
  --v2v_every 2 \
  --v2v_heads 4 \
  --v2v_dropout 0.05 \
  --v2v_ffn_mult 2 \
  --n2n_mode attn \
  --n2n_attn_heads 4 \
  --n2n_attn_dropout 0.05 \
  --n2n_attn_ffn_mult 2 \
  --dropout 0.05 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 4 \
  --alpha 0.5 \
  --consistency \
  --lam_pair 1.0 \
  --lam_row 0.1 \
  --lam_cons 0.01 \
  --pair_pos_samples 256 \
  --pair_neg_samples 256 \
  --refine_threads 8 \
  --eval_deterministic \
  --eval_seed 12345 \
  --ckpt_monitor val/cost_refined \
  --num_workers 8 \
  --check_val_every_n_epoch 1 \
  --two_opt_iter 256 \
  --read_pyvrp_budget_ms 500 \
  --read_pyvrp_space_shortlist 64 \
  --read_pyvrp_min_k 10 \
  --read_pyvrp_max_k 48\
  --read_projector_topk 8 \
  --read_projector_cum_prob 0.90 \
  --read_projector_lam_balance 0.10 \
  --read_projector_lam_compact 0.15 \
  --n2n_knn_k 24 \
  --no_use_k_predictor \
  --test_examples 1000 \
  --ckpt_path /home/aiworker/code/Fast-T2T-main/data/cvrp/models/cvrp100_kmax13_pairwise_rowanchor_pilot/checkpoints/epoch=90-step=103740.ckpt \
  --do_test \
  --report_time \
  --report_time_split test \
  --report_time_only_cost 1 \
  --offline