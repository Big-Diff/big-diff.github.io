#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train.py \
  --task hfvrp \
  --fp16 \
  --project_name hfvrp \
  --wandb_logger_name hfvrp_50_test \
  --do_test \
  --ckpt_path /home/aiworker/code/Fast-T2T-main/models/hfvrp50/epoch=13-step=16394.ckpt \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/hfvrp \
  --train_split train/hfvrp50_hfv_solved.npz \
  --validation_split val/hfvrp50_hfv_val_solved.npz \
  --test_split test/hfvrp50_hfv_test_solved.npz \
  --sparse_factor -1 \
  --hf_slot_order attribute \
  --batch_size 64 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --assignment_backbone hf_lite_edgeupd \
  --graph_in_dim 10 \
  --dyn_refresh_every 2 \
  --intra_every 2 \
  --n2n_knn_k 24 \
  --inference_schedule cosine \
  --inference_diffusion_steps 1 \
  --parallel_sampling 1 \
  --consistency \
  --alpha 0.5 \
  --xt_jitter 0.0 \
  --hf_lam_type 0.0 \
  --hf_lam_pair 1.0 \
  --hf_lam_row 0.1 \
  --hf_lam_cons 0.0 \
  --hf_pair_pos_samples 512 \
  --hf_pair_neg_samples 512 \
  --refine_threads 8 \
  --eval_cost_every 8 \
  --eval_cost_batches 100000 \
  --eval_deterministic \
  --eval_seed 12345 \
  --num_workers 8 \
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
  --read_pyvrp_budget_ms 1000 \
  --hf_log_cost_gap \
  --test_examples 128 \
  --report_time \
  --report_time_split test \
  --report_time_only_cost 1 \
  --offline