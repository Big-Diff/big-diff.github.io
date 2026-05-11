#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train.py \
  --task hfvrp \
  --fp16 \
  --project_name hfvrp50_full_pilot \
  --wandb_logger_name hfvrp50_newgnn_full_pilot \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler cosine-decay \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/hfvrp \
  --train_split train/hfvrp50_hfv_solved.npz \
  --validation_split val/hfvrp50_hfv_val_solved.npz \
  --test_split test/hfvrp50_hfv_test_solved.npz \
  --sparse_factor -1 \
  --hf_slot_order attribute \
  --batch_size 128 \
  --num_epochs 100 \
  --hidden_dim 192 \
  --gnn_layers 4 \
  --biattn_heads 4 \
  --assignment_backbone hf_lite_edgeupd \
  --graph_in_dim 10 \
  --dyn_refresh_every 2 \
  --intra_every 2 \
  --n2n_knn_k 16 \
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
  --read_use_pyvrp \
  --read_pyvrp_budget_ms 10 \
  --read_pyvrp_space_shortlist 48 \
  --read_pyvrp_min_k 8 \
  --read_pyvrp_max_k 24 \
  --read_projector_topk 6 \
  --read_projector_cum_prob 0.95 \
  --hf_log_cost_gap