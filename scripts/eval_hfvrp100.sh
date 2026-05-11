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
  --wandb_logger_name hfvrp100_competitive_gate \
  --do_test \
  --ckpt_path /home/aiworker/code/Fast-T2T-main/models/hfvrp100/epoch=64-step=76115.ckpt \
  --storage_path /home/aiworker/code/Fast-T2T-main/data/hfvrp \
  --train_split train/hfvrp100_train_solved.npz \
  --validation_split val/hfvrp100_hfv_val_solved.npz \
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
  --read_use_pyvrp \
  --read_pyvrp_budget_ms 1000 \
  --read_pyvrp_num_neighbours 50 \
  --read_pyvrp_symmetric_proximity \
  --read_pyvrp_neigh_mode heat \
  --read_heat_max_neigh 50 \
  --read_heat_base_geo_k 28 \
  --read_heat_geo_core_k 28 \
  --read_heat_geo_pool_k 50 \
  --read_heat_geom_shortlist 96 \
  --read_heat_tier_k 12 \
  --read_heat_tier_k_low 16 \
  --read_heat_route_cover_top_slots 5 \
  --read_heat_route_cover_top_slots_low 6 \
  --read_heat_route_cover_per_slot 2 \
  --read_heat_slot_sim_weight 0.0 \
  --read_heat_tier_sim_weight 0.25 \
  --read_heat_dist_penalty 1.0 \
  \
  --read_heat_global_conf_adapt \
  --read_heat_global_tier_p1_min 0.72 \
  --read_heat_global_tier_margin_min 0.10 \
  --read_heat_global_low_frac_max 0.35 \
  \
  --hf_log_cost_gap \
  --test_examples 1000 \
  --report_time \
  --report_time_split test \
  --report_time_only_cost 1 \
  --offline


