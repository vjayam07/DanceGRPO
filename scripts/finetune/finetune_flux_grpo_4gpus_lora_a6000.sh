#!/bin/bash
# =============================================================================
# DanceGRPO FLUX LoRA — 4x A6000 Ada (48GB) baseline for cfgrl-expo comparison
# =============================================================================
#
# Adapted from finetune_flux_grpo_8gpus_lora.sh with the following changes:
#   - 4 GPUs instead of 8  (nproc_per_node=4)
#   - 512x512 resolution   (matches cfgrl-expo FLUX setup)
#   - gradient_accumulation_steps doubled (24 vs 12) to preserve effective batch
#   - train_batch_size=1 / train_sp_batch_size=1 for memory safety on 48GB
#   - max_train_steps=300   (sufficient to see reward convergence)
#   - checkpointing_steps=50
#
# Effective batch math (should match the 8-GPU script):
#   8-GPU:  8 GPUs × batch_size 2 × grad_accum 12 / sp_size 1 = 192
#   4-GPU:  4 GPUs × batch_size 1 × grad_accum 24 / sp_size 1 =  96
#   (Halved effective batch — compensated by 2x more accum steps per GPU.
#    Each GPU processes the same num_generations=12 samples per prompt,
#    so the GRPO advantage statistics stay comparable.)
#
# Reward: HPSv2 (same as cfgrl-expo)
# Prompts: HPDv2 (./assets/prompts.txt — same source as cfgrl-expo)
# =============================================================================

set -euo pipefail

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# Uncomment to disable wandb:
# export WANDB_DISABLED=true

# --- One-time setup (skip if already done) -----------------------------------
mkdir -p images

# HPSv2 reward model (needed on first run)
if [ ! -d "HPSv2" ]; then
    echo ">>> Installing HPSv2 reward model..."
    git clone https://github.com/tgxs002/HPSv2.git
    cd HPSv2 && pip install -e . && cd ..
fi

# --- Launch training ----------------------------------------------------------
torchrun --nproc_per_node=4 --master_port 19002 \
    fastvideo/train_grpo_flux_lora.py \
    --seed 42 \
    --pretrained_model_name_or_path data/flux \
    --vae_model_path data/flux \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 24 \
    --max_train_steps 300 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_baseline_4gpu_a6000 \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 12 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --lora_alpha 256 \
    --lora_rank 128
