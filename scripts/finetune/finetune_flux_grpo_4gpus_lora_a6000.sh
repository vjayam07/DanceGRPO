#!/bin/bash
#SBATCH --job-name=danceGRPO-finetune
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --gres=gpu:a6000ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_finetune_%j.out
#SBATCH --error=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_finetune_%j.err

# =============================================================================
# DanceGRPO FLUX LoRA — 4x A6000 Ada (48GB) baseline for cfgrl-expo comparison
# =============================================================================
#
# Adapted from finetune_flux_grpo_8gpus_lora.sh with the following changes:
#   - 4 GPUs instead of 8  (nproc_per_node=4)
#   - 512x512 resolution   (matches cfgrl-expo FLUX setup)
#   - gradient_accumulation_steps = num_generations = 12
#   - train_batch_size=1 / train_sp_batch_size=1 for memory safety on 48GB
#   - max_train_steps=300   (sufficient to see reward convergence)
#   - checkpointing_steps=50
#
# Effective batch math (should match the 8-GPU script):
#   8-GPU:  8 GPUs × batch_size 2 × grad_accum 12 / sp_size 1 = 192
#   4-GPU:  4 GPUs × batch_size 1 × grad_accum 12 / sp_size 1 =  48
#   (grad_accum must be <= num_generations for the optimizer to step)
#
# Reward: HPSv2 (same as cfgrl-expo)
# Prompts: HPDv2 (./assets/prompts.txt — same source as cfgrl-expo)
# =============================================================================

set -euo pipefail

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# Uncomment to disable wandb:
# export WANDB_DISABLED=true

# --- Storage paths (all saving/logging goes to atlas2) ------------------------
ATLAS_BASE="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
OUTPUT_DIR="${ATLAS_BASE}/data/outputs/grpo_baseline_4gpu_a6000"
DATA_DIR="${ATLAS_BASE}/data"
CACHE_DIR="${ATLAS_BASE}/data/.cache"
HPS_CKPT_DIR="${ATLAS_BASE}/hps_ckpt"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}/rl_embeddings"
mkdir -p "${CACHE_DIR}"
mkdir -p "${HPS_CKPT_DIR}"

# --- One-time setup (skip if already done) -----------------------------------
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
  --pretrained_model_name_or_path "${DATA_DIR}/flux" \
  --vae_model_path "${DATA_DIR}/flux" \
  --cache_dir "${CACHE_DIR}" \
  --data_json_path "${DATA_DIR}/rl_embeddings/videos2caption.json" \
  --gradient_checkpointing \
  --train_batch_size 1 \
  --num_latent_t 1 \
  --sp_size 1 \
  --train_sp_batch_size 1 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 12 \
  --max_train_steps 1000 \
  --learning_rate 3e-4 \
  --mixed_precision bf16 \
  --checkpointing_steps 50 \
  --allow_tf32 \
  --cfg 0.0 \
  --output_dir "${OUTPUT_DIR}" \
  --h 512 \
  --w 512 \
  --t 1 \
  --sampling_steps 8 \
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
  --lora_rank 128 \
  --hps_ckpt_dir "${HPS_CKPT_DIR}"
