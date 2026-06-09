#!/bin/bash
#SBATCH --account=m5319
#SBATCH --job-name=grpo-flux-full-4step
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=4
#SBATCH --output=/pscratch/sd/v/vjayam/DanceGRPO/slurm_logs/grpo_flux_full_4step_%j.out
#SBATCH --error=/pscratch/sd/v/vjayam/DanceGRPO/slurm_logs/grpo_flux_full_4step_%j.err
#SBATCH --reservation=baselines_cfgrl_expo

# =============================================================================
# DanceGRPO FLUX full-model GRPO - 4x A100 (80GB)
# =============================================================================
#
# Matches finetune_flux_grpo_4gpus_a6000.sh:
#   - 4 GPUs, 512x512 images, 4 sampling steps, and seed 42
#   - batch size 1, 12 gradient accumulation steps, and 12 generations
#   - HPSv2 reward and full-model FSDP training without LoRA or EMA
#
# With timestep_fraction=0.6, each four-step rollout trains one randomly
# selected transition from the first three transitions.
# =============================================================================

set -euo pipefail

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# Uncomment to disable wandb:
# export WANDB_DISABLED=true

SCRATCH_BASE="/pscratch/sd/v/vjayam/DanceGRPO"
OUTPUT_DIR="${SCRATCH_BASE}/data/outputs/grpo_full_4gpu_a100_4step"
DATA_DIR="${SCRATCH_BASE}/data"
CACHE_DIR="${SCRATCH_BASE}/data/.cache"
HPS_CKPT_DIR="${SCRATCH_BASE}/hps_ckpt"
EVAL_STEPS="${EVAL_STEPS:-50}"
NUM_EVAL_PROMPTS="${NUM_EVAL_PROMPTS:-16}"
NUM_EVAL_IMAGES="${NUM_EVAL_IMAGES:-6}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}/rl_embeddings"
mkdir -p "${CACHE_DIR}"
mkdir -p "${HPS_CKPT_DIR}"
mkdir -p "${SCRATCH_BASE}/slurm_logs"

for checkpoint in open_clip_pytorch_model.bin HPS_v2.1_compressed.pt; do
  if [ ! -f "${HPS_CKPT_DIR}/${checkpoint}" ]; then
    echo "Missing HPSv2 checkpoint: ${HPS_CKPT_DIR}/${checkpoint}" >&2
    exit 1
  fi
done

torchrun --nproc_per_node=4 --master_port 19002 \
  fastvideo/train_grpo_flux.py \
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
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --master_weight_type bf16 \
  --fsdp_sharding_startegy full \
  --checkpointing_steps 50 \
  --eval_steps "${EVAL_STEPS}" \
  --num_eval_prompts "${NUM_EVAL_PROMPTS}" \
  --num_eval_images "${NUM_EVAL_IMAGES}" \
  --eval_seed 42 \
  --allow_tf32 \
  --cfg 0.0 \
  --output_dir "${OUTPUT_DIR}" \
  --h 512 \
  --w 512 \
  --t 1 \
  --sampling_steps 4 \
  --eta 0.3 \
  --lr_warmup_steps 0 \
  --sampler_seed 1223627 \
  --max_grad_norm 0.01 \
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
  --hps_ckpt_dir "${HPS_CKPT_DIR}"
